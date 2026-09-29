"""RECAP / Delta-0 pour WLA : modèle de valeur et étiquetage de l'avantage par morceau d'action.

Boucle d'amélioration (docs/G1D_Constats.md §15) :

1. politique de départ par imitation (fine-tuning WLA) ;
2. rollouts : la politique joue, un opérateur corrige (``sim/recap_rollouts.py`` en sim, téléop en réel) ;
3. MODÈLE DE VALEUR (ici) : prédit le nombre de pas restant jusqu'à la réussite à partir des images et
   de l'état ; échec = valeur maximale ;
4. ÉTIQUETAGE (ici) : chaque morceau d'action (``--chunk`` pas) est positif s'il fait progresser la
   valeur d'au moins ``--threshold`` × la progression nominale, négatif sinon ; les corrections de
   l'opérateur sont positives ; l'étiquette est écrite par pas (``advantage`` : 1 / 0) dans data.json ;
5. conversion (``g1d_wla.convert_teleop``, colonnes advantage/intervention) puis fine-tuning avec
   ``advantage_key`` dans la config de données : le prompt reçoit « Advantage: positive|negative ».
   À l'exécution, le serveur demande « positive » (``--advantage positive``).

Le modèle de valeur : ResNet18 ImageNet GELÉ sur la tête (``color_0``) et le poignet droit (dernière
image), caractéristiques mises en cache, + état (bras, pinces, buste) -> MLP -> nombre de pas restant
normalisé par ``--horizon`` (1 = échec / très loin, 0 = réussi).

Usage (env uv du projet) ::

    .venv/bin/python -m g1d_wla.recap train-value --episodes playground/sim_raw/sim_novares_n10 \\
        playground/sim_raw/recap_novares_r1 --out playground/recap/value_r1.pt
    .venv/bin/python -m g1d_wla.recap label --value playground/recap/value_r1.pt \\
        --episodes playground/sim_raw/recap_novares_r1
    .venv/bin/python -m g1d_wla.recap label --all-positive --episodes playground/sim_raw/sim_novares_n10
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("g1d_wla.recap")
IMG_HW = (224, 224)


# ------------------------------------------------------------------ épisodes
def episode_dirs(roots) -> list[Path]:
    out = []
    for r in roots:
        r = Path(r)
        if (r / "data.json").exists():
            out.append(r)
        else:
            out.extend(sorted(p for p in r.glob("episode_*") if (p / "data.json").exists()))
    return out


def load(ep: Path) -> dict:
    return json.loads((ep / "data.json").read_text())


def state_vector(step: dict) -> np.ndarray:
    st = step["states"]
    body = st.get("body", {}).get("qpos") or [0.0] * 35
    return np.asarray(st["left_arm"]["qpos"] + st["right_arm"]["qpos"]
                      + [st["left_ee"]["qpos"][0] / 5.4, st["right_ee"]["qpos"][0] / 5.4, body[13]], np.float32)


def steps_to_go(doc: dict, horizon: int) -> np.ndarray:
    """Cible par pas : pas restants jusqu'à la réussite / horizon (0 après la réussite, 1 si échec)."""
    n = len(doc["data"])
    info = doc["info"]
    s = info.get("success_step")
    if info.get("outcome", "success" if s is not None else "failure") != "success" or s is None:
        return np.ones(n, np.float32)
    t = np.arange(n)
    return np.clip((s - t) / horizon, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ caractéristiques images
class Encoder:
    """ResNet18 ImageNet gelé -> 512 par image ; cache .npy par épisode (``features_r18.npy``)."""

    def __init__(self, device: str) -> None:
        import torchvision
        self.device = device
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.m = m.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, paths: list[Path], batch: int = 64) -> np.ndarray:
        from PIL import Image
        out = []
        for i in range(0, len(paths), batch):
            ims = [np.asarray(Image.open(p).convert("RGB").resize(IMG_HW[::-1])) for p in paths[i:i + batch]]
            x = torch.from_numpy(np.stack(ims)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            out.append(self.m((x - self.mean) / self.std).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def episode_features(ep: Path, doc: dict, enc: Encoder) -> np.ndarray:
    """(T, 1024) : tête + poignet droit ; mis en cache à côté de l'épisode."""
    cache = ep / "features_r18.npy"
    n = len(doc["data"])
    if cache.exists():
        f = np.load(cache)
        if len(f) == n:
            return f
    keys = sorted(doc["data"][0]["colors"])
    head, wrist_r = keys[0], keys[-1]
    fh = enc([ep / s["colors"][head] for s in doc["data"]])
    fw = enc([ep / s["colors"][wrist_r] for s in doc["data"]])
    f = np.concatenate([fh, fw], axis=1)
    np.save(cache, f)
    return f


# ------------------------------------------------------------------ modèle de valeur
class ValueHead(nn.Module):
    def __init__(self, d_img: int = 1024, d_state: int = 17) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_img + d_state, 512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, f: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([f, s], dim=-1))).squeeze(-1)


def build_dataset(eps: list[Path], enc: Encoder, horizon: int):
    F, S, Y, E = [], [], [], []
    for k, ep in enumerate(eps):
        doc = load(ep)
        F.append(episode_features(ep, doc, enc))
        S.append(np.stack([state_vector(st) for st in doc["data"]]))
        Y.append(steps_to_go(doc, horizon))
        E.append(np.full(len(doc["data"]), k))
    return (np.concatenate(F), np.concatenate(S), np.concatenate(Y), np.concatenate(E))


def cmd_train_value(a) -> None:
    eps = episode_dirs(a.episodes)
    if not eps:
        raise SystemExit("aucun épisode")
    enc = Encoder(a.device)
    F, S, Y, E = build_dataset(eps, enc, a.horizon)
    n_fail = sum(load(e)["info"].get("outcome") == "failure" for e in eps)
    log.info("%d épisodes (%d échecs), %d frames", len(eps), n_fail, len(Y))
    rng = np.random.default_rng(0)
    val_eps = set(rng.choice(len(eps), max(1, len(eps) // 10), replace=False).tolist())
    va = np.isin(E, list(val_eps))
    s_mean, s_std = S.mean(0), S.std(0) + 1e-6
    dev = a.device
    tF, tS, tY = (torch.from_numpy(x).to(dev) for x in (F, (S - s_mean) / s_std, Y))
    tr_idx, va_idx = np.where(~va)[0], np.where(va)[0]
    model = ValueHead(F.shape[1], S.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    for epoch in range(a.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 256):
            b = torch.from_numpy(perm[i:i + 256]).to(dev)
            loss = nn.functional.mse_loss(model(tF[b], tS[b]), tY[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % max(1, a.epochs // 5) == 0 or epoch == a.epochs - 1:
            model.eval()
            with torch.no_grad():
                vb = torch.from_numpy(va_idx).to(dev)
                err = (model(tF[vb], tS[vb]) - tY[vb]).abs().mean().item() * a.horizon
            log.info("époque %d : perte %.4f | validation : erreur moyenne %.1f pas", epoch, loss.item(), err)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "s_mean": s_mean, "s_std": s_std, "horizon": a.horizon,
                "d_img": F.shape[1], "d_state": S.shape[1], "episodes": [str(e) for e in eps]}, out)
    log.info("modèle de valeur écrit : %s", out)


def load_value(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ValueHead(ck["d_img"], ck["d_state"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


# ------------------------------------------------------------------ étiquetage
def cmd_label(a) -> None:
    eps = episode_dirs(a.episodes)
    if a.all_positive:
        for ep in eps:
            doc = load(ep)
            for st in doc["data"]:
                st["advantage"] = 1.0
            (ep / "data.json").write_text(json.dumps(doc))
        log.info("%d épisodes étiquetés entièrement positifs (démos)", len(eps))
        return
    if not a.value:
        raise SystemExit("--value requis (ou --all-positive)")
    enc = Encoder(a.device)
    model, ck = load_value(a.value, a.device)
    H = ck["horizon"]
    n_pos = n_neg = n_int = 0
    for ep in eps:
        doc = load(ep)
        n = len(doc["data"])
        f = torch.from_numpy(episode_features(ep, doc, enc)).to(a.device)
        s = np.stack([state_vector(st) for st in doc["data"]])
        s = torch.from_numpy((s - ck["s_mean"]) / ck["s_std"]).float().to(a.device)
        with torch.no_grad():
            v = model(f, s).cpu().numpy() * H              # pas restants prédits
        inter = np.array([float(st.get("intervention", 0)) for st in doc["data"]])
        adv = np.zeros(n, np.float32)
        for t0 in range(0, n, a.chunk):
            t1 = min(t0 + a.chunk, n - 1)
            progress = v[t0] - v[t1]                       # pas gagnés sur ce morceau
            nominal = t1 - t0
            adv[t0:t0 + a.chunk] = 1.0 if nominal > 0 and progress >= a.threshold * nominal else 0.0
        adv[inter > 0.5] = 1.0                             # corrections de l'opérateur : positives
        for st, x in zip(doc["data"], adv):
            st["advantage"] = float(x)
        doc["info"]["advantage_labeling"] = {"value": str(a.value), "threshold": a.threshold, "chunk": a.chunk}
        (ep / "data.json").write_text(json.dumps(doc))
        pol = inter < 0.5
        n_pos += int((adv[pol] > 0.5).sum())
        n_neg += int((adv[pol] < 0.5).sum())
        n_int += int((~pol).sum())
        log.info("%s : %s, politique %d pas (%d positifs), opérateur %d pas", ep.name,
                 doc["info"].get("outcome"), int(pol.sum()), int((adv[pol] > 0.5).sum()), int((~pol).sum()))
    tot = max(1, n_pos + n_neg)
    log.info("pas de la politique : %d positifs (%.0f %%), %d négatifs | pas de l'opérateur (positifs) : %d",
             n_pos, 100 * n_pos / tot, n_neg, n_int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-value", help="entraîner le modèle de valeur")
    t.add_argument("--episodes", nargs="+", required=True, help="dossiers d'épisodes (démos + rollouts)")
    t.add_argument("--out", required=True)
    t.add_argument("--horizon", type=int, default=300)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    lb = sub.add_parser("label", help="étiqueter l'avantage par morceau d'action (écrit dans data.json)")
    lb.add_argument("--episodes", nargs="+", required=True)
    lb.add_argument("--value", default=None, help="modèle de valeur (train-value)")
    lb.add_argument("--all-positive", dest="all_positive", action="store_true", help="démos : tout positif")
    lb.add_argument("--chunk", type=int, default=30)
    lb.add_argument("--threshold", type=float, default=0.5,
                    help="positif si le morceau gagne au moins threshold × sa durée en pas restants")
    lb.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    {"train-value": cmd_train_value, "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    main()
