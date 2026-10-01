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

from .frames import DEX1_OPEN

log = logging.getLogger("g1d_wla.recap")
IMG_HW = (224, 224)


# ------------------------------------------------------------------ épisodes

#: pas restants prédits sous lesquels la tâche est considérée finie (étiquette positive)
DONE_STEPS = 5

def episode_dirs(roots) -> list[Path]:
    out = []
    for r in roots:
        r = Path(r)
        if (r / "data.json").exists():
            out.append(r.resolve())
        else:
            out.extend(sorted(p.resolve() for p in r.glob("episode_*") if (p / "data.json").exists()))
    return out      # chemins ABSOLUS : clé stable du pli de chaque épisode (audit du 1/10)


def load(ep: Path) -> dict:
    return json.loads((ep / "data.json").read_text())


def torso_pitch_reader(doc: dict):
    """Lecture du tangage du buste d'un épisode : ``info.body_layout`` (constante ou indice), sinon 13."""
    lay = doc.get("info", {}).get("body_layout", {}) or {}
    if lay.get("torso_pitch_const") is not None:
        c = float(lay["torso_pitch_const"])
        return lambda body: c
    idx = lay.get("torso_pitch") if lay.get("torso_pitch") is not None else 13
    return lambda body: float(body[idx]) if len(body) > idx else 0.0


def state_vector(step: dict, pitch=lambda body: float(body[13])) -> np.ndarray:
    st = step["states"]
    body = st.get("body", {}).get("qpos") or [0.0] * 35
    return np.asarray(st["left_arm"]["qpos"] + st["right_arm"]["qpos"]
                      + [st["left_ee"]["qpos"][0] / DEX1_OPEN, st["right_ee"]["qpos"][0] / DEX1_OPEN, pitch(body)],
                      np.float32)


def episode_outcome(doc: dict, is_demo: bool = False) -> tuple[str, int | None]:
    """(« success » | « failure » | « unknown », pas de réussite). Une démo sans en-tête d'issue
    (enregistrement réel xr_teleoperate) est une réussite à son dernier pas."""
    info = doc["info"]
    out, s = info.get("outcome"), info.get("success_step")
    if out is None and is_demo:
        return "success", len(doc["data"]) - 1 if s is None else s
    if out == "success" and s is not None:
        return "success", int(s)
    if out == "failure":
        return "failure", None
    return "unknown", None


def steps_to_go(doc: dict, horizon: int, is_demo: bool = False) -> np.ndarray | None:
    """Cible par pas : pas restants jusqu'à la réussite / horizon (0 après, 1 si échec) ; None si
    l'issue est inconnue (épisode ignoré). L'horizon doit dépasser la durée des épisodes corrigés
    (~400 pas en sim), sinon leurs premiers pas ont la même cible qu'un échec."""
    n = len(doc["data"])
    out, s = episode_outcome(doc, is_demo)
    if out == "unknown":
        return None
    if out == "failure":
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


def build_dataset(eps: list[Path], enc: Encoder, horizon: int, demo_set: set):
    F, S, Y, E, kept = [], [], [], [], []
    for ep in eps:
        doc = load(ep)
        y = steps_to_go(doc, horizon, is_demo=ep in demo_set)
        if y is None:
            log.warning("%s : issue inconnue (pas de X/Y en fin d'essai ?) -> ignoré", ep)
            continue
        F.append(episode_features(ep, doc, enc))
        S.append(np.stack([state_vector(st, torso_pitch_reader(doc)) for st in doc["data"]]))
        Y.append(y)
        E.append(np.full(len(doc["data"]), len(kept)))
        kept.append(ep)
    return np.concatenate(F), np.concatenate(S), np.concatenate(Y), np.concatenate(E), kept


def _fit(F, S, Y, idx, s_mean, s_std, a, rng, val_idx=None, tag=""):
    dev = a.device
    tF, tS, tY = (torch.from_numpy(x).to(dev) for x in (F, (S - s_mean) / s_std, Y))
    model = ValueHead(F.shape[1], S.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    for epoch in range(a.epochs):
        model.train()
        perm = rng.permutation(idx)
        for i in range(0, len(perm), 256):
            b = torch.from_numpy(perm[i:i + 256]).to(dev)
            loss = nn.functional.mse_loss(model(tF[b], tS[b]), tY[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    err = None
    if val_idx is not None and len(val_idx):
        model.eval()
        with torch.no_grad():
            vb = torch.from_numpy(val_idx).to(dev)
            err = (model(tF[vb], tS[vb]) - tY[vb]).abs().mean().item() * a.horizon
        log.info("%s perte finale %.4f | épisodes tenus hors entraînement : erreur moyenne %.1f pas", tag, loss.item(), err)
    return model, err


def cmd_train_value(a) -> None:
    """Entraînement en K PLIS (cross-fitting) : chaque épisode est ensuite étiqueté par le modèle qui
    ne l'a PAS vu. Sinon un modèle ajusté sur ses propres données donne une progression « parfaite »
    sur tout épisode réussi et l'étiquetage devient trivial (presque tout positif)."""
    demo_set = set(episode_dirs(a.demos)) if a.demos else set()
    eps = episode_dirs(list(a.episodes) + list(a.demos or []))
    if not eps:
        raise SystemExit("aucun épisode")
    enc = Encoder(a.device)
    F, S, Y, E, eps = build_dataset(eps, enc, a.horizon, demo_set)
    outcomes = [episode_outcome(load(e), e in demo_set)[0] for e in eps]
    log.info("%d épisodes (%d réussis, %d ratés), %d frames, horizon %d", len(eps), outcomes.count("success"),
             outcomes.count("failure"), len(Y), a.horizon)
    rng = np.random.default_rng(0)
    s_mean, s_std = S.mean(0), S.std(0) + 1e-6
    k = max(2, min(a.folds, len(eps)))
    fold_of = rng.permutation(np.arange(len(eps)) % k)
    models, errs = [], []
    for f in range(k):
        tr, va = np.where(fold_of[E] != f)[0], np.where(fold_of[E] == f)[0]
        m, err = _fit(F, S, Y, tr, s_mean, s_std, a, rng, va, tag=f"pli {f + 1}/{k} :")
        models.append(m.state_dict())
        errs.append(err)
    log.info("validation croisée : erreur moyenne %.1f pas (horizon %d)", float(np.mean(errs)), a.horizon)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"folds": models, "fold_of": {str(e): int(fold_of[i]) for i, e in enumerate(eps)},
                "s_mean": s_mean, "s_std": s_std, "horizon": a.horizon, "d_img": F.shape[1], "d_state": S.shape[1],
                "cv_error_steps": float(np.mean(errs))}, out)
    log.info("modèle de valeur écrit : %s (%d plis)", out, k)


def load_value(path: str, device: str):
    """-> (liste des modèles par pli, checkpoint)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    models = []
    for sd in ck["folds"]:
        m = ValueHead(ck["d_img"], ck["d_state"]).to(device).eval()
        m.load_state_dict(sd)
        models.append(m)
    return models, ck


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
    models, ck = load_value(a.value, a.device)
    H = ck["horizon"]
    n_pos = n_neg = n_int = 0
    for ep in eps:
        doc = load(ep)
        n = len(doc["data"])
        f = torch.from_numpy(episode_features(ep, doc, enc)).to(a.device)
        s = np.stack([state_vector(st, torso_pitch_reader(doc)) for st in doc["data"]])
        s = torch.from_numpy((s - ck["s_mean"]) / ck["s_std"]).float().to(a.device)
        # modèle du pli qui n'a PAS vu cet épisode ; épisode nouveau : moyenne des plis
        fold = ck["fold_of"].get(str(ep))
        if fold is None and str(ep) in {str(Path(k).resolve()) for k in ck["fold_of"]}:
            fold = {str(Path(k).resolve()): v for k, v in ck["fold_of"].items()}[str(ep)]
        if fold is None and any(Path(k).name == ep.name and Path(k).parent.name == ep.parent.name for k in ck["fold_of"]):
            log.warning("%s : vu à l'entraînement de la valeur mais pli introuvable (chemin changé ?) -> "
                        "moyenne de tous les plis, PAS hors pli", ep)
        use = [models[fold]] if fold is not None else models
        with torch.no_grad():
            v = np.mean([m(f, s).cpu().numpy() for m in use], axis=0) * H   # pas restants prédits
        inter = np.array([float(st.get("intervention", 0)) for st in doc["data"]])
        # étiquette PAR PAS sur la fenêtre [t, t + chunk) : c'est le morceau d'action que le
        # modèle prédit à partir du pas t (et pas un bloc fixe aligné sur 0)
        t1 = np.minimum(np.arange(n) + a.chunk, n - 1)
        nominal = t1 - np.arange(n)
        progress = v - v[t1]
        # on ne peut pas gagner plus de pas qu'il n'en reste : le gain attendu est plafonné par v[t].
        # Sans ce plafond, la fin de tâche (lâcher, poser) et la tenue étaient étiquetées négatives
        # (audit du 1/10). Tâche quasi finie (moins de DONE_STEPS pas restants prédits) : positif.
        ideal = np.minimum(nominal, np.maximum(v, 0.0))
        adv = ((ideal < DONE_STEPS) | (progress >= a.threshold * ideal)).astype(np.float32)
        adv[nominal == 0] = adv[max(0, n - 2)] if n > 1 else 1.0
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
    t.add_argument("--demos", nargs="*", default=None,
                   help="dossiers de DÉMOS sans en-tête d'issue (réel) : comptées réussies à leur dernier pas")
    t.add_argument("--out", required=True)
    t.add_argument("--horizon", type=int, default=500,
                   help="doit dépasser la durée des épisodes corrigés (max-steps + expert)")
    t.add_argument("--folds", type=int, default=5, help="validation croisée : chaque épisode étiqueté hors pli")
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
