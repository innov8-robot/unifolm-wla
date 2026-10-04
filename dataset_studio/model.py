"""Couche données du studio (sans Qt) : un dossier de tâche xr_teleoperate = des ``episode_XXXX/``
contenant ``data.json`` et ``colors/*.jpg``.

Toute modification destructive passe par la CORBEILLE ``<tâche>/.corbeille/`` (ignorée par
l'enregistreur et le convertisseur, qui ne regardent que les dossiers ``episode_*`` du niveau
supérieur) : un épisode supprimé y est DÉPLACÉ, un rognage y range l'ancien ``data.json`` et les
images retirées. Après une suppression, les épisodes restants sont renumérotés sans trou à partir
de 0000 (renommage en deux temps : aucun écrasement possible).

Découpage en sous-tâches : ``<tâche>/tags.json`` liste les étiquettes (nom, consigne du modèle) et
chaque ``episode_XXXX/segments.json`` ses segments ``{tag, start, end}`` (pas inclus). Le fichier
suit l'épisode quand il est renuméroté ou mis à la corbeille ; un rognage décale les segments.
``export_segments`` fabrique, à côté de la tâche, un dossier de tâche ordinaire dont chaque épisode
est un segment (images en liens physiques, consigne de l'étiquette) : il se convertit comme les autres.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EP_RE = re.compile(r"^episode_(\d+)$")
TAG_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
TRASH = ".corbeille"
TAGS_FILE = "tags.json"
SEGMENTS_FILE = "segments.json"

#: libellés des images selon la disposition (mêmes conventions que g1d_wla.convert_teleop)
CAMERA_LABELS = {
    "binocular": {"color_0": "Tête — œil gauche", "color_1": "Tête — œil droit",
                  "color_2": "Poignet gauche", "color_3": "Poignet droit"},
    "mono": {"color_0": "Tête", "color_1": "Poignet gauche", "color_2": "Poignet droit"},
    "right-only": {"color_0": "Tête — œil gauche", "color_1": "Tête — œil droit", "color_2": "Poignet droit"},
}


def guess_layout(doc: dict) -> str:
    n = len(doc["data"][0]["colors"]) if doc["data"] else 0
    if n == 4:
        return "binocular"
    if n == 3 and str(doc.get("info", {}).get("source", "")).startswith("sim/"):
        return "mono"
    return "right-only" if n == 3 else "binocular"


@dataclass
class EpisodeSummary:
    name: str
    path: Path
    n: int = 0
    seconds: float = 0.0
    goal: str = ""
    outcome: str = ""
    layout: str = ""
    left_gripper_used: bool = False
    right_gripper_used: bool = False
    base_moved: bool = False
    column_range: tuple | None = None
    missing_images: int = 0
    error: str = ""
    flags: list = field(default_factory=list)


def load_doc(ep: Path) -> dict:
    return json.loads((ep / "data.json").read_text())


def _write_json(path: Path, obj, indent=None) -> None:
    """Écriture ATOMIQUE d'un fichier JSON."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
    os.replace(tmp, path)


def save_doc(ep: Path, doc: dict) -> None:
    _write_json(ep / "data.json", doc)


def load_segments(ep: Path) -> list[dict]:
    f = ep / SEGMENTS_FILE
    if not f.exists():
        return []
    try:
        segs = json.loads(f.read_text()).get("segments", [])
    except Exception:
        return []
    return sorted((s for s in segs if {"tag", "start", "end"} <= set(s)), key=lambda s: (s["start"], s["end"]))


def save_segments(ep: Path, segs: list[dict]) -> None:
    segs = sorted(({**s, "start": int(s["start"]), "end": int(s["end"])} for s in segs),
                  key=lambda s: (s["start"], s["end"]))
    if segs:
        _write_json(ep / SEGMENTS_FILE, {"segments": segs}, indent=1)
    elif (ep / SEGMENTS_FILE).exists():
        (ep / SEGMENTS_FILE).unlink()


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)                      # même disque : aucune place en plus
    except OSError:
        shutil.copy2(src, dst)


def summarize(ep: Path, fps: float = 30.0, check_images: bool = False) -> EpisodeSummary:
    s = EpisodeSummary(ep.name, ep)
    try:
        doc = load_doc(ep)
    except Exception as e:                     # épisode interrompu (JSON non fermé) ou illisible
        s.error = f"data.json illisible : {e}"
        s.flags.append("ILLISIBLE")
        return s
    data = doc.get("data", [])
    s.n = len(data)
    s.seconds = s.n / fps
    s.goal = doc.get("text", {}).get("goal", "")
    s.outcome = str(doc.get("info", {}).get("outcome", "") or "")
    if not data:
        s.flags.append("VIDE")
        return s
    s.layout = guess_layout(doc)
    gl = [st["states"].get("left_ee", {}).get("qpos", [5.4])[0] for st in data]
    gr = [st["states"].get("right_ee", {}).get("qpos", [5.4])[0] for st in data]
    s.left_gripper_used = (max(gl) - min(gl)) > 0.5
    s.right_gripper_used = (max(gr) - min(gr)) > 0.5
    base = [st["actions"].get("body", {}).get("qpos") or [0, 0, 0] for st in data]
    s.base_moved = any(abs(b[0]) > 1e-6 or abs(b[2]) > 1e-6 for b in base if len(b) >= 3)
    col = [st["states"]["column"]["qpos"][0] for st in data if "column" in st["states"]]
    if col:
        s.column_range = (min(col), max(col))
    if check_images:
        s.missing_images = sum(1 for st in data for p in st["colors"].values() if not (ep / p).exists())
    if s.base_moved:
        s.flags.append("base")
    if s.column_range and s.column_range[1] - s.column_range[0] > 0.005:
        s.flags.append("colonne")
    if s.missing_images:
        s.flags.append(f"{s.missing_images} img manquantes")
    if s.n < 30:
        s.flags.append("< 1 s")
    return s


def signals(doc: dict) -> dict:
    """Séries temporelles d'un épisode, par groupe -> {nom: tableau (T,)}."""
    data = doc["data"]

    def arr(get):
        out = []
        for st in data:
            try:
                out.append(get(st))
            except (KeyError, IndexError, TypeError):
                out.append(np.nan)
        return np.asarray(out, float)

    g = {}
    for side, fr in (("left", "gauche"), ("right", "droit")):
        q = np.array([st["states"][f"{side}_arm"]["qpos"] for st in data], float)
        a = np.array([st["actions"][f"{side}_arm"]["qpos"] for st in data], float)
        g[f"Bras {fr} — état"] = {f"q{i}": q[:, i] for i in range(q.shape[1])}
        g[f"Bras {fr} — consigne"] = {f"q{i}": a[:, i] for i in range(a.shape[1])}
    g["Pinces (Dex1, 0 fermée → 5,4)"] = {
        "gauche état": arr(lambda st: st["states"]["left_ee"]["qpos"][0]),
        "gauche consigne": arr(lambda st: st["actions"]["left_ee"]["qpos"][0]),
        "droite état": arr(lambda st: st["states"]["right_ee"]["qpos"][0]),
        "droite consigne": arr(lambda st: st["actions"]["right_ee"]["qpos"][0]),
    }
    lay = doc.get("info", {}).get("body_layout") or {}
    yi = lay.get("torso_yaw")
    if yi is not None:
        g["Buste — rotation (rad)"] = {"mesure": arr(lambda st: st["states"]["body"]["qpos"][yi])}
    if any("column" in st["states"] for st in data):
        g["Colonne"] = {"hauteur (m)": arr(lambda st: st["states"]["column"]["qpos"][0]),
                        "consigne vitesse": arr(lambda st: st["actions"]["column"]["qpos"][0])}
    g["Base roulante"] = {"vx (m/s)": arr(lambda st: st["actions"]["body"]["qpos"][0]),
                          "vyaw (rad/s)": arr(lambda st: st["actions"]["body"]["qpos"][2])}
    if any("intervention" in st for st in data):
        g["Intervention / avantage"] = {"intervention": arr(lambda st: st.get("intervention", np.nan)),
                                        "avantage": arr(lambda st: st.get("advantage", np.nan))}
    return g


class TaskDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)

    # ------------------------------------------------------------------ lecture
    def episodes(self) -> list[Path]:
        eps = [p for p in self.path.iterdir() if p.is_dir() and EP_RE.match(p.name)]
        return sorted(eps, key=lambda p: int(EP_RE.match(p.name).group(1)))

    def trash_dir(self) -> Path:
        return self.path / TRASH

    def trashed(self) -> list[Path]:
        t = self.trash_dir()
        return sorted((p for p in t.iterdir() if p.is_dir() and "__suppr__" in p.name), reverse=True) if t.exists() else []

    # ------------------------------------------------------------------ suppression / renumérotation
    def delete(self, names: list[str]) -> list[Path]:
        """Déplace les épisodes dans la corbeille puis renumérote le reste. Rend les chemins en corbeille."""
        t = self.trash_dir()
        t.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        moved = []
        for n in names:
            src = self.path / n
            if not src.is_dir():
                continue
            dst = t / f"{stamp}__suppr__{n}"
            k = 1
            while dst.exists():
                dst = t / f"{stamp}__suppr__{n}_{k}"
                k += 1
            shutil.move(str(src), str(dst))
            moved.append(dst)
        self.renumber()
        return moved

    def renumber(self, start: int = 0) -> dict[str, str]:
        """episode_* -> episode_0000.. contigus (ordre conservé). Renommage en deux temps."""
        eps = self.episodes()
        plan = {p.name: f"episode_{i + start:04d}" for i, p in enumerate(eps)}
        if all(a == b for a, b in plan.items()):
            return {}
        tmp = {}
        for p in eps:
            t = p.with_name(f".renum_{p.name}")
            p.rename(t)
            tmp[p.name] = t
        for old, new in plan.items():
            tmp[old].rename(self.path / new)
        return {a: b for a, b in plan.items() if a != b}

    def restore(self, trashed: Path) -> str:
        """Remet un épisode de la corbeille à la FIN du dataset (nouveau numéro)."""
        eps = self.episodes()
        nxt = (int(EP_RE.match(eps[-1].name).group(1)) + 1) if eps else 0
        name = f"episode_{nxt:04d}"
        shutil.move(str(trashed), str(self.path / name))
        return name

    def empty_trash(self) -> None:
        shutil.rmtree(self.trash_dir(), ignore_errors=True)

    # ------------------------------------------------------------------ modifications d'un épisode
    def trim(self, name: str, start: int, end: int) -> int:
        """Garde les pas [start, end] (inclus). Ancien data.json et images retirées -> corbeille.
        Rend le nouveau nombre de pas."""
        ep = self.path / name
        doc = load_doc(ep)
        n = len(doc["data"])
        start, end = max(0, int(start)), min(n - 1, int(end))
        if start == 0 and end == n - 1:
            return n
        if end < start:
            raise ValueError("fin avant début")
        bak = self.trash_dir() / f"{time.strftime('%Y%m%d_%H%M%S')}__rognage__{name}"
        (bak / "colors").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ep / "data.json", bak / "data.json")
        kept = doc["data"][start:end + 1]
        keep_paths = {p for st in kept for p in st["colors"].values()}
        for st in doc["data"][:start] + doc["data"][end + 1:]:
            for p in st["colors"].values():
                if p not in keep_paths and (ep / p).exists():
                    shutil.move(str(ep / p), str(bak / p))
        for k, st in enumerate(kept):
            st["idx"] = k
        info = doc.setdefault("info", {})
        for key in ("success_step", "takeover_step"):
            v = info.get(key)
            if isinstance(v, int):
                info[key] = (v - start) if start <= v <= end else None
        info["trimmed"] = {"from": name, "start": start, "end": end, "backup": bak.name}
        doc["data"] = kept
        save_doc(ep, doc)
        segs = load_segments(ep)
        if segs:                                # segments décalés et recoupés ; l'ancien fichier est sauvegardé
            shutil.copy2(ep / SEGMENTS_FILE, bak / SEGMENTS_FILE)
            save_segments(ep, [{**s, "start": max(s["start"], start) - start, "end": min(s["end"], end) - start}
                               for s in segs if s["end"] >= start and s["start"] <= end])
        return len(kept)

    # ------------------------------------------------------------------ étiquettes et segments
    def tags(self) -> list[dict]:
        """[{name, instruction}] dans l'ordre de création (touches 1 à 9 du studio)."""
        f = self.path / TAGS_FILE
        if not f.exists():
            return []
        try:
            return [t for t in json.loads(f.read_text()).get("tags", []) if "name" in t]
        except Exception:
            return []

    def save_tags(self, tags: list[dict]) -> None:
        _write_json(self.path / TAGS_FILE, {"tags": tags}, indent=1)

    def add_tag(self, name: str, instruction: str) -> None:
        if not TAG_RE.match(name):
            raise ValueError("nom d'étiquette : lettres, chiffres, _ et - (40 au plus), sans espace")
        tags = self.tags()
        if any(t["name"] == name for t in tags):
            raise ValueError(f"l'étiquette « {name} » existe déjà")
        self.save_tags(tags + [{"name": name, "instruction": instruction.strip()}])

    def set_tag_instruction(self, name: str, instruction: str) -> None:
        self.save_tags([{**t, "instruction": instruction.strip()} if t["name"] == name else t for t in self.tags()])

    def tag_usage(self) -> dict[str, tuple[int, int]]:
        """{étiquette: (nombre de segments, nombre de pas)} sur tout le dataset."""
        out: dict[str, list[int]] = {}
        for ep in self.episodes():
            for s in load_segments(ep):
                u = out.setdefault(s["tag"], [0, 0])
                u[0] += 1
                u[1] += s["end"] - s["start"] + 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def remove_tag(self, name: str, with_segments: bool = False) -> int:
        """Supprime l'étiquette ; avec ``with_segments``, aussi ses segments dans tous les épisodes.
        Rend le nombre de segments supprimés."""
        n = self.tag_usage().get(name, (0, 0))[0]
        if n and not with_segments:
            raise ValueError(f"l'étiquette « {name} » est utilisée par {n} segment(s)")
        if n:
            for ep in self.episodes():
                segs = load_segments(ep)
                if any(s["tag"] == name for s in segs):
                    save_segments(ep, [s for s in segs if s["tag"] != name])
        self.save_tags([t for t in self.tags() if t["name"] != name])
        return n

    def rename_tag(self, old: str, new: str) -> None:
        """Renomme l'étiquette et tous ses segments."""
        if not TAG_RE.match(new):
            raise ValueError("nom d'étiquette : lettres, chiffres, _ et - (40 au plus), sans espace")
        tags = self.tags()
        if any(t["name"] == new for t in tags):
            raise ValueError(f"l'étiquette « {new} » existe déjà")
        self.save_tags([{**t, "name": new} if t["name"] == old else t for t in tags])
        for ep in self.episodes():
            segs = load_segments(ep)
            if any(s["tag"] == old for s in segs):
                save_segments(ep, [{**s, "tag": new} if s["tag"] == old else s for s in segs])

    def segments(self, name: str) -> list[dict]:
        return load_segments(self.path / name)

    def add_segment(self, name: str, tag: str, start: int, end: int) -> dict:
        ep = self.path / name
        n = len(load_doc(ep)["data"])
        start, end = max(0, int(start)), min(n - 1, int(end))
        if end <= start:
            raise ValueError("la fin du segment doit être après son début")
        if not any(t["name"] == tag for t in self.tags()):
            raise ValueError(f"étiquette inconnue : {tag}")
        seg = {"tag": tag, "start": start, "end": end}
        save_segments(ep, load_segments(ep) + [seg])
        return seg

    def remove_segment(self, name: str, seg: dict) -> None:
        ep = self.path / name
        segs = load_segments(ep)
        for i, s in enumerate(segs):
            if (s["tag"], s["start"], s["end"]) == (seg["tag"], seg["start"], seg["end"]):
                del segs[i]
                break
        save_segments(ep, segs)

    def export_segments(self, tags: list[str], out: Path) -> int:
        """Dossier de tâche ``out`` dont chaque épisode est un segment portant une des ``tags``,
        avec la consigne de son étiquette (vide : consigne de l'épisode). ``out`` est REMPLACÉ.
        Rend le nombre d'épisodes écrits."""
        out = Path(out)
        if out.resolve() == self.path.resolve():
            raise ValueError("l'export ne peut pas remplacer la tâche source")
        if out.exists() and not is_export(out):
            raise ValueError(f"{out.name} existe et n'est pas un export de segments : il ne sera pas remplacé")
        instr = {t["name"]: t.get("instruction", "") for t in self.tags()}
        tmp = out.with_name(f".export_{out.name}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        k = 0
        for ep in self.episodes():
            segs = [s for s in load_segments(ep) if s["tag"] in tags]
            if not segs:
                continue
            doc = load_doc(ep)
            for s in segs:
                steps = [dict(st) for st in doc["data"][s["start"]:s["end"] + 1]]
                if len(steps) < 2:
                    continue
                dst = tmp / f"episode_{k:04d}"
                dst.mkdir()
                for sub in {Path(p).parent for st in steps for p in st["colors"].values()}:
                    (dst / sub).mkdir(parents=True, exist_ok=True)
                for i, st in enumerate(steps):
                    st["idx"] = i
                    for p in st["colors"].values():
                        if (ep / p).exists() and not (dst / p).exists():
                            _link_or_copy(ep / p, dst / p)
                info = {key: v for key, v in doc.get("info", {}).items()
                        if key not in ("success_step", "takeover_step", "trimmed", "outcome")}
                if doc.get("info", {}).get("outcome") == "success":
                    info.update(outcome="success", success_step=len(steps) - 1)
                info["segment"] = {"source": str(self.path), "episode": ep.name, "tag": s["tag"],
                                   "start": s["start"], "end": s["end"]}
                text = dict(doc.get("text", {}))
                if instr.get(s["tag"]):
                    text["goal"] = instr[s["tag"]]
                save_doc(dst, {"info": info, "text": text, "data": steps})
                k += 1
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        return k

    def set_goal(self, names: list[str], goal: str) -> None:
        for n in names:
            ep = self.path / n
            doc = load_doc(ep)
            doc.setdefault("text", {})["goal"] = goal
            save_doc(ep, doc)

    def set_outcome(self, name: str, outcome: str | None) -> None:
        ep = self.path / name
        doc = load_doc(ep)
        info = doc.setdefault("info", {})
        if outcome:
            info["outcome"] = outcome
            if outcome == "success" and info.get("success_step") is None:
                info["success_step"] = len(doc["data"]) - 1
        else:
            info.pop("outcome", None)
        save_doc(ep, doc)


def is_export(task: Path) -> bool:
    """Vrai si tous les épisodes de ``task`` viennent d'``export_segments`` (dossier régénérable)."""
    eps = [p for p in Path(task).iterdir() if p.is_dir() and EP_RE.match(p.name)]
    try:
        return all("segment" in load_doc(p).get("info", {}) for p in eps)
    except Exception:
        return False


def find_tasks(root: str | Path) -> list[Path]:
    """Dossiers de tâche sous ``root`` (eux-mêmes, ou leurs enfants contenant des episode_*)."""
    root = Path(root)
    if any(EP_RE.match(p.name) for p in root.iterdir() if p.is_dir()):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
                  and any(EP_RE.match(q.name) for q in p.iterdir() if q.is_dir()))
