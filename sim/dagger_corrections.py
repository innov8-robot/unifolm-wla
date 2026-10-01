"""Corrections « à la DAgger » tirées des rollouts avec opérateur (``recap_rollouts.py``).

Garde, au format xr_teleoperate, les morceaux d'épisodes qui montrent la BONNE conduite à partir des
états où la politique se met elle-même :

* épisode réussi par la politique seule : gardé en entier ;
* épisode repris par l'opérateur puis réussi : gardé à partir de ``takeover_step - --context`` pas
  (l'état dévié + la correction de l'expert jusqu'à la fin) ;
* épisode raté : écarté.

Les épisodes produits passent ensuite par le convertisseur comme des démos ordinaires.

Usage ::

    python sim/dagger_corrections.py --rollouts playground/sim_raw/stack_dagger_r1 \\
        --out playground/sim_raw/stack_dagger_r1_corr
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--context", type=int, default=0,
                    help="pas de la politique gardés avant la reprise. 0 par défaut : ces pas SONT l'échec "
                         "(ex. pince fermée à vide) et seraient appris par clonage (audit du 1/10)")
    ap.add_argument("--min-steps", dest="min_steps", type=int, default=40, help="morceau minimal gardé")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.out.resolve() == a.rollouts.resolve():
        raise SystemExit("--out doit différer de --rollouts")
    if a.out.exists():
        if not a.overwrite:
            raise SystemExit(f"{a.out} existe déjà (--overwrite)")
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    kept = {"politique": 0, "correction": 0}
    n_out = 0
    for ep in sorted(a.rollouts.glob("episode_*")):
        d = json.loads((ep / "data.json").read_text())
        info = d["info"]
        if info.get("outcome") != "success":
            continue
        tk = info.get("takeover_step")
        start = 0 if tk is None else max(0, tk - a.context)
        items = [it for it in d["data"] if it["idx"] >= start]
        if len(items) < a.min_steps:
            continue
        dst = a.out / f"episode_{n_out:04d}"
        (dst / "colors").mkdir(parents=True)
        new = []
        for k, it in enumerate(items):
            it = dict(it, idx=k, colors=dict(it["colors"]))
            for cam, rel in it["colors"].items():
                name = f"colors/{k:06d}_{cam}.jpg"
                try:
                    os.link(ep / rel, dst / name)   # lien dur : pas de place disque en plus
                except OSError:
                    shutil.copy2(ep / rel, dst / name)   # autre système de fichiers
                it["colors"][cam] = name
            new.append(it)
        ss = info.get("success_step")
        d["info"] = dict(info, success_step=None if ss is None else ss - start,
                         takeover_step=None if tk is None else tk - start, sliced_from=f"{ep.name}@{start}")
        d["data"] = new
        (dst / "data.json").write_text(json.dumps(d))
        kept["politique" if tk is None else "correction"] += 1
        n_out += 1
    print(f"{n_out} épisodes écrits : {kept['politique']} réussites de la politique, "
          f"{kept['correction']} corrections de l'opérateur")


if __name__ == "__main__":
    main()
