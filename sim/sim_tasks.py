"""Registre des tâches de sim, partagé par l'enregistreur de démos et le client d'évaluation.

Chaque tâche fournit : scène, instruction, corps de l'objet, placement aléatoire, expert scripté,
seuil de levée pour la réussite.

* ``cube``    : cube rouge de 4 cm (scène versionnée), expert à orientation de pince G1.
* ``novares`` : pièce Novares prise par sa zone PEINTE (mpc_any), pince verticale. Demande les
  fichiers non versionnés de la pièce (voir ``novares_task.py``).
* ``novares_shift`` : idem, pièce dans une zone décalée jamais vue (test de généralisation / RECAP).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from g1d_sim import SCENE_CUBE_XML, SCENE_XML, G1DSim


@dataclass
class SimTask:
    name: str
    scene_xml: object
    instruction: str
    body: str
    lift_success: float
    #: (sim, rng) -> None : place l'objet au hasard, après reset + go_ready
    place: Callable
    #: (sim, rng, on_step) -> dict résumé (success, lifted, ik_refused, ...)
    expert: Callable


def _cube() -> SimTask:
    import cube_task as C
    return SimTask("cube", SCENE_CUBE_XML, C.INSTRUCTION, "cube", C.LIFT_SUCCESS,
                   place=lambda sim, rng: C.sample_cube(sim, rng),
                   expert=lambda sim, rng, on_step=None: C.run_expert(sim, rng, C.ExpertParams(), on_step=on_step))


#: zone DÉCALÉE pour tester la généralisation et la boucle RECAP (équivalent de la « cuisine inversée »
#: de Delta-0) : 4 à 7 cm plus loin que la zone d'entraînement (±3 cm). L'expert y réussit 12/12.
NOVARES_SHIFT_DX = (0.04, 0.07)


def _novares(shift: bool = False) -> SimTask:
    import novares_task as N
    grasp = N.NovaresGrasp()
    pose0 = {}
    dx = NOVARES_SHIFT_DX if shift else N.DX

    def place(sim: G1DSim, rng: np.random.Generator) -> None:
        if "T" not in pose0:                       # pose stable de la scène, lue une fois après reset
            pose0["T"] = sim.object_pose("piece").copy()
        N.sample_piece(sim, rng, pose0["T"], dx_range=dx)

    return SimTask("novares_shift" if shift else "novares", SCENE_XML, N.INSTRUCTION, "piece", N.LIFT_SUCCESS,
                   place=place, expert=lambda sim, rng, on_step=None: N.run_expert(sim, grasp, on_step=on_step))


TASKS = {"cube": _cube, "novares": _novares, "novares_shift": lambda: _novares(shift=True)}


def get_task(name: str) -> SimTask:
    if name not in TASKS:
        raise ValueError(f"tâche inconnue {name!r} (connues : {sorted(TASKS)})")
    return TASKS[name]()
