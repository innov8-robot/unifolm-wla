"""Cinématique d'un bras du G1-D (7 DoF) sur ``g1_d_dex1.urdf`` — FK + IK Pinocchio.

Extrait de ``mpc_any/src/oc_svfp/motion/kinematics.py`` (même algorithme, mêmes
constantes), sans la correction de TCP par main du dépôt d'origine.

Modèle réduit : tout est verrouillé sauf les 7 joints du bras choisi (base, colonne,
torse, autre bras et mors figés). Les valeurs verrouillées doivent être les MÊMES que
les consignes servos de la sim, sinon la FK ment.

TCP : ``*_gripper_base_link`` décalé de ``TCP_OFFSET_M`` le long des doigts (+y du
repère pince Dex1-1). ⚠ Ce point est à 16 mm du vrai centre des patins
(``TCP_DOIGTS_M``) — convention historique de mpc_any, conservée telle quelle.

IK : moindres carrés amortis sur l'erreur log6, joints clampés aux limites URDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")

#: point de saisie, repère base de la pince (doigts = +y) — convention mpc_any
TCP_OFFSET_M = np.array([0.0, 0.105, 0.0])
#: milieu réel des deux bouts de doigt selon l'URDF (non utilisé par défaut)
TCP_DOIGTS_M = np.array([0.0, 0.0973, 0.0142])


class ArmKinematics:
    def __init__(self, urdf_path: str | Path, side: str = "right",
                 locked_joints: dict[str, float] | None = None,
                 tcp_offset_m=TCP_OFFSET_M) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side='{side}' (attendu left|right)")
        self.side = side
        full = pin.buildModelFromUrdf(str(urdf_path))

        arm = [f"{side}_{j}_joint" for j in ARM_JOINTS]
        q_ref = pin.neutral(full)
        for name, value in (locked_joints or {}).items():
            j = full.joints[full.getJointId(name)]
            if j.nq == 1:
                q_ref[j.idx_q] = value
        lock_ids = [i for i in range(1, full.njoints) if full.names[i] not in arm]
        self.model = pin.buildReducedModel(full, lock_ids, q_ref)
        self.data = self.model.createData()
        # l'ordre des joints du modèle réduit EST l'ordre URDF du bras
        self.joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]

        self._fid = self.model.getFrameId(f"{side}_gripper_base_link")
        self.tcp_offset_m = np.asarray(tcp_offset_m, float)
        self._tcp = pin.SE3(np.eye(3), self.tcp_offset_m)
        self.lower = self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit

    def fk(self, q: np.ndarray) -> pin.SE3:
        """Pose du TCP (repère du modèle Pinocchio) pour la configuration ``q`` [7]."""
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q, float))
        return self.data.oMf[self._fid] * self._tcp

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Couple de gravité [7] — le feedforward que la sim applique sur le bras."""
        return pin.computeGeneralizedGravity(self.model, self.data, np.asarray(q, float))

    def ik(self, target: pin.SE3, q0: np.ndarray | None = None, *,
           max_iters: int = 150, tol: float = 1e-4, damping: float = 1e-6,
           step: float = 0.5, posture: np.ndarray | None = None,
           posture_gain: float = 0.3, follow: bool = False) -> tuple[np.ndarray, bool]:
        """Résout ``fk(q) = target``. Retourne ``(q [7], converged)``.

        ``follow=True`` : UNE SEULE descente depuis ``q0`` — à utiliser pour suivre une
        trajectoire (chunks du VLA) : un re-seed multi-graines peut sauter de branche
        articulaire entre deux sous-cibles et faire balayer le bras.
        ``posture`` : biais dans le nullspace (7 DoF pour une tâche 6D)."""
        if follow:
            if q0 is None:
                raise ValueError("follow=True exige q0 (la branche à suivre)")
            q, err = self._ik_dls(target, np.asarray(q0, float).copy(), max_iters,
                                  tol, damping, step, posture, posture_gain)
            return q, err < tol
        return self._ik_multiseed(target, q0, max_iters, tol, damping, step,
                                  posture, posture_gain)

    def _ik_multiseed(self, target, q0, max_iters, tol, damping, step,
                      posture, posture_gain):
        seeds = []
        if q0 is not None:
            seeds.append(np.asarray(q0, float).copy())
        if posture is not None:
            posture = np.asarray(posture, float)
            seeds.append(posture.copy())
        mid = (self.lower + self.upper) / 2
        span = self.upper - self.lower
        seeds.append(pin.neutral(self.model))
        for frac in (0.25, -0.25, 0.15, -0.35):
            seeds.append(np.clip(mid + frac * span, self.lower, self.upper))

        best_q, best_err = seeds[0], np.inf
        for seed in seeds:
            q, err = self._ik_dls(target, seed, max_iters, tol, damping, step,
                                  posture, posture_gain)
            if err < tol:
                return q, True
            if err < best_err:
                best_q, best_err = q, err
        return best_q, False

    def _ik_dls(self, target: pin.SE3, q0: np.ndarray, max_iters: int, tol: float,
                damping: float, step: float, posture: np.ndarray | None = None,
                posture_gain: float = 0.3) -> tuple[np.ndarray, float]:
        """Une descente DLS depuis ``q0``. Retourne (q, norme de l'erreur finale)."""
        goal_F = target * self._tcp.inverse()      # cible exprimée pour le frame pince
        q = q0.copy()
        I6 = np.eye(6)
        I7 = np.eye(self.model.nv)
        for _ in range(max_iters):
            pin.framesForwardKinematics(self.model, self.data, q)
            err = pin.log(self.data.oMf[self._fid].actInv(goal_F)).vector
            n = np.linalg.norm(err)
            if n < tol:
                return q, n
            J = pin.computeFrameJacobian(self.model, self.data, q, self._fid,
                                         pin.ReferenceFrame.LOCAL)
            Jpinv = J.T @ np.linalg.inv(J @ J.T + damping * I6)
            dq = Jpinv @ err
            if posture is not None:
                dq += (I7 - Jpinv @ J) @ (posture_gain * (posture - q))
            q = np.clip(pin.integrate(self.model, q, step * dq), self.lower, self.upper)
        pin.framesForwardKinematics(self.model, self.data, q)
        return q, float(np.linalg.norm(pin.log(self.data.oMf[self._fid].actInv(goal_F)).vector))
