"""Pinces Dex1 CÂBLÉES EN INTERNE (G1-D) pour la téléop : même interface que
``Dex1_1_Gripper_Controller`` (valeurs « gâchette » en entrée, état/action en sortie), mais la
commande passe par les moteurs 31/33 du LowCmd publié par ``G1_29_ArmController`` : un seul
publieur sur rt/lowcmd (un second LowCmd écraserait la commande des bras).

Sur ce robot, le service ``dex1_1_gripper`` et les topics ``rt/dex1/*`` ne répondent pas.
"""
import threading
import time

import numpy as np
import logging_mp

logger_mp = logging_mp.getLogger(__name__)

TRIGGER_MIN, TRIGGER_MAX = 5.0, 7.0      # mêmes bornes que Dex1_1_Gripper_Controller
DEX1_OPEN = 5.4


class Dex1InternalGripperController:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock,
                 dual_gripper_state_out, dual_gripper_action_out, arm_ctrl, fps: float = 100.0):
        self.left_in, self.right_in = left_gripper_value_in, right_gripper_value_in
        self.lock, self.state_out, self.action_out = dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out
        self.arm_ctrl, self.dt = arm_ctrl, 1.0 / fps
        arm_ctrl.enable_internal_grippers()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger_mp.info("Initialize Dex1InternalGripperController OK!")

    def _loop(self):
        while self.running:
            with self.left_in.get_lock():
                lv = self.left_in.value
            with self.right_in.get_lock():
                rv = self.right_in.value
            if lv != 0.0 or rv != 0.0:     # entrée initialisée (comme le contrôleur d'origine)
                tl = float(np.interp(lv, [TRIGGER_MIN, TRIGGER_MAX], [0.0, DEX1_OPEN]))
                tr = float(np.interp(rv, [TRIGGER_MIN, TRIGGER_MAX], [0.0, DEX1_OPEN]))
                self.arm_ctrl.set_gripper_targets(tl, tr)
            q = self.arm_ctrl.get_gripper_q()
            tgt = self.arm_ctrl.get_gripper_targets()
            with self.lock:
                self.state_out[0], self.state_out[1] = q
                self.action_out[0], self.action_out[1] = tgt
            time.sleep(self.dt)
