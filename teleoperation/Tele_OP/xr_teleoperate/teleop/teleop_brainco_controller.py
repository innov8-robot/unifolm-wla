# main_teleop.py
import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Array, Lock, Value

import logging_mp
import numpy as np
from sshkeyboard import listen_keyboard, stop_listening

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # dds
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from televuer import TeleVuerWrapper
from teleimager.image_client import ImageClient
from teleop.robot_control.robot_arm import (
    G1_23_ArmController,
    G1_29_ArmController,
    H1_2_ArmController,
    H1_ArmController,
)
from teleop.robot_control.robot_arm_ik import (
    G1_23_ArmIK,
    G1_29_ArmIK,
    H1_2_ArmIK,
    H1_ArmIK,
)
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import LocoClientWrapper, MotionSwitcher


def publish_reset_category(category: int, publisher: ChannelPublisher) -> None:
    """Scene Reset signal (simulation)."""
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


@dataclass
class RuntimeFlags:
    """Single source of truth for runtime flags (no 'global' pitfalls)."""

    start: bool = False
    stop: bool = False
    ready: bool = False
    record_running: bool = False
    record_toggle: bool = False
    calib_capture_open: bool = False
    calib_capture_closed: bool = False


def _get_attr_first(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def get_controller_trigger_value(tele_data, side: str) -> float:
    """Robustly read Quest controller index trigger value in [0..1] across different schemas."""
    side = side.lower()
    if side == "left":
        candidates = [
            "left_ctrl_triggerValue",
            "left_ctrl_indexTriggerValue",
            "left_triggerValue",
            "left_trigger",
            "l_triggerValue",
            "l_trigger",
        ]
    else:
        candidates = [
            "right_ctrl_triggerValue",
            "right_ctrl_indexTriggerValue",
            "right_triggerValue",
            "right_trigger",
            "r_triggerValue",
            "r_trigger",
        ]

    v = _get_attr_first(tele_data, candidates)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def smoothstep(x: float) -> float:
    """Ease-in/out for nicer finger motion."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class HandTemplates:
    open75: np.ndarray | None = None
    closed75: np.ndarray | None = None

    def ready(self) -> bool:
        return self.open75 is not None and self.closed75 is not None


class BraincoTemplateCalibrator:
    """Capture OPEN/CLOSED 75D poses from real hand-tracking and reuse them for controller interpolation."""

    def __init__(self, save_dir: str = "./brainco_templates", tag: str = "quest3"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.left = HandTemplates()
        self.right = HandTemplates()
        self._load()

    def _path(self, side: str, kind: str) -> Path:
        return self.save_dir / f"brainco_{self.tag}_{side}_{kind}.npy"

    def _load(self) -> None:
        for side in ("left", "right"):
            for kind in ("open", "closed"):
                p = self._path(side, kind)
                if not p.exists():
                    continue
                arr = np.load(p).astype(np.float64).reshape(75)
                if side == "left" and kind == "open":
                    self.left.open75 = arr
                elif side == "left" and kind == "closed":
                    self.left.closed75 = arr
                elif side == "right" and kind == "open":
                    self.right.open75 = arr
                elif side == "right" and kind == "closed":
                    self.right.closed75 = arr

    def save_open(self, left75: np.ndarray, right75: np.ndarray) -> None:
        left75 = np.asarray(left75, dtype=np.float64).reshape(75)
        right75 = np.asarray(right75, dtype=np.float64).reshape(75)
        np.save(self._path("left", "open"), left75)
        np.save(self._path("right", "open"), right75)
        self.left.open75 = left75
        self.right.open75 = right75
        # Log supprimé

    def save_closed(self, left75: np.ndarray, right75: np.ndarray) -> None:
        left75 = np.asarray(left75, dtype=np.float64).reshape(75)
        right75 = np.asarray(right75, dtype=np.float64).reshape(75)
        np.save(self._path("left", "closed"), left75)
        np.save(self._path("right", "closed"), right75)
        self.left.closed75 = left75
        self.right.closed75 = right75
        # Log supprimé

    def lerp(self, side: str, a: float) -> np.ndarray | None:
        a = smoothstep(float(np.clip(a, 0.0, 1.0)))
        if side == "left":
            if not self.left.ready():
                return None
            return (1.0 - a) * self.left.open75 + a * self.left.closed75
        if not self.right.ready():
            return None
        return (1.0 - a) * self.right.open75 + a * self.right.closed75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="G1_29", choices=["G1_29", "G1_23", "H1_2", "H1"])
    parser.add_argument("--ee", type=str, default="brainco", choices=["dex3", "dex1", "inspire_dfx", "inspire_ftp", "brainco"])
    parser.add_argument("--input-mode", type=str, default="hand", choices=["hand", "controller"])
    parser.add_argument("--display-mode", type=str, default="pass-through", choices=["pass-through", "head-camera", "immersive", "ego"])
    parser.add_argument("--img-server-ip", type=str, default="127.0.0.1")
    parser.add_argument("--network-interface", type=str, default=None, help="Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--ipc", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--affinity", action="store_true")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--task-dir", type=str, default="./tasks")
    parser.add_argument("--task-name", type=str, default="default_task")
    parser.add_argument("--task-goal", type=str, default="")
    parser.add_argument("--task-desc", type=str, default="")
    parser.add_argument("--task-steps", type=str, default="")
    parser.add_argument("--brainco-template-tag", type=str, default="quest3")
    parser.add_argument("--brainco-template-dir", type=str, default="./brainco_templates")
    return parser.parse_args()


def main():
    args = parse_args()
    flags = RuntimeFlags()

    if args.sim:
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
    else:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    brainco_calib = BraincoTemplateCalibrator(
        save_dir=args.brainco_template_dir,
        tag=args.brainco_template_tag,
    )

    def on_press(key):
        if key == "r":
            if flags.ready:
                flags.start = True
        elif key == "q":
            flags.stop = True
        elif key == "s":
            if flags.ready:
                flags.record_toggle = True
        elif key == "o":
            flags.calib_capture_open = True
        elif key == "c":
            flags.calib_capture_closed = True

    if args.ipc:
        ipc_server = IPC_Server()
        ipc_server.start()
        listen_keyboard_thread = None
    else:
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press})
        listen_keyboard_thread.daemon = True
        listen_keyboard_thread.start()

    try:
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        xr_need_local_img = not (args.display_mode == "pass-through" or camera_config["head_camera"]["enable_webrtc"])

        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=args.input_mode == "hand",
            binocular=camera_config["head_camera"]["binocular"],
            img_shape=camera_config["head_camera"]["image_shape"],
            display_mode=args.display_mode,
            zmq=camera_config["head_camera"]["enable_zmq"],
            webrtc=camera_config["head_camera"]["enable_webrtc"],
            webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
        )

        loco_wrapper = None
        if args.motion:
            loco_wrapper = LocoClientWrapper() if args.input_mode == "controller" else None
        else:
            motion_switcher = MotionSwitcher()
            status, _ = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        else:
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # End-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller

            left_hand_pos_array = Array("d", 75, lock=True)
            right_hand_pos_array = Array("d", 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 14, lock=False)
            dual_hand_action_array = Array("d", 14, lock=False)
            _ = Dex3_1_Controller(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                simulation_mode=args.sim,
            )

        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller

            left_gripper_value = Value("d", 0.0, lock=True)
            right_gripper_value = Value("d", 0.0, lock=True)
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array("d", 2, lock=False)
            dual_gripper_action_array = Array("d", 2, lock=False)
            _ = Dex1_1_Gripper_Controller(
                left_gripper_value,
                right_gripper_value,
                dual_gripper_data_lock,
                dual_gripper_state_array,
                dual_gripper_action_array,
                simulation_mode=args.sim,
            )

        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX

            left_hand_pos_array = Array("d", 75, lock=True)
            right_hand_pos_array = Array("d", 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)
            dual_hand_action_array = Array("d", 12, lock=False)
            _ = Inspire_Controller_DFX(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                simulation_mode=args.sim,
            )

        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP

            left_hand_pos_array = Array("d", 75, lock=True)
            right_hand_pos_array = Array("d", 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)
            dual_hand_action_array = Array("d", 12, lock=False)
            _ = Inspire_Controller_FTP(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                simulation_mode=args.sim,
            )

        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller

            left_hand_pos_array = Array("d", 75, lock=True)
            right_hand_pos_array = Array("d", 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)
            dual_hand_action_array = Array("d", 12, lock=False)
            _ = Brainco_Controller(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                simulation_mode=args.sim,
            )
        else:
            raise ValueError(f"Unknown ee: {args.ee}")

        # Simulation
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe

            sim_state_subscriber = start_sim_state_subscribe()

        # Recording
        if args.record:
            recorder = EpisodeWriter(
                task_dir=os.path.join(args.task_dir, args.task_name),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                rerun_log=not args.headless,
            )
        else:
            recorder = None

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢 [r] start tracking")
        logger_mp.info("🔴 [q] quit")
        logger_mp.info("🟣 [o] capture BrainCo OPEN (hand mode)")
        logger_mp.info("🟣 [c] capture BrainCo CLOSED (hand mode)")
        if args.record:
            logger_mp.info("🟡 [s] toggle record")
        logger_mp.info("----------------------------------------------------------------")

        flags.ready = True

        while not flags.start and not flags.stop:
            # Gestion des boutons A (start) et B (stop) de la manette Meta Quest 3
            tele_data = tv_wrapper.get_tele_data()
            if getattr(tele_data, "right_ctrl_aButton", False):
                flags.start = True
            if getattr(tele_data, "right_ctrl_bButton", False):
                flags.stop = True
            time.sleep(0.033)
            if camera_config["head_camera"]["enable_zmq"] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀 start Tracking 🚀-------------------------")
        arm_ctrl.speed_gradual_max()

        trig_print_t = 0.0

        while not flags.stop:
            loop_t0 = time.time()

            # images
            if camera_config["head_camera"]["enable_zmq"]:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)

            if args.record and flags.record_toggle and recorder is not None:
                flags.record_toggle = False
                if not flags.record_running:
                    if recorder.create_episode():
                        flags.record_running = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    flags.record_running = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # XR data
            tele_data = tv_wrapper.get_tele_data()

            # Détection boutons A/B/X/Y pour start/stop (A=start, B=stop)
            if getattr(tele_data, 'right_ctrl_aButton', False) or getattr(tele_data, 'left_ctrl_aButton', False):
                flags.start = True
            if getattr(tele_data, 'right_ctrl_bButton', False) or getattr(tele_data, 'left_ctrl_bButton', False):
                flags.stop = True

            # BrainCo hand mode: pass-through + allow template capture
            if args.ee == "brainco" and args.input_mode == "hand":
                left75 = np.asarray(tele_data.left_hand_pos.flatten(), dtype=np.float64).reshape(75)
                right75 = np.asarray(tele_data.right_hand_pos.flatten(), dtype=np.float64).reshape(75)

                if flags.calib_capture_open:
                    flags.calib_capture_open = False
                    brainco_calib.save_open(left75, right75)

                if flags.calib_capture_closed:
                    flags.calib_capture_closed = False
                    brainco_calib.save_closed(left75, right75)

                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = left75.tolist()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = right75.tolist()

            # BrainCo controller mode: interpolate real templates by trigger
            elif args.ee == "brainco" and args.input_mode == "controller":
                tl = get_controller_trigger_value(tele_data, "left")
                tr = get_controller_trigger_value(tele_data, "right")

                # Suppression des logs [TRIG]
                if np.isnan(tl) or np.isnan(tr):
                    fields = [k for k in dir(tele_data) if ("trigger" in k.lower() or "grip" in k.lower())]

                left75 = brainco_calib.lerp("left", 0.0 if np.isnan(tl) else tl)
                right75 = brainco_calib.lerp("right", 0.0 if np.isnan(tr) else tr)

                if left75 is None or right75 is None:
                    # Not calibrated: keep hands neutral (better: do calibration once in hand mode).
                    left75 = np.zeros(75, dtype=np.float64) if left75 is None else left75
                    right75 = np.zeros(75, dtype=np.float64) if right75 is None else right75

                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = left75.tolist()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = right75.tolist()

            # Other EEs (kept compatible with your existing behavior)
            else:
                if (args.ee in ["dex3", "inspire_dfx", "inspire_ftp"]) and args.input_mode == "hand":
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_ctrl_triggerValue
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_ctrl_triggerValue
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_hand_pinchValue
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_hand_pinchValue

            # motion base
            if args.input_mode == "controller" and args.motion and loco_wrapper is not None:
                if getattr(tele_data, "right_ctrl_aButton", False):
                    flags.start = False
                    flags.stop = True
                if getattr(tele_data, "left_ctrl_thumbstick", False) and getattr(tele_data, "right_ctrl_thumbstick", False):
                    loco_wrapper.Damp()
                loco_wrapper.Move(
                    -tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                    -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                    -tele_data.right_ctrl_thumbstickValue[0] * 0.3,
                )

            # arm IK + control
            current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            sol_q, sol_tauff = arm_ik.solve_ik(
                tele_data.left_wrist_pose,
                tele_data.right_wrist_pose,
                current_lr_arm_q,
                current_lr_arm_dq,
            )
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record
            if recorder is not None:
                flags.ready = recorder.is_ready()

            # timing
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, (1.0 / args.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program.")
    except Exception:
        import traceback

        logger_mp.error(traceback.format_exc())
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                if listen_keyboard_thread is not None:
                    listen_keyboard_thread.join(timeout=1.0)
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")

        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        logger_mp.info("✅ Exiting.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()