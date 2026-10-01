import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import cv2
import logging_mp
import numpy as np
import pinocchio as pin
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

def compose_xr_image(head_bgr, left_wrist_bgr=None, right_wrist_bgr=None, binocular=True, scale=0.28, margin=10, recording=False):
    """Overlay wrist-camera frames as picture-in-picture insets onto a COPY of the head image for VR display.
    Does NOT modify head_bgr (that array is what gets recorded). Insets are drawn into each eye half so they
    appear in stereo. When recording=True, a blinking red dot is drawn bottom-right of each eye.
    Returns head_bgr unchanged if there is nothing to overlay."""
    if head_bgr is None:
        return head_bgr
    if left_wrist_bgr is None and right_wrist_bgr is None and not recording:
        return head_bgr
    out = head_bgr.copy()
    h, w = out.shape[:2]
    eye_w = w // 2 if binocular else w
    pip_w = max(1, int(eye_w * scale))
    pip_h = max(1, pip_w * 3 // 4)  # wrist cams are 4:3
    # blink ~1 Hz: on for 0.5 s, off for 0.5 s
    blink_on = int(time.time() * 2) % 2 == 0
    dot_r = max(6, int(eye_w * 0.02))
    for eye in range(2 if binocular else 1):
        x_off = eye * eye_w
        y0 = margin
        y1 = min(h, y0 + pip_h)
        if left_wrist_bgr is not None:
            out[y0:y1, x_off + margin:x_off + margin + pip_w] = cv2.resize(left_wrist_bgr, (pip_w, pip_h))
        if right_wrist_bgr is not None:
            xr1 = x_off + eye_w - margin
            out[y0:y1, xr1 - pip_w:xr1] = cv2.resize(right_wrist_bgr, (pip_w, pip_h))
        if recording and blink_on:
            cx = x_off + eye_w - margin - dot_r
            cy = h - margin - dot_r
            cv2.circle(out, (cx, cy), dot_r, (0, 0, 255), -1)          # filled red (BGR)
            cv2.circle(out, (cx, cy), dot_r, (255, 255, 255), max(1, dot_r // 6))  # white ring for contrast
    return out

# state transition
TORSO_STICK_DEADZONE = 0.2
CAL_POSE_Q = np.zeros(14)    # posture de calibration (Y) : position zéro du G1 = coudes ~80°, avant-bras vers l'avant
CAL_MOVE_S = 2.5             # durée du trajet vers la posture de calibration (s)  # joystick droit : zone morte de la rotation du buste
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECORD_CANCEL  = False  # Discard the current in-progress recording

#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--wrist-pip', dest='wrist_pip', action='store_true', default=True, help='Show wrist cameras as picture-in-picture insets in the VR head view (ZMQ display modes only)')
    parser.add_argument('--no-wrist-pip', dest='wrist_pip', action='store_false', help='Disable wrist-camera picture-in-picture in VR')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--right-only', dest='right_only', action='store_true', help='Freeze the LEFT arm at its startup pose (held, not limp) and freeze the left gripper; drop the left wrist camera from both the VR view and the recording. Head + right wrist + right arm/ee still tracked and recorded.')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    # mode POLITIQUE (G1-D, RECAP / Delta-0) : la politique WLA pilote, l'opérateur corrige en delta
    parser.add_argument('--policy-uri', type=str, default=None,
                        help='Serveur UnifoLM-WLA (ex. ws://192.168.123.2:8600). Active le mode politique : '
                             'grip maintenu = correction en delta du bras de ce côté, gâchette = pince ; '
                             'X gauche = essai RÉUSSI, Y gauche = essai RATÉ (fin et sauvegarde de l\'épisode).')
    parser.add_argument('--policy-instruction', type=str, default=None, help='instruction envoyée au modèle (défaut : --task-goal)')
    parser.add_argument('--policy-advantage', type=str, default=None, help='condition RECAP envoyée au modèle, ex. positive')
    parser.add_argument('--policy-exec-steps', type=int, default=30, help='pas exécutés par chunk avant replanification')
    parser.add_argument('--policy-max-speed', type=float, default=0.10, help='vitesse max des cibles (m/s) ; commencer bas')
    parser.add_argument('--torso-pitch-index', type=int, default=13, help='indice du tangage du buste dans les 35 moteurs (HYPOTHÈSE)')
    parser.add_argument('--torso-pitch', type=float, default=None, help='tangage du buste constant (rad), remplace --torso-pitch-index')
    parser.add_argument('--ik-smooth', choices=['standard', 'light', 'off'], default='standard',
                        help="lissage des angles IK : standard = filtre amont 4 pas (~2 pas de retard), light = 2 pas, "
                             "off = aucun. Moins de lissage = moins de latence, plus de tremblement")
    parser.add_argument('--record-fps', type=float, default=30.0,
                        help="fréquence d'ENREGISTREMENT (WLA : 30). Avec --frequency plus haut, un pas sur N est gardé")
    parser.add_argument('--timing', action='store_true',
                        help="journal toutes les 2 s : fréquence de boucle, temps d'IK, retard des bras sur la consigne")
    parser.add_argument('--dex1-bus', choices=['internal', 'usb'], default='internal',
                        help="Dex1 : 'internal' = câblées par les poignets, moteurs 31/33 du LowCmd (G1-D) ; "
                             "'usb' = service dex1_1_gripper et topics rt/dex1/* (amont)")
    parser.add_argument('--torso-yaw-index', type=int, default=None,
                        help='G1-D : indice du moteur de rotation du buste (torso_Joint) dans les 35 moteurs. '
                             'Active la rotation au joystick droit (gauche/droite). HYPOTHÈSE à vérifier sur le robot')
    parser.add_argument('--torso-yaw-max', type=float, default=0.6, help='rotation du buste max (rad, ±)')
    parser.add_argument('--torso-yaw-rate', type=float, default=0.5, help='vitesse de rotation du buste max (rad/s)')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")
    if args.policy_uri and not (args.arm == "G1_29" and args.ee == "dex1" and args.input_mode == "controller" and not args.motion):
        parser.error("--policy-uri demande --arm G1_29 --ee dex1 --input-mode controller, sans --motion")
    if args.ee == "dex1" and args.dex1_bus == "internal" and not args.sim and args.arm != "G1_29":
        parser.error("--dex1-bus internal demande --arm G1_29 (commande dans le LowCmd des bras)")
    if args.policy_uri and abs(args.frequency - 30.0) > 1e-6:
        parser.error("--policy-uri : les chunks du modèle sont à 30 Hz, garder --frequency 30")
    RECORD_STRIDE = max(1, int(round(args.frequency / args.record_fps)))
    if abs(args.frequency / RECORD_STRIDE - args.record_fps) > 0.5:
        parser.error(f"--frequency {args.frequency} n'est pas un multiple de --record-fps {args.record_fps}")
    if args.torso_yaw_index is not None and not (args.arm == "G1_29" and args.input_mode == "controller" and not args.motion):
        parser.error("--torso-yaw-index demande --arm G1_29 --input-mode controller, sans --motion "
                     "(en --motion, le joystick droit tourne la base)")
    LEFT_TRIGGER_PREV = 0.0
    RIGHT_TRIGGER_PREV = 0.0

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        # --right-only: drop the left wrist camera everywhere (VR PiP + recording) by
        # forcing its enable_zmq off locally; every downstream check already guards on it.
        if args.right_only:
            camera_config['left_wrist_camera']['enable_zmq'] = False
            logger_mp.info("[right-only] left wrist camera disabled (VR view + recording)")

        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # arm
        # mode politique (G1-D, RECAP) : connexion au serveur WLA AVANT le mode debug et l'init des bras :
        # un serveur injoignable arrête le programme sans jamais toucher au robot (audits du 29/09 et du 1/10)
        bridge = None
        if args.policy_uri:
            if not camera_config['head_camera']['enable_zmq']:
                raise SystemExit("mode politique : la caméra de tête doit être servie en ZMQ (enable_zmq: true)")
            from policy_bridge import PolicyBridge, PolicyError
            bridge = PolicyBridge(args.policy_uri, args.policy_instruction or args.task_goal,
                                  advantage=args.policy_advantage, exec_steps=args.policy_exec_steps,
                                  frequency=args.frequency, max_speed=args.policy_max_speed,
                                  torso_pitch_index=args.torso_pitch_index, torso_pitch=args.torso_pitch,
                                  control_left=not args.right_only, torso_yaw_index=args.torso_yaw_index)
            logger_mp.info(f"🤖  Mode POLITIQUE : {args.policy_uri} | grip = corriger, gâchette = pince, "
                           f"X gauche = réussi, Y gauche = raté | vitesse max {args.policy_max_speed} m/s")

        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            # latence : le filtre amont (4 pas pondérés 0.4/0.3/0.2/0.1) retarde la consigne d'environ 2 pas
            arm_ik.smooth_filter = WeightedMovingFilter(
                np.array({"standard": [0.4, 0.3, 0.2, 0.1], "light": [0.7, 0.3], "off": [1.0]}[args.ik_smooth]), 14)
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            if args.torso_yaw_index is not None:
                logger_mp.warning(f"rotation du buste sur le moteur {args.torso_yaw_index} : HYPOTHÈSE, vérifier "
                                  f"l'indice sur le robot avant tout essai (bras dégagés)")
                # mode politique : la rotation du buste emporte les bras hors de la borne --policy-max-speed ;
                # vitesse du buste bornée pour qu'une main à 0,5 m de l'axe reste sous cette vitesse
                yaw_rate = min(args.torso_yaw_rate, args.policy_max_speed / 0.5) if bridge is not None else args.torso_yaw_rate
                arm_ctrl.enable_torso_yaw(args.torso_yaw_index, args.torso_yaw_max, yaw_rate,
                                          forbidden=(args.torso_pitch_index,) if args.torso_pitch is None else ())
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            if args.dex1_bus == "internal" and not args.sim:
                # G1-D : Dex1 câblées en interne, moteurs 31/33 du LowCmd des bras (un seul publieur)
                from teleop.robot_control.dex1_internal import Dex1InternalGripperController
                gripper_ctrl = Dex1InternalGripperController(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                             dual_gripper_state_array, dual_gripper_action_array, arm_ctrl)
            else:
                gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                         dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        policy_intervention = False
        policy_failed = False          # erreur du serveur : la politique est suspendue jusqu'au prochain essai
        policy_paused = False          # pas où la politique ne pilote pas : pas enregistrés
        hold_ik = None                 # cibles figées quand la politique ne pilote pas
        torso_yaw_offset = 0.0         # mode politique : correction de rotation du buste par l'opérateur
        last_noimg_log = 0.0
        cal = {"paused": False, "t0": 0.0, "q0": np.zeros(14),
               "offset": {"left": np.zeros(3), "right": np.zeros(3)}}
        prev_cY = False
        loop_count = -1
        timing_acc = {"n": 0, "ik": 0.0, "lag": 0.0, "dpos": 0.0, "prev_t": None, "t0": time.time()}
        episode_steps = 0
        head_img = left_wrist_img = right_wrist_img = None      # caméra désactivée : reste None
        prev_lX = prev_lY = False

        # record + headless / non-headless mode
        if args.record:
            # Real recorded color/depth resolution = head camera (color_0 / depth_0).
            # image_shape is [height, width]; binocular splits width across the two eyes.
            _head_shape = camera_config['head_camera']['image_shape']
            _rec_w = _head_shape[1] // (2 if camera_config['head_camera']['binocular'] else 1)
            _rec_h = _head_shape[0]
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.record_fps,
                                     image_size = [_rec_w, _rec_h],
                                     rerun_log = not args.headless)
            # indices du buste dans body.qpos, lus par le convertisseur (g1d_wla.convert_teleop)
            recorder.info["body_layout"] = {"torso_pitch": args.torso_pitch_index, "torso_yaw": args.torso_yaw_index,
                                            "torso_pitch_const": args.torso_pitch}

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    lw = img_client.get_left_wrist_frame().bgr  if (args.wrist_pip and camera_config['left_wrist_camera']['enable_zmq'])  else None
                    rw = img_client.get_right_wrist_frame().bgr if (args.wrist_pip and camera_config['right_wrist_camera']['enable_zmq']) else None
                    tv_wrapper.render_to_xr(compose_xr_image(head_img.bgr, lw, rw, camera_config['head_camera']['binocular']))

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        frozen_left_arm_q = None  # --right-only: left arm pose captured at first loop iteration
        prev_rA = prev_rB = False  # previous right-controller A/B states, for rising-edge detection
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img or bridge is not None:
                    head_img = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record or (args.wrist_pip and xr_need_local_img) or bridge is not None:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record or (args.wrist_pip and xr_need_local_img) or bridge is not None:
                    right_wrist_img = img_client.get_right_wrist_frame()
            if xr_need_local_img and camera_config['head_camera']['enable_zmq'] and head_img.bgr is not None:
                lw = left_wrist_img.bgr  if (args.wrist_pip and camera_config['left_wrist_camera']['enable_zmq'])  else None
                rw = right_wrist_img.bgr if (args.wrist_pip and camera_config['right_wrist_camera']['enable_zmq']) else None
                tv_wrapper.render_to_xr(compose_xr_image(head_img.bgr, lw, rw, camera_config['head_camera']['binocular'], recording=RECORD_RUNNING))

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                        episode_steps = 0
                        torso_yaw_offset = 0.0
                        if bridge is not None:
                            policy_failed = False
                            try:
                                bridge.reset()
                            except Exception as e:
                                policy_failed = True
                                logger_mp.error(f"🤖  serveur WLA injoignable, politique suspendue : {e}")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            # cancel/discard the current in-progress recording (controller B button)
            if args.record and RECORD_CANCEL:
                RECORD_CANCEL = False
                if RECORD_RUNNING:
                    RECORD_RUNNING = False
                    recorder.cancel_episode()
                    logger_mp.info("🚫  Recording CANCELLED (episode discarded).")

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()

            # record control via right controller (rising-edge): A = start/stop toggle, B = cancel.
            # A is only bound to recording when NOT in motion mode (there A = quit teleop).
            if args.record and args.input_mode == "controller" and START:
                rA = bool(tele_data.right_ctrl_aButton)
                rB = bool(tele_data.right_ctrl_bButton)
                if rA and not prev_rA and not args.motion:
                    RECORD_TOGGLE = True
                    if bridge is not None and RECORD_RUNNING:     # fin sans X/Y : issue inconnue (ignorée par RECAP)
                        recorder.set_episode_info({"outcome": "unknown"})
                if rB and not prev_rB:
                    RECORD_CANCEL = True
                prev_rA, prev_rB = rA, rB
            # Y gauche (téléop, hors mode politique) : PAUSE du suivi -> bras vers la posture de calibration
            # (q = 0 : bras le long du corps, avant-bras vers l'avant, coude ~80°) ; second appui = REPRISE
            # avec recalage : la pose actuelle des manettes devient celle des mains du robot (pas de saut)
            if bridge is None and args.input_mode == "controller" and START and args.arm == "G1_29":
                cY = bool(tele_data.left_ctrl_bButton)
                if cY and not prev_cY:
                    if not cal["paused"]:
                        cal.update(paused=True, t0=time.time(), q0=np.asarray(arm_ctrl.get_current_dual_arm_q()).copy())
                        logger_mp.info("⏸️  Suivi en PAUSE : bras vers la posture de calibration (coudes à 90°). "
                                       "Mettez-vous dans la même posture puis Y pour reprendre.")
                    else:
                        pin.framesForwardKinematics(arm_ik.reduced_robot.model, arm_ik.reduced_robot.data,
                                                    np.asarray(arm_ctrl.get_current_dual_arm_q()))
                        for s_, fr in (("left", "L_ee"), ("right", "R_ee")):
                            p_robot = arm_ik.reduced_robot.data.oMf[arm_ik.reduced_robot.model.getFrameId(fr)].translation
                            p_hand = np.asarray(getattr(tele_data, f"{s_}_wrist_pose"))[:3, 3]
                            cal["offset"][s_] = np.asarray(p_robot) - p_hand
                        arm_ik.smooth_filter = WeightedMovingFilter(arm_ik.smooth_filter._weights, 14)
                        cal["paused"] = False
                        logger_mp.info(f"▶️  Suivi REPRIS, recalé : décalage gauche {np.round(cal['offset']['left'], 3)} m, "
                                       f"droite {np.round(cal['offset']['right'], 3)} m")
                prev_cY = cY
            if bridge is not None and args.record and START:
                lX, lY = bool(tele_data.left_ctrl_aButton), bool(tele_data.left_ctrl_bButton)
                if RECORD_RUNNING and not RECORD_CANCEL and not tele_data.right_ctrl_bButton \
                        and ((lX and not prev_lX) or (lY and not prev_lY)):
                    ok_ep = lX and not prev_lX
                    recorder.set_episode_info({"success_step": episode_steps if ok_ep else None,
                                               "outcome": "success" if ok_ep else "failure"})
                    logger_mp.info(f"🏁  Essai {'RÉUSSI' if ok_ep else 'RATÉ'} au pas {episode_steps}")
                    RECORD_TOGGLE = True                     # arrête et sauvegarde l'épisode
                prev_lX, prev_lY = lX, lY
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller" and bridge is not None:
                pass                       # mode politique : la pince est commandée plus bas (politique ou tenue)
            elif args.ee == "dex1" and args.input_mode == "controller" and cal["paused"]:
                pass                                   # pause de calibration : pinces figées
            elif args.ee == "dex1" and args.input_mode == "controller":
                if not args.right_only:
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                if not args.right_only:
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # --right-only: capture the left arm pose once, at the first iteration
            if args.right_only and frozen_left_arm_q is None:
                frozen_left_arm_q = current_lr_arm_q[:7].copy()
                logger_mp.info(f"[right-only] left arm frozen at startup pose: {frozen_left_arm_q}")
                if args.ee == "dex1":
                    # pince gauche figée à son ouverture mesurée : sans consigne, la valeur 0 la fermait
                    # (« gâchette » 5 = fermée, 7 = ouverte, unité Dex1 0 -> 5.4 ; audit du 1/10)
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = 5.0 + 2.0 * float(np.clip(dual_gripper_state_array[0] / 5.4, 0.0, 1.0))

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            left_target, right_target = tele_data.left_wrist_pose, tele_data.right_wrist_pose
            if bridge is None and (np.any(cal["offset"]["left"]) or np.any(cal["offset"]["right"])):
                left_target, right_target = np.array(left_target, float), np.array(right_target, float)
                left_target[:3, 3] += cal["offset"]["left"]
                right_target[:3, 3] += cal["offset"]["right"]
            if bridge is not None:
                policy_active = (RECORD_RUNNING if args.record else START) and not policy_failed
                body_q = arm_ctrl.get_current_motor_q()
                imgs_ok = (head_img is not None and head_img.bgr is not None and right_wrist_img is not None
                           and right_wrist_img.bgr is not None
                           and (args.right_only or (left_wrist_img is not None and left_wrist_img.bgr is not None)))
                res = None
                if policy_active and imgs_ok:
                    head = head_img.bgr
                    if camera_config['head_camera']['binocular']:
                        head = head[:, :head.shape[1] // 2]            # œil GAUCHE (brut), comme les datasets
                    # --right-only : pas de caméra de poignet gauche -> image noire (jamais celle de droite)
                    lw = np.zeros_like(right_wrist_img.bgr) if args.right_only else left_wrist_img.bgr
                    with dual_gripper_data_lock:
                        grip_meas = np.array([dual_gripper_state_array[0], dual_gripper_state_array[1]])
                    try:
                        res = bridge.step({"head": head, "left_wrist": lw, "right_wrist": right_wrist_img.bgr},
                                          np.asarray(current_lr_arm_q), grip_meas, body_q, tele_data)
                    except Exception as e:                    # serveur muet, erreur, actions invalides
                        policy_failed = True
                        bridge.pause()
                        logger_mp.error(f"🤖  POLITIQUE SUSPENDUE (bras tenus) : {e} — X/Y pour finir l'essai, "
                                        f"B pour l'annuler")
                elif policy_active and not imgs_ok:
                    bridge.pause()                            # image absente : on tient, la reprise repart de la mesure
                    if time.time() - last_noimg_log > 2.0:
                        last_noimg_log = time.time()
                        logger_mp.warning("🤖  image(s) absente(s) (tête / poignets) : politique en tenue")
                if res is not None:
                    hold_ik = None
                    policy_paused = False
                    left_target, right_target = res["left"], res["right"]
                    policy_intervention = res["intervention"]
                    if not args.right_only:
                        with left_gripper_value.get_lock():
                            left_gripper_value.value = res["trigger"]["left"]
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = res["trigger"]["right"]
                else:
                    # politique inactive : cibles et pince FIGÉES à l'entrée en tenue (recalculer chaque tour
                    # depuis la mesure fait dériver les bras vers q = 0, audit du 29/09)
                    if hold_ik is None:
                        hold_ik = bridge.measured_ik(np.asarray(current_lr_arm_q), body_q)
                        with dual_gripper_data_lock:
                            g_hold = [dual_gripper_state_array[0], dual_gripper_state_array[1]]
                        from policy_bridge import dex1_to_trigger
                        if not args.right_only:
                            with left_gripper_value.get_lock():
                                left_gripper_value.value = dex1_to_trigger(g_hold[0])
                        with right_gripper_value.get_lock():
                            right_gripper_value.value = dex1_to_trigger(g_hold[1])
                    left_target, right_target = hold_ik["left"], hold_ik["right"]
                    policy_intervention = False
                    policy_paused = True
            # rotation du buste (G1-D) : joystick droit gauche/droite. Les cibles IK sont dans un repère lié
            # au torse : tourner le buste emporte les bras (comme quand l'opérateur pivote sur lui-même).
            if args.torso_yaw_index is not None and START:
                stick = float(tele_data.right_ctrl_thumbstickValue[0])
                d_yaw = 0.0 if abs(stick) < TORSO_STICK_DEADZONE else -stick * args.torso_yaw_rate / args.frequency
                if bridge is None:
                    arm_ctrl.set_torso_yaw(arm_ctrl.get_torso_yaw_target() + d_yaw)
                elif res is not None and res.get("torso_yaw") is not None:
                    # mode politique : rotation prédite + correction de l'opérateur, comptée comme intervention
                    # TANT QU'ELLE EST NON NULLE (pas seulement quand le joystick bouge)
                    torso_yaw_offset = float(np.clip(torso_yaw_offset + d_yaw, -args.torso_yaw_max, args.torso_yaw_max))
                    arm_ctrl.set_torso_yaw(res["torso_yaw"] + torso_yaw_offset)
                    if abs(torso_yaw_offset) > 1e-3:
                        policy_intervention = True
                else:
                    # politique en tenue, ou rotation non prédite : buste FIGÉ à sa consigne courante
                    arm_ctrl.hold_torso_yaw()
            if cal["paused"]:
                a_ = min(1.0, (time.time() - cal["t0"]) / CAL_MOVE_S)
                a_ = a_ * a_ * (3 - 2 * a_)                       # départ et arrivée en douceur
                sol_q = (1 - a_) * cal["q0"] + a_ * CAL_POSE_Q
                sol_tauff = pin.rnea(arm_ik.reduced_robot.model, arm_ik.reduced_robot.data, sol_q,
                                     np.zeros(14), np.zeros(14))  # compensation de gravité
                if args.torso_yaw_index is not None:
                    arm_ctrl.hold_torso_yaw()
            else:
                sol_q, sol_tauff  = arm_ik.solve_ik(left_target, right_target, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            if args.timing:
                tm = timing_acc
                tm["n"] += 1
                tm["ik"] += time_ik_end - time_ik_start
                tm["lag"] = max(tm["lag"], float(np.max(np.abs(np.asarray(sol_q) - np.asarray(current_lr_arm_q)))))
                tm["dpos"] = max(tm["dpos"], float(np.linalg.norm(np.asarray(right_target)[:3, 3] - tm["prev_t"]))) if tm["prev_t"] is not None else 0.0
                tm["prev_t"] = np.asarray(right_target)[:3, 3].copy()
                if time.time() - tm["t0"] > 2.0:
                    dt = time.time() - tm["t0"]
                    logger_mp.info(f"[timing] boucle {tm['n']/dt:.1f} Hz | IK {1000*tm['ik']/max(tm['n'],1):.1f} ms | "
                                   f"écart max consigne-mesure bras {tm['lag']:.3f} rad | saut max cible main D {1000*tm['dpos']:.0f} mm/pas")
                    timing_acc.update(n=0, ik=0.0, lag=0.0, dpos=0.0, t0=time.time())
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            # --right-only: hold the left arm at its captured pose (position-held, no feedforward)
            if args.right_only and frozen_left_arm_q is not None:
                sol_q[:7] = frozen_left_arm_q
                sol_tauff[:7] = 0.0
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data (un pas sur RECORD_STRIDE si la boucle tourne plus vite que l'enregistrement)
            loop_count += 1
            if args.record and loop_count % RECORD_STRIDE == 0:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        # commande de base : seulement si ce programme la pilote (--motion). Sinon la base
                        # ne bouge pas et le joystick droit sert au buste : enregistrer 0 (base à l'arrêt)
                        current_body_action = ([-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                                -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                                -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                                               if args.motion else [0.0, 0.0, 0.0])
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    frame_ok = True            # une caméra active sans image : pas NON enregistré (audit du 1/10)
                    # Sequential color indexing: each ENABLED camera reserves the next
                    # color_N slot (a None frame leaves its reserved slot empty but does
                    # not shift the others). With every camera enabled this yields the
                    # historical keys (head=0[,1], left wrist, right wrist); when a camera
                    # is disabled (e.g. --right-only) the remaining keys stay contiguous.
                    ci = 0
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None and head_img.bgr is not None:
                            half = camera_config['head_camera']['image_shape'][1] // 2
                            colors[f"color_{ci}"]     = head_img.bgr[:, :half]
                            colors[f"color_{ci + 1}"] = head_img.bgr[:, half:]
                        else:
                            frame_ok = False
                            logger_mp.warning("Head image is None!")
                        ci += 2
                    else:
                        if head_img is not None and head_img.bgr is not None:
                            colors[f"color_{ci}"] = head_img.bgr
                        else:
                            frame_ok = False
                            logger_mp.warning("Head image is None!")
                        ci += 1
                    if camera_config['left_wrist_camera']['enable_zmq']:
                        if left_wrist_img is not None and left_wrist_img.bgr is not None:
                            colors[f"color_{ci}"] = left_wrist_img.bgr
                        else:
                            frame_ok = False
                            logger_mp.warning("Left wrist image is None!")
                        ci += 1
                    if camera_config['right_wrist_camera']['enable_zmq']:
                        if right_wrist_img is not None and right_wrist_img.bgr is not None:
                            colors[f"color_{ci}"] = right_wrist_img.bgr
                        else:
                            frame_ok = False
                            logger_mp.warning("Right wrist image is None!")
                        ci += 1
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    extra_kw = {"sim_state": sim_state_subscriber.read_data()} if args.sim else {}
                    if not frame_ok:
                        pass                           # image manquante : pas sauté (le convertisseur l'exige)
                    elif cal["paused"]:
                        pass                           # pause de calibration : pas non enregistrés
                    elif bridge is None:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, **extra_kw)
                        episode_steps += 1
                    elif not policy_paused:        # mode politique : pas de pas « tenus » dans les données
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions,
                                          extra={"intervention": int(policy_intervention)}, **extra_kw)
                        episode_steps += 1

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if getattr(arm_ctrl, "torso_yaw_index", None) is not None:
                arm_ctrl.set_torso_yaw(0.0)            # buste ramené droit, vitesse bornée
                time.sleep(args.torso_yaw_max / args.torso_yaw_rate + 0.5)
            arm_ctrl.ctrl_dual_arm_go_home(slow=True)  # retour au repos à vitesse réduite (audit du 1/10)
        except NameError:
            pass                                       # arrêt avant l'init des bras : rien à ramener
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if bridge is not None:
                bridge.close()
        except Exception as e:
            logger_mp.error(f"Failed to close policy bridge: {e}")

        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
