#!/usr/bin/env python3
# Copyright 2025 unifolm_wla community. All rights reserved.
# Licensed under the MIT License.
"""Unitree G1 msgpack/websocket action server.

Speaks the exact protocol of
(``WebsocketClientPolicy``):

  - On connect, the server immediately sends msgpack-packed ``metadata``. The
    client reads it as ``self._server_metadata`` and drives what obs keys it
    will send from ``metadata["data_keys"]``.
  - Each request is a msgpack dict ``{"type": <str>, ...}``. ``get_action``
    carries ``{"obs": {...}}`` and expects an action dict back; the other
    lifecycle types (``policy_reset`` / ``episode_end`` / ...) are no-ops for
    this stateless inference server.

Observation keys the client sends (declared in ``metadata["data_keys"]``):
    observation.images.cam_left_high     -> head_left camera   (H,W,3) uint8 BGR
    observation.images.cam_left_wrist    -> left  wrist camera
    observation.images.cam_right_wrist   -> right wrist camera
    observation.state.left_ee_6d         -> (9,) xyz + rot6d (first two cols)
    observation.state.right_ee_6d        -> (9,) xyz + rot6d
    observation.state.left_gripper       -> (1,)
    observation.state.right_gripper      -> (1,)
    observation.state.lower_body         -> (15,) left_leg(6)+right_leg(6)+waist(3)

Each obs value may carry a leading time/batch axis ((T,...) or (1,...)); the
server uses the LAST frame.

Action keys returned (whole predicted chunk, UNNORMALIZED, ABSOLUTE). Each key
carries a leading batch axis of 1:
    action.left_ee_rpy    (1, T, 6)  xyz + rpy, composed onto the current left EE
    action.right_ee_rpy   (1, T, 6)  xyz + rpy, composed onto the current right EE
    action.left_gripper   (1, T, 1)
    action.right_gripper  (1, T, 1)
    action.lower_body     (1, T, 15) left_leg(6)+right_leg(6)+waist(3)
    action.base_command   (1, T, 4)  vx, vy, vw, height
    action.pivot          (1, T, 7)  vx, vy, vw, yaw, pitch, roll, height
        vx/vy/vw/height are the same 4 values as base_command; yaw/pitch/roll
        come from the waist (lower_body last 3, ordered yaw,roll,pitch) remapped
        into pivot's yaw,pitch,roll order.

The EE state the client sends is ALREADY xyz+rot6d (the state layout the model
was trained on: STATE_SLICES["*_xyz_rot6d"]), so it is placed straight into the
unified state -- NOT run through map_state's pose_to_xyz_rot6d_from_format
(which expects a 6/7-dim rpy/quat pose). The rot6d convention (first two matrix
columns, [R00,R10,R20,R01,R11,R21]) matches se3_utils.matrix_to_rot6d exactly.

Usage:
    python -m model_server.action_server_wbc_msgpack_unitree \\
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_<N>_model.safetensors \\
        --host 0.0.0.0 --port 8600 --instruction "pick up the object"
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import websockets.asyncio.server as _ws_server
import websockets.frames
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from tools import msgpack_numpy  # noqa: E402
from tools.debug_saver import StepDebugSaver  # noqa: E402
from unifolm_wla_action_adapter import (  # noqa: E402
    WBC_IMAGE_ROLES,
    build_unitree_fullbody_action_mask,
    _compose_ee9,
    _rot6d_to_matrix,
)

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES, STATE_DIM  # noqa: E402

from unifolm_wla.model.framework.base_framework import baseframework  # noqa: E402
from unifolm_wla.model.framework.share_tools import read_mode_config  # noqa: E402

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Client obs keys -> image role order (matches WBC_IMAGE_ROLES).
_IMAGE_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# The obs / action contract advertised to the client via metadata["data_keys"].
_DATA_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.state.left_ee_6d",
    "observation.state.right_ee_6d",
    "observation.state.left_gripper",
    "observation.state.right_gripper",
    "observation.state.lower_body",
    "action.left_ee_rpy",
    "action.right_ee_rpy",
    "action.left_gripper",
    "action.right_gripper",
    "action.lower_body",
    "action.base_command",
    "action.pivot",
]


def _last_frame(arr: np.ndarray) -> np.ndarray:
    """Client may send obs with a leading (T,...)/(1,...) axis. For a state
    vector that means ndim==2 -> take the last row; for an image ndim==4 ->
    take the last frame. A bare vector/image is returned unchanged."""
    a = np.asarray(arr)
    if a.ndim == 2 or a.ndim == 4:
        return a[-1]
    return a


def _compose_ee_to_xyz_rpy(anchor_9: np.ndarray, rel_6_chunk: np.ndarray) -> np.ndarray:
    """Compose a (T,6) xyz+rotvec RELATIVE EE chunk onto a (9,) xyz+rot6d
    ABSOLUTE EE state anchor and return (T,6) xyz + rpy (euler 'xyz').

    Reuses the adapter's `_compose_ee9` for the SE3 compose (T_abs = T_curr @
    T_rel -> xyz+rot6d) and only converts the trailing rot6d to rpy, which is
    the client's action.left/right_ee_rpy convention.
    """
    ee9 = _compose_ee9(anchor_9, rel_6_chunk)                   # (T,9) xyz+rot6d
    xyz_abs = ee9[:, :3]
    rpy_abs = Rotation.from_matrix(_rot6d_to_matrix(ee9[:, 3:9])).as_euler("xyz")
    return np.concatenate([xyz_abs, rpy_abs], axis=1).astype(np.float32)  # (T,6)


class ActionServerWBCMsgpack:

    def __init__(self, args):
        self.args = args
        self.instruction = args.instruction
        logging.info("Loading model from: %s", args.ckpt_path)
        self.model = baseframework.from_pretrained(args.ckpt_path)
        if args.use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(DEVICE).eval()

        mode_config, _ = read_mode_config(args.ckpt_path)
        am_cfg = mode_config.get("framework", {}).get("action_model", {})
        self._action_chunk_size = int(am_cfg.get("action_horizon", 30))
        logging.info("action_chunk_size=%d", self._action_chunk_size)

        self._norm_stats_all = self.model.norm_stats
        self._norm_arrays = {
            key: {
                "action_offset": np.array(stats["action"]["offset"], dtype=np.float32),
                "action_scale": np.array(stats["action"]["scale"], dtype=np.float32),
                "state_offset": np.array(stats["state"]["offset"], dtype=np.float32),
                "state_scale": np.array(stats["state"]["scale"], dtype=np.float32),
            }
            for key, stats in self._norm_stats_all.items()
        }
        if args.unnorm_key is not None:
            self._default_unnorm_key = args.unnorm_key
        elif len(self._norm_stats_all) == 1:
            self._default_unnorm_key = next(iter(self._norm_stats_all.keys()))
        else:
            self._default_unnorm_key = None
            logging.warning(
                "Multi-dataset checkpoint; client must send unnorm_key per request. Available: %s",
                list(self._norm_stats_all.keys()))

        self._action_mask = build_unitree_fullbody_action_mask()  # (54,) bool
        self._image_size = tuple(args.image_size) if args.image_size and all(args.image_size) else None
        logging.info("image_size=%s", self._image_size)

        cv2.setNumThreads(0)
        self._img_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="imgdecode")

        self._debug_saver = StepDebugSaver(args.debug_save_dir)
        # Real-time chunking (ajout G1-D) : dernier chunk prédit, pour construire un préfixe.
        self._rtc_prev = None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "env": "unifolm_wla_unitree_wbc_msgpack",
            "ckpt_path": self.args.ckpt_path,
            "data_keys": _DATA_KEYS,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": list(self._norm_stats_all.keys()),
            "default_unnorm_key": self._default_unnorm_key,
        }

    # ── obs -> unified state ───────────────────────────────────────────────

    def _resolve_unnorm_key(self, obs: dict) -> str:
        req_key = obs.get("unnorm_key")
        if req_key and req_key in self._norm_stats_all:
            return req_key
        if self._default_unnorm_key is not None:
            return self._default_unnorm_key
        raise ValueError(
            f"No unnorm_key in obs and no default set. Available: {list(self._norm_stats_all.keys())}")

    def _build_state_unnorm(self, obs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Client obs keys -> unified (60,) state_unnorm + (60,) state_mask.

        The EE poses arrive ALREADY as xyz+rot6d (9-dim), so they go straight
        into STATE_SLICES["*_xyz_rot6d"] with no format conversion.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        mask = np.zeros(STATE_DIM, dtype=bool)

        left_ee = _last_frame(obs["observation.state.left_ee_6d"]).astype(np.float32).ravel()
        right_ee = _last_frame(obs["observation.state.right_ee_6d"]).astype(np.float32).ravel()
        state[STATE_SLICES["left_xyz_rot6d"]] = left_ee[:9]
        mask[STATE_SLICES["left_xyz_rot6d"]] = True
        state[STATE_SLICES["right_xyz_rot6d"]] = right_ee[:9]
        mask[STATE_SLICES["right_xyz_rot6d"]] = True

        left_grip = _last_frame(obs["observation.state.left_gripper"]).astype(np.float32).ravel()
        right_grip = _last_frame(obs["observation.state.right_gripper"]).astype(np.float32).ravel()
        state[STATE_SLICES["left_gripper"]] = left_grip[:1]
        mask[STATE_SLICES["left_gripper"]] = True
        state[STATE_SLICES["right_gripper"]] = right_grip[:1]
        mask[STATE_SLICES["right_gripper"]] = True

        # lower_body = left_leg(6) + right_leg(6) + waist(3)
        lower = _last_frame(obs["observation.state.lower_body"]).astype(np.float32).ravel()
        state[STATE_SLICES["left_leg_joint"]] = lower[0:6]
        mask[STATE_SLICES["left_leg_joint"]] = True
        state[STATE_SLICES["right_leg_joint"]] = lower[6:12]
        mask[STATE_SLICES["right_leg_joint"]] = True
        state[STATE_SLICES["waist_joint"]] = lower[12:15]
        mask[STATE_SLICES["waist_joint"]] = True

        return state, mask

    def _decode_one_image(self, key_obs: Tuple[str, dict]) -> Optional[np.ndarray]:
        """Client obs[key] -> (H,W,3) uint8 RGB, resized to self._image_size. Runs on
        the image thread pool; cv2 releases the GIL for resize/cvtColor."""
        key, obs = key_obs
        if key not in obs:
            return None
        img = _last_frame(obs[key])            # (H,W,3) uint8 BGR (OpenCV)
        if self._image_size is not None:
            target_h, target_w = self._image_size
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # cv2: (W,H)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _rtc_prefix(self, obs: dict, unnorm_key: str, state_unnorm: np.ndarray):
        """Préfixe (H, 54) normalisé + poids (H,) à partir du chunk précédent, ou (None, None).

        Le client indique ``rtc_executed`` (pas du chunk précédent déjà exécutés) et
        ``rtc_prefix`` (nombre de pas suivants à imposer). Les slots effecteurs, relatifs à
        l'ancre, sont RÉ-EXPRIMÉS par rapport à la nouvelle ancre mesurée ; les autres slots
        reprennent les valeurs normalisées précédentes."""
        d = int(obs.get("rtc_prefix", 0) or 0)
        e = int(obs.get("rtc_executed", 0) or 0)
        prev = self._rtc_prev
        if d <= 0 or prev is None or prev["unnorm_key"] != unnorm_key or e + d > len(prev["pred_norm"]):
            return None, None
        H = self._action_chunk_size
        norm = self._norm_arrays[unnorm_key]
        prefix = np.zeros((H, prev["pred_norm"].shape[1]), dtype=np.float32)
        prefix[:d] = prev["pred_norm"][e:e + d]
        for side in ("left", "right"):
            anchor = state_unnorm[STATE_SLICES[f"{side}_xyz_rot6d"]]
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = _rot6d_to_matrix(anchor[3:9].reshape(1, 6))[0], anchor[:3]
            ee9 = prev["abs_ee9"][side][e:e + d]
            Tb = np.tile(np.eye(4), (d, 1, 1))
            Tb[:, :3, :3] = _rot6d_to_matrix(ee9[:, 3:9])
            Tb[:, :3, 3] = ee9[:, :3]
            rel = np.linalg.inv(A)[None] @ Tb
            rel6 = np.concatenate([rel[:, :3, 3], Rotation.from_matrix(rel[:, :3, :3]).as_rotvec()], axis=1)
            sl = SLICES[f"{side}_xyz_rotvec"]
            prefix[:d, sl] = (rel6 - norm["action_offset"][sl]) / (norm["action_scale"][sl] + 1e-8)
        weights = np.zeros(H, dtype=np.float32)
        weights[:d] = 1.0
        # raccord doux (RTC, Black et al. 2025) : au-delà du préfixe, poids décroissant
        # exponentiellement sur ``rtc_soft`` pas, en prolongeant la dernière action connue
        soft = int(obs.get("rtc_soft", 0) or 0)
        if soft > 0 and d < H:
            n = min(soft, H - d)
            k = np.arange(1, n + 1, dtype=np.float32)
            weights[d:d + n] = np.exp(-3.0 * k / n)
            prefix[d:d + n] = prefix[d - 1]
        return prefix, weights

    def _build_example(self, obs: dict) -> dict:
        unnorm_key = self._resolve_unnorm_key(obs)

        images = [img for img in self._img_pool.map(self._decode_one_image, ((k, obs) for k in _IMAGE_KEYS))
                  if img is not None]

        state_unnorm, state_mask = self._build_state_unnorm(obs)
        norm = self._norm_arrays[unnorm_key]
        state_norm = (state_unnorm - norm["state_offset"]) / (norm["state_scale"] + 1e-8)

        instruction = obs.get("instruction") or obs.get("task") or self.instruction

        example = {
            "image": images,
            "image_roles": WBC_IMAGE_ROLES[: len(images)],
            "lang": instruction,
            "action_mask": self._action_mask,
            "state": state_norm.astype(np.float32),
            "state_mask": state_mask,
            "arm_type": "dual_with_legs",
            "robot_type": "unitree",
        }
        prefix, weights = self._rtc_prefix(obs, unnorm_key, state_unnorm)
        if prefix is not None:
            example["action_prefix"] = prefix
            example["action_prefix_weights"] = weights
        return {"example": example, "unnorm_key": unnorm_key, "state_unnorm": state_unnorm}

    # ── unified action -> client action dict ───────────────────────────────

    def _encode_action(self, pred_norm: np.ndarray, unnorm_key: str, state_unnorm: np.ndarray) -> Dict[str, np.ndarray]:
        norm = self._norm_arrays[unnorm_key]
        unnorm = (pred_norm * norm["action_scale"] + norm["action_offset"]).astype(np.float32)  # (T,54)

        left_anchor = state_unnorm[STATE_SLICES["left_xyz_rot6d"]]
        right_anchor = state_unnorm[STATE_SLICES["right_xyz_rot6d"]]
        left_ee_rpy = _compose_ee_to_xyz_rpy(left_anchor, unnorm[:, SLICES["left_xyz_rotvec"]])    # (T,6)
        right_ee_rpy = _compose_ee_to_xyz_rpy(right_anchor, unnorm[:, SLICES["right_xyz_rotvec"]])  # (T,6)

        lower_body = np.concatenate([
            unnorm[:, SLICES["left_leg_joint"]],    # (T,6)
            unnorm[:, SLICES["right_leg_joint"]],   # (T,6)
            unnorm[:, SLICES["waist_joint"]],       # (T,3)
        ], axis=1).astype(np.float32)               # (T,15)

        left_gripper = unnorm[:, SLICES["left_gripper"]].astype(np.float32)    # (T,1)
        right_gripper = unnorm[:, SLICES["right_gripper"]].astype(np.float32)  # (T,1)

        vx_vy = unnorm[:, SLICES["base_vx_vy"]]     # (T,2)
        vw = unnorm[:, SLICES["base_vw"]]           # (T,1)
        height = unnorm[:, SLICES["height"]]        # (T,1)
        base_command = np.concatenate([vx_vy, vw, height], axis=1).astype(np.float32)  # (T,4)

        # pivot = [vx, vy, vw, yaw, pitch, roll, height]. The waist (lower_body
        # last 3) is ordered [yaw, roll, pitch], so remap: yaw=waist[0],
        # pitch=waist[2], roll=waist[1].
        waist = unnorm[:, SLICES["waist_joint"]]    # (T,3) = [yaw, roll, pitch]
        pivot = np.concatenate([
            vx_vy, vw,
            waist[:, 0:1],   # yaw
            waist[:, 2:3],   # pitch
            waist[:, 1:2],   # roll
            height,
        ], axis=1).astype(np.float32)               # (T,7)

        # Each key carries a leading batch axis of 1: (1, T, D). The client
        # indexes action_pred[key][0, :chunk].
        return {
            "action.left_ee_rpy": left_ee_rpy[None],
            "action.right_ee_rpy": right_ee_rpy[None],
            "action.left_gripper": left_gripper[None],
            "action.right_gripper": right_gripper[None],
            "action.lower_body": lower_body[None],
            "action.base_command": base_command[None],
            "action.pivot": pivot[None],
        }

    # ── inference ──────────────────────────────────────────────────────────

    def get_action(self, obs: dict) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        prep = self._build_example(obs)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model.predict_action(examples=[prep["example"]])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        pred_norm = out["normalized_actions"][0]   # (T,54)
        action = self._encode_action(pred_norm, prep["unnorm_key"], prep["state_unnorm"])
        su = prep["state_unnorm"]
        norm = self._norm_arrays[prep["unnorm_key"]]
        unnorm = (pred_norm * norm["action_scale"] + norm["action_offset"]).astype(np.float32)
        self._rtc_prev = {
            "unnorm_key": prep["unnorm_key"], "pred_norm": np.asarray(pred_norm, np.float32),
            "abs_ee9": {s: _compose_ee9(su[STATE_SLICES[f"{s}_xyz_rot6d"]], unnorm[:, SLICES[f"{s}_xyz_rotvec"]])
                        for s in ("left", "right")},
        }

        self._debug_saver.save_step(
            images=prep["example"]["image"],
            left_pose=action["action.left_ee_rpy"][0],
            right_pose=action["action.right_ee_rpy"][0],
            prefix_len=0,
            chunk_id=0,
            state_unnorm=prep["state_unnorm"],
            pred_norm=pred_norm,
            unnorm_key=prep["unnorm_key"],
        )
        t3 = time.perf_counter()
        logging.info("[get_action] T=%d preprocess=%.1fms predict=%.1fms encode=%.1fms",
                     pred_norm.shape[0], (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
        return action

    # ── websocket serving (mirrors robotdeploy policy_server.py) ───────────

    async def _handler(self, websocket: "_ws_server.ServerConnection"):
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                mtype = msg["type"]

                if mtype == "get_action":
                    return_data = self.get_action(msg["obs"])
                elif mtype in ("policy_reset", "update_dataset"):
                    self._rtc_prev = None
                    return_data = {}
                elif mtype in ("policy_train", "policy_save", "policy_load", "episode_end", "obs_init"):
                    return_data = None
                elif mtype == "obs_chunk":
                    # streaming obs: no response expected by the client
                    continue
                else:
                    raise ValueError(f"Invalid message type: {mtype}")

                await websocket.send(packer.pack(return_data))

            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _run(self, host: str, port: int):
        async with _ws_server.serve(
            self._handler, host, port,
            compression=None, max_size=None, ping_interval=None, ping_timeout=None,
        ) as server:
            await server.serve_forever()

    def run(self, host: str = "0.0.0.0", port: int = 8600):
        logging.info("WBC msgpack server (unifolm_wla/Unitree) listening on ws://%s:%d", host, port)
        logging.info("metadata=%s", self.metadata)
        asyncio.run(self._run(host, port))


def main():
    parser = argparse.ArgumentParser(
        description="unifolm_wla Unitree G1 msgpack/websocket action server (unitree_rl_wbc client protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint .pt/.safetensors")
    parser.add_argument("--instruction", default="",
                        help="Default language instruction if obs carries none (obs['instruction']/'task' override).")
    parser.add_argument("--unnorm_key", default=None,
                        help="Dataset key for norm stats (auto-detected if single dataset)")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 448], metavar=("H", "W"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--debug_save_dir", default=None,
                        help="If set, save a per-step camera+action debug image to this "
                             "directory on a background thread.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    server = ActionServerWBCMsgpack(args)
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
