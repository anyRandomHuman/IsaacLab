# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat

def get_active_object_pos(
    env: ManagerBasedRLEnv,
    asset_names
):
    """
        Returns the (X, Y, Z) position of the currently active object from the pool.
        Assumes inactive objects are hidden far below the ground (e.g., Z = -100).
        """
    # 1. Gather the positions of every object in the pool
    # Each tensor is shape (num_envs, 3)
    positions = [env.scene[name].data.root_pos_w for name in asset_names]

    # Stack them into a single tensor of shape: (num_envs, num_objects, 3)
    stacked_positions = torch.stack(positions, dim=1)

    # 2. Find the index of the object with the highest Z-coordinate (index 2)
    # argmax returns the index of the max value along the object dimension
    # Shape: (num_envs,)
    active_indices = torch.argmax(stacked_positions[..., 2], dim=1)

    # 3. Extract the exact (X, Y, Z) for the active object in each environment
    # We use advanced PyTorch indexing to pluck the right object out of the stack
    batch_indices = torch.arange(env.num_envs, device=env.device)
    active_pos = stacked_positions[batch_indices, active_indices]

    return active_pos
