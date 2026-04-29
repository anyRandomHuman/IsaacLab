from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sim.views import XformPrimView

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _is_mapping_like(obj) -> bool:
    return isinstance(obj, dict) or (hasattr(obj, "keys") and callable(obj.keys))


def build_getter(root, path: str):
    parts: list[str | tuple[str, int]] = []
    for part in path.split("."):
        match = re.compile(r"^(\w+)\[(\d+)\]$").match(part)
        if match:
            parts.append((match.group(1), int(match.group(2))))
        else:
            parts.append(part)

    def _getter():
        value = root
        for part in parts:
            if isinstance(part, tuple):
                name, idx = part
                value = value[name] if _is_mapping_like(value) else getattr(value, name)
                value = value[idx]
            else:
                value = value[part] if _is_mapping_like(value) else getattr(value, part)
        return value

    return _getter


class CentralADRManager(ManagerTermBase):
    """Central coordinator that assigns random boundary-eval modes per reset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.total_dims = 0
        self.train_ratio = float(getattr(cfg, "train_ratio", 0.6))
        self.train_envs_count = int(self.num_envs * self.train_ratio)

        self.modes = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.prev_modes = self.modes.clone()
        env.central_adr_manager = self

    def register_worker(self, num_dims: int) -> int:
        offset = self.total_dims
        self.total_dims += int(num_dims)
        return offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.prev_modes[env_ids] = self.modes[env_ids]

        if self.total_dims == 0:
            self.modes[env_ids] = -1
            return

        modes = torch.full(env_ids.shape, -1, device=self.device, dtype=torch.long)
        train_mask = torch.rand(modes.shape, device=self.device) < self.train_ratio
        param_ids = torch.randint(self.total_dims, modes.shape, device=self.device, dtype=torch.long)
        self.modes[env_ids] = torch.where(train_mask, modes, param_ids)

    def get_reset_instructions(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return self.modes[ids].clone()

    def get_episode_result_instructions(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return self.prev_modes[ids].clone()

    def __call__(self, *args):
        return None


class ADRDifficultyCore:
    """Reusable ADR difficulty state machine independent of asset/property application."""

    def __init__(
        self,
        *,
        device: torch.device,
        num_envs: int,
        central_adr_manager: CentralADRManager,
        param_shape: int,
        eval_interval: int,
        max_difficulty: int,
        upgrade_threshold: float,
        downgrade_threshold: float,
    ):
        if param_shape <= 0:
            raise ValueError(f"ADRDifficultyCore requires param_shape > 0. Got: {param_shape}.")
        self.device = device
        self.num_envs = int(num_envs)
        self.param_shape = int(param_shape)
        self.eval_interval = int(eval_interval)
        self.max_difficulty = int(max_difficulty)
        self.upgrade_threshold = float(upgrade_threshold)
        self.downgrade_threshold = float(downgrade_threshold)

        self.mode_offset = int(central_adr_manager.register_worker(self.param_shape))
        self.mode_range = torch.arange(self.mode_offset, self.mode_offset + self.param_shape, device=self.device, dtype=torch.long)
        self.difficulties = torch.zeros(self.param_shape, device=self.device)
        self.consecutive_success_counter = torch.zeros((self.param_shape, self.eval_interval), device=self.device)
        self.eval_counts = torch.zeros(self.param_shape, dtype=torch.long, device=self.device)
        self.first_update = True

    def update_from_episode(
        self,
        env_ids: torch.Tensor,
        central_adr_manager: CentralADRManager,
        success_values: torch.Tensor,
    ) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        if self.first_update:
            self.first_update = False
            return

        success_values = torch.as_tensor(success_values, device=self.device).reshape(-1)
        if success_values.shape[0] != self.num_envs:
            raise ValueError(
                f"ADRDifficultyCore expected success values with leading dim num_envs={self.num_envs}, "
                f"got shape {tuple(success_values.shape)}."
            )

        reset_modes = central_adr_manager.get_episode_result_instructions(env_ids).to(self.device)
        is_bound_envs = torch.isin(reset_modes, self.mode_range)
        if not is_bound_envs.any():
            return

        bound_env_ids = env_ids[is_bound_envs]
        bound_results = success_values[bound_env_ids].float()
        param_idx = (reset_modes[is_bound_envs] - self.mode_offset).long()

        for idx in torch.unique(param_idx):
            idx_int = int(idx.item())
            idx_mask = param_idx == idx
            idx_results = bound_results[idx_mask]
            num_new = int(idx_results.numel())
            start = int(self.eval_counts[idx_int].item())
            insert_indices = (start + torch.arange(num_new, device=self.device)) % self.eval_interval
            self.consecutive_success_counter[idx_int, insert_indices] = idx_results
            self.eval_counts[idx_int] += num_new

            if int(self.eval_counts[idx_int].item()) < self.eval_interval:
                continue

            avg_success = float(self.consecutive_success_counter[idx_int].mean().item())
            if avg_success >= self.upgrade_threshold:
                self.difficulties[idx_int] = torch.clamp(self.difficulties[idx_int] + 1.0, 0.0, float(self.max_difficulty))
                self.consecutive_success_counter[idx_int] = 0.0
                self.eval_counts[idx_int] = 0
            elif avg_success <= self.downgrade_threshold:
                self.difficulties[idx_int] = torch.clamp(self.difficulties[idx_int] - 1.0, 0.0, float(self.max_difficulty))
                self.consecutive_success_counter[idx_int] = 0.0
                self.eval_counts[idx_int] = 0

    def sample_uniform_values(
        self,
        *,
        env_ids: torch.Tensor,
        reset_modes: torch.Tensor,
        initial_low: torch.Tensor,
        initial_high: torch.Tensor,
        step_size: torch.Tensor,
        limit_low: torch.Tensor,
        limit_high: torch.Tensor,
    ) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return torch.zeros((0, initial_low.shape[-1]), device=self.device, dtype=initial_low.dtype)

        low_diff, high_diff = self.difficulties[0::2], self.difficulties[1::2]
        low = initial_low[env_ids] - (low_diff[None] * step_size[env_ids])
        low = torch.clamp(low, min=limit_low[env_ids], max=limit_high[env_ids])
        high = initial_high[env_ids] + (high_diff[None] * step_size[env_ids])
        high = torch.clamp(high, min=limit_low[env_ids], max=limit_high[env_ids])

        values = torch.rand_like(high) * (high - low) + low

        is_bound_envs = torch.isin(reset_modes, self.mode_range)
        if is_bound_envs.any():
            param_idx = (reset_modes[is_bound_envs] - self.mode_offset).long()
            physical_dim_idx = param_idx // 2
            is_lower_bound = (param_idx % 2 == 0)
            bound_rows = torch.nonzero(is_bound_envs).squeeze(-1)
            lower_rows = bound_rows[is_lower_bound]
            upper_rows = bound_rows[~is_lower_bound]
            lower_dims = physical_dim_idx[is_lower_bound]
            upper_dims = physical_dim_idx[~is_lower_bound]
            values[lower_rows, lower_dims] = low[lower_rows, lower_dims]
            values[upper_rows, upper_dims] = high[upper_rows, upper_dims]

        return values

    def summarize(self, max_difficulty: float) -> dict[str, torch.Tensor]:
        difficulties = self.difficulties.float()
        return {
            "mean": difficulties.mean(),
            "std": difficulties.std(unbiased=False),
            "min": difficulties.min(),
            "max": difficulties.max(),
            "p50": torch.quantile(difficulties, 0.5),
            "p90": torch.quantile(difficulties, 0.9),
            "frac_at_max": (difficulties >= float(max_difficulty)).float().mean(),
        }


class ConsecutiveSuccessADR(ManagerTermBase):
    """
    Refactored ADR worker:
    - difficulty/mode state in ADRDifficultyCore
    - subclasses only define defaults/limits and `set_values`.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg = cfg.params.get("asset_cfg", None)
        self.asset: Articulation | RigidObject | None = None
        if self.asset_cfg is not None:
            self.asset = env.scene[self.asset_cfg.name]

        self.max_difficulty = int(cfg.params.get("max_difficulty", 20))
        self.upgrade_threshold = float(cfg.params.get("upgrade_threshold", 0.7))
        self.downgrade_threshold = float(cfg.params.get("downgrade_threshold", 0.1))
        self.eval_interval = int(cfg.params.get("eval_interval", 256))

        self.log_every_resets = max(1, int(cfg.params.get("log_every_resets", 1)))
        self.log_watch_indices = [int(i) for i in cfg.params.get("log_watch_indices", [])]
        self._log_counter = 0

        success_path = cfg.params.get(
            "success_key",
            "command_manager._terms.object_pose.metrics.consecutive_success",
        )
        self.success_getter = build_getter(self._env, success_path)

        self.central_adr_manager: CentralADRManager = env.central_adr_manager
        self._external_adr_core: ADRDifficultyCore | None = cfg.params.get("shared_adr_core", None)
        self._owns_difficulty_updates = bool(cfg.params.get("owns_difficulty_updates", self._external_adr_core is None))

        self.defaults: torch.Tensor
        self._set_defaults()

        self._defer_adr_init = bool(getattr(self, "_defer_adr_init", False))
        self.param_shape = (
            int(getattr(self, "param_shape"))
            if hasattr(self, "param_shape")
            else (0 if self._defer_adr_init else 2 * int(self._resolve_feature_dim()))
        )

        self.limits_set = False
        self._initialized = False
        self.adr_core: ADRDifficultyCore | None = None
        if not self._defer_adr_init:
            self._initialize_worker_state()

    @property
    def difficulties(self) -> torch.Tensor:
        return self.adr_core.difficulties if self.adr_core is not None else torch.zeros(0, device=self.device)

    @property
    def mode_offset(self) -> int:
        return int(self.adr_core.mode_offset) if self.adr_core is not None else -1

    @property
    def mode_range(self) -> torch.Tensor:
        if self.adr_core is None:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return self.adr_core.mode_range

    @property
    def consecutive_success_counter(self) -> torch.Tensor:
        if self.adr_core is None:
            return torch.zeros((0, self.eval_interval), device=self.device)
        return self.adr_core.consecutive_success_counter

    @property
    def eval_counts(self) -> torch.Tensor:
        if self.adr_core is None:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return self.adr_core.eval_counts

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _resolve_feature_dim(self) -> int:
        if self.defaults is None:
            raise ValueError(f"{self.__class__.__name__} defaults must be initialized before resolving feature dim.")
        return int(self.defaults.shape[-1])

    def _initialize_worker_state(self) -> None:
        if self._initialized:
            return

        if not self.limits_set:
            self._set_limits_and_stepsize()

        if self._external_adr_core is not None:
            self.adr_core = self._external_adr_core
            if self.param_shape <= 0:
                self.param_shape = int(self.adr_core.param_shape)
        else:
            feature_dim = self._resolve_feature_dim()
            if self.param_shape <= 0:
                self.param_shape = 2 * feature_dim
            self.adr_core = ADRDifficultyCore(
                device=self.device,
                num_envs=self.num_envs,
                central_adr_manager=self.central_adr_manager,
                param_shape=self.param_shape,
                eval_interval=self.eval_interval,
                max_difficulty=self.max_difficulty,
                upgrade_threshold=self.upgrade_threshold,
                downgrade_threshold=self.downgrade_threshold,
            )

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._initialize_worker_state()

    def _expand_param(self, value, default_value: torch.Tensor) -> torch.Tensor:
        if value is None:
            tensor = default_value.clone()
        elif isinstance(value, torch.Tensor):
            tensor = value.to(default_value.device, dtype=default_value.dtype)
        else:
            tensor = torch.tensor(value, device=default_value.device, dtype=default_value.dtype)

        feature_dim = int(default_value.shape[-1])

        if tensor.ndim == 0:
            tensor = tensor.repeat(feature_dim)
        if tensor.ndim == 1:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(feature_dim)
            elif tensor.shape[0] != feature_dim:
                raise ValueError(
                    f"{self.__class__.__name__} parameter has wrong shape {tuple(tensor.shape)} for feature dim={feature_dim}."
                )
            return tensor.unsqueeze(0).repeat(self.num_envs, 1)
        if tensor.ndim == 2:
            if tensor.shape[1] != feature_dim:
                raise ValueError(
                    f"{self.__class__.__name__} parameter has wrong shape {tuple(tensor.shape)} for feature dim={feature_dim}."
                )
            if tensor.shape[0] == 1:
                return tensor.repeat(self.num_envs, 1)
            if tensor.shape[0] != self.num_envs:
                raise ValueError(
                    f"{self.__class__.__name__} parameter has wrong env dimension {tensor.shape[0]} for num_envs={self.num_envs}."
                )
            return tensor

        raise ValueError(f"{self.__class__.__name__} parameter rank must be <= 2. Got: {tensor.ndim}.")

    def sample_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        reset_modes = self.central_adr_manager.get_reset_instructions(env_ids).to(self.device)
        return self.adr_core.sample_uniform_values(
            env_ids=env_ids,
            reset_modes=reset_modes,
            initial_low=self.initial_low,
            initial_high=self.initial_high,
            step_size=self.step_size,
            limit_low=self.limit_low,
            limit_high=self.limit_high,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._ensure_initialized()
        env_ids_t = self._normalize_env_ids(env_ids)
        if len(env_ids_t) == 0:
            return

        if self._owns_difficulty_updates:
            self.adr_core.update_from_episode(
                env_ids=env_ids_t,
                central_adr_manager=self.central_adr_manager,
                success_values=self.success_getter(),
            )

        values = self.sample_values(env_ids_t)
        self.set_values(env_ids_t, values)

    def change_property(self, env_ids, values):
        # Compatibility shim for older subclasses/call sites.
        self.set_values(env_ids, values)

    def set_values(self, env_ids, values):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self._ensure_initialized()
        self._log_counter += 1
        if self._log_counter % self.log_every_resets != 0:
            return None
        out = self.adr_core.summarize(self.max_difficulty)
        for idx in self.log_watch_indices:
            if 0 <= idx < self.difficulties.numel():
                out[f"watch_{idx}"] = self.difficulties[idx]
        self._env.max_difficulty = self.difficulties.max() if self.difficulties.numel() > 0 else torch.tensor(0.0)
        self._env.mean_difficulty = self.difficulties.mean() if self.difficulties.numel() > 0 else torch.tensor(0.0)
        return out

    def _set_defaults(self):
        raise NotImplementedError

    def _set_limits_and_stepsize(self):
        step_default = self.defaults / 100.0
        self.initial_low = self._expand_param(self.cfg.params.get("initial_low", None), self.defaults)
        self.initial_high = self._expand_param(self.cfg.params.get("initial_high", None), self.defaults)
        self.step_size = self._expand_param(self.cfg.params.get("step_size", None), step_default)
        self.limit_low = self._expand_param(self.cfg.params.get("limit_low", None), torch.zeros_like(self.defaults))
        self.limit_high = self._expand_param(self.cfg.params.get("limit_high", None), 2.0 * self.defaults)
        self.limits_set = True


class RobotCSADR(ConsecutiveSuccessADR):
    def __init__(self, cfg, env):
        asset_cfg = cfg.params.get("asset_cfg")
        assert isinstance(env.scene[asset_cfg.name], Articulation)
        if not asset_cfg.joint_names:
            asset_cfg.joint_names = ".*"
        if not asset_cfg.body_names:
            asset_cfg.body_names = ".*"
        self.joint_ids, self.joint_names = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names)
        self.body_ids, self.body_names = env.scene[asset_cfg.name].find_bodies(asset_cfg.body_names)
        super().__init__(cfg, env)


class RobotJointCSADR(RobotCSADR):
    pass


class RobotBodyCSADR(RobotCSADR):
    pass


class RobotJointStiffnessCSADR(RobotJointCSADR):
    def set_values(self, env_ids, values):
        self.asset.write_joint_stiffness_to_sim(values, env_ids=env_ids, joint_ids=self.joint_ids)

    def _set_defaults(self):
        self.defaults = self.asset.data.joint_stiffness[:, self.joint_ids].clone()


class RobotDampingADR(RobotJointCSADR):
    def set_values(self, env_ids, values):
        self.asset.write_joint_damping_to_sim(values, env_ids=env_ids, joint_ids=self.joint_ids)

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_damping[:, self.joint_ids].clone()


class RobotJointStaticFrictionADR(RobotJointCSADR):
    def set_values(self, env_ids, values):
        self.asset.write_joint_friction_coefficient_to_sim(values, env_ids=env_ids, joint_ids=self.joint_ids)

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_friction_coeff[:, self.joint_ids].clone()


class RobotJointDynamicFrictionADR(RobotJointCSADR):
    def set_values(self, env_ids, values):
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=self.asset.data.joint_friction_coeff[env_ids[:, None], self.joint_ids],
            joint_dynamic_friction_coeff=values,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_dynamic_friction_coeff[:, self.joint_ids].clone()


class RobotJointViscousFrictionADR(RobotJointCSADR):
    def set_values(self, env_ids, values):
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=self.asset.data.joint_friction_coeff[env_ids[:, None], self.joint_ids],
            joint_viscous_friction_coeff=values,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_viscous_friction_coeff[:, self.joint_ids].clone()


class IndividualArmatureADR(ConsecutiveSuccessADR):
    def __init__(self, cfg, env):
        self.actuator = cfg.params.get("actuators", "joint")
        super().__init__(cfg, env)

    def set_values(self, env_ids, values):
        actuator = self.asset.actuators[self.actuator]
        if isinstance(self._selected_joint_ids_local, slice):
            actuator.armature[env_ids] = values.to(actuator.armature.device)
        else:
            actuator.armature[env_ids[:, None], self._selected_joint_ids_local] = values.to(actuator.armature.device)
        self.asset.write_joint_armature_to_sim(values, joint_ids=self._selected_joint_ids_global, env_ids=env_ids)

    def _set_defaults(self):
        actuator = self.asset.actuators[self.actuator]
        self._selected_joint_ids_global, self._selected_joint_ids_local = self._resolve_selected_joint_ids(actuator)
        if isinstance(self._selected_joint_ids_local, slice):
            self.defaults = actuator.armature.to(self.device).clone()
        else:
            self.defaults = actuator.armature[:, self._selected_joint_ids_local].to(self.device).clone()

    def _resolve_actuator_global_joint_ids(self, actuator) -> torch.Tensor:
        if isinstance(actuator.joint_indices, slice):
            return torch.arange(self.asset.num_joints, device=self.device, dtype=torch.long)[actuator.joint_indices]
        if isinstance(actuator.joint_indices, torch.Tensor):
            return actuator.joint_indices.to(self.device).long()
        raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")

    def _resolve_selected_joint_ids(self, actuator) -> tuple[slice | torch.Tensor, slice | torch.Tensor]:
        joint_ids = self.cfg.params.get("joint_ids", None)
        joint_names = self.cfg.params.get("joint_names", None)
        if joint_ids is None and self.asset_cfg is not None and self.asset_cfg.joint_ids != slice(None):
            joint_ids = self.asset_cfg.joint_ids
        if joint_names is None and self.asset_cfg is not None and self.asset_cfg.joint_names is not None:
            joint_names = self.asset_cfg.joint_names

        if joint_ids is None and joint_names is None:
            return self._resolve_actuator_global_joint_ids(actuator), slice(None)

        selected_global_ids = None
        if joint_ids is not None:
            if isinstance(joint_ids, slice):
                selected_global_ids = torch.arange(self.asset.num_joints, device=self.device, dtype=torch.long)[joint_ids]
            else:
                selected_global_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device).view(-1)

        if joint_names is not None:
            joint_expr = joint_names if isinstance(joint_names, str) else "|".join(joint_names)
            matched_ids, _ = self.asset.find_joints(joint_expr)
            matched_ids = torch.as_tensor(matched_ids, dtype=torch.long, device=self.device).view(-1)
            if selected_global_ids is None:
                selected_global_ids = matched_ids
            else:
                selected_global_ids = selected_global_ids[torch.isin(selected_global_ids, matched_ids)]

        if selected_global_ids is None or selected_global_ids.numel() == 0:
            raise ValueError(
                f"IndividualArmatureADR for actuator '{self.actuator}' received no valid joint selection."
            )

        actuator_global_ids = self._resolve_actuator_global_joint_ids(actuator)
        is_selected = torch.isin(actuator_global_ids, selected_global_ids)
        local_indices = torch.nonzero(is_selected).view(-1)
        if local_indices.numel() == 0:
            raise ValueError(
                f"Selected joints {selected_global_ids.tolist()} are not part of actuator '{self.actuator}'."
            )
        return actuator_global_ids[local_indices], local_indices


class RobotArmatureADR(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg")
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.actuators = getattr(cfg, "actuators", list(self.asset.actuators.keys()))
        self.adrs = []
        original_params = cfg.params
        requested_global_ids = self._resolve_requested_global_joint_ids(original_params)

        for actuator_name in self.actuators:
            cfg.params = original_params | {"actuators": actuator_name}
            if requested_global_ids is not None:
                joint_indices = self.asset.actuators[actuator_name].joint_indices
                if isinstance(joint_indices, slice):
                    actuator_global_ids = torch.arange(self.asset.num_joints, device=self.device, dtype=torch.long)[joint_indices]
                elif isinstance(joint_indices, torch.Tensor):
                    actuator_global_ids = joint_indices.to(self.device).long()
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")

                per_actuator_ids = actuator_global_ids[torch.isin(actuator_global_ids, requested_global_ids)]
                if per_actuator_ids.numel() == 0:
                    continue
                cfg.params = cfg.params | {"joint_ids": per_actuator_ids.tolist(), "joint_names": None}
            self.adrs.append(IndividualArmatureADR(cfg, env))

        if len(self.adrs) == 0:
            raise ValueError("RobotArmatureADR found no actuator containing the requested joints.")

    def _resolve_requested_global_joint_ids(self, params) -> torch.Tensor | None:
        joint_ids = params.get("joint_ids", None)
        joint_names = params.get("joint_names", None)
        if joint_ids is None and joint_names is None:
            if self.asset_cfg.joint_ids != slice(None):
                joint_ids = self.asset_cfg.joint_ids
            elif self.asset_cfg.joint_names is not None:
                joint_names = self.asset_cfg.joint_names
        if joint_ids is None and joint_names is None:
            return None

        selected_ids = None
        if joint_ids is not None:
            if isinstance(joint_ids, slice):
                selected_ids = torch.arange(self.asset.num_joints, device=self.device, dtype=torch.long)[joint_ids]
            else:
                selected_ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long).view(-1)

        if joint_names is not None:
            joint_expr = joint_names if isinstance(joint_names, str) else "|".join(joint_names)
            matched_joint_ids, _ = self.asset.find_joints(joint_expr)
            matched_joint_ids = torch.as_tensor(matched_joint_ids, dtype=torch.long, device=self.device).view(-1)
            if selected_ids is None:
                selected_ids = matched_joint_ids
            else:
                selected_ids = selected_ids[torch.isin(selected_ids, matched_joint_ids)]

        if selected_ids is None or selected_ids.numel() == 0:
            raise ValueError("RobotArmatureADR received an empty joint selection from `joint_ids` / `joint_names`.")
        return selected_ids

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for adr in self.adrs:
            adr.reset(env_ids)

    def __call__(self, *args, **kwargs):
        if len(self.adrs) == 0:
            return None
        difficulties = torch.cat([adr.difficulties.float().reshape(-1) for adr in self.adrs], dim=0)
        if difficulties.numel() == 0:
            return None
        first_adr = self.adrs[0]
        first_adr._log_counter += 1
        if first_adr._log_counter % first_adr.log_every_resets != 0:
            return None
        return {
            "mean": difficulties.mean(),
            "std": difficulties.std(unbiased=False),
            "min": difficulties.min(),
            "max": difficulties.max(),
            "p50": torch.quantile(difficulties, 0.5),
            "p90": torch.quantile(difficulties, 0.9),
            "frac_at_max": (difficulties >= max(float(adr.max_difficulty) for adr in self.adrs)).float().mean(),
        }


class MaterialPropertyADR(ConsecutiveSuccessADR):
    PHYSX_MATERIAL_LIMIT = 64000

    def __init__(self, cfg, env):
        self.randomize_viscous = bool(cfg.params.get("randomize_viscous", False))
        super().__init__(cfg, env)
        self.num_buckets = int(cfg.params.get("num_buckets", 512))
        if self.num_buckets < 2:
            raise ValueError("MaterialPropertyADR requires num_buckets >= 2 for explicit min/max boundary buckets.")
        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term not supported for asset '{self.asset_cfg.name}' with type '{type(self.asset)}'."
            )
        self.num_shapes_per_body = None

        max_unique_materials = int(cfg.params.get("max_unique_materials", self.PHYSX_MATERIAL_LIMIT))
        total_candidate = (self.max_difficulty + 1) * (self.max_difficulty + 1) * self.num_buckets
        if total_candidate > max_unique_materials:
            self.num_buckets = self.PHYSX_MATERIAL_LIMIT // ((self.max_difficulty + 1) * (self.max_difficulty + 1))

        self.material_buckets = torch.zeros(
            (self.max_difficulty + 1, self.max_difficulty + 1, self.num_buckets, 3), device=self.device
        )
        for low_i in range(self.max_difficulty + 1):
            low = self.initial_low - (low_i * self.step_size)
            low = torch.clamp(low, min=self.limit_low, max=self.limit_high)
            for high_i in range(self.max_difficulty + 1):
                high = self.initial_high + (high_i * self.step_size)
                high = torch.clamp(high, min=self.limit_low, max=self.limit_high)
                samples = torch.rand((self.num_buckets, 3), device=self.device)
                samples[0] = 0.0
                samples[1] = 1.0
                self.material_buckets[low_i, high_i] = samples * (high - low) + low

        if bool(cfg.params.get("make_consistent", False)):
            self.material_buckets[..., 1] = torch.min(self.material_buckets[..., 0], self.material_buckets[..., 1])

    def _set_defaults(self):
        self.total_num_shapes = self.asset.root_physx_view.max_shapes
        all_materials = self.asset.root_physx_view.get_material_properties()[0]
        if self.randomize_viscous:
            self.defaults = all_materials.flatten().to(self.device).clone()
        else:
            self.defaults = all_materials[:, :2].flatten().to(self.device).clone()
        self.param_shape = 2 * self.total_num_shapes

    def _set_limits_and_stepsize(self):
        self.initial_low = torch.as_tensor(
            self.cfg.params.get("initial_low", torch.tensor([1.0, 1.0, 0.0], device=self.device)),
            device=self.device,
            dtype=self.defaults.dtype,
        )
        self.initial_high = torch.as_tensor(
            self.cfg.params.get("initial_high", torch.tensor([1.0, 1.0, 0.0], device=self.device)),
            device=self.device,
            dtype=self.defaults.dtype,
        )
        self.step_size = torch.as_tensor(
            self.cfg.params.get("step_size", (self.initial_low / 100.0)),
            device=self.device,
            dtype=self.defaults.dtype,
        )
        self.limit_low = torch.as_tensor(
            self.cfg.params.get("limit_low", torch.tensor([0.8, 0.6, 0.0], device=self.device)),
            device=self.device,
            dtype=self.defaults.dtype,
        )
        self.limit_high = torch.as_tensor(
            self.cfg.params.get("limit_high", torch.tensor([2.0, 1.6, 0.0], device=self.device)),
            device=self.device,
            dtype=self.defaults.dtype,
        )
        self.limits_set = True

    def sample_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        idx = torch.randint(0, self.num_buckets, (len(env_ids), self.total_num_shapes), device=self.device)

        reset_modes = self.central_adr_manager.get_reset_instructions(env_ids).to(self.device)
        is_bound_envs = torch.isin(reset_modes, self.mode_range)

        low_diff = self.difficulties[0::2].clamp(0, self.max_difficulty).long()
        high_diff = self.difficulties[1::2].clamp(0, self.max_difficulty).long()
        low_idx = low_diff.unsqueeze(0).expand(len(env_ids), -1)
        high_idx = high_diff.unsqueeze(0).expand(len(env_ids), -1)

        if is_bound_envs.any():
            param_idx = (reset_modes[is_bound_envs] - self.mode_offset).long()
            physical_dim_idx = param_idx // 2
            is_lower_bound = (param_idx % 2 == 0)
            bound_rows = torch.nonzero(is_bound_envs).squeeze(-1)
            idx[bound_rows[is_lower_bound], physical_dim_idx[is_lower_bound]] = 0
            idx[bound_rows[~is_lower_bound], physical_dim_idx[~is_lower_bound]] = 1

        sampled = self.material_buckets[low_idx, high_idx, idx]  # (n_envs, n_shapes, 3)
        return sampled

    def set_values(self, env_ids, values):
        value_device = self.asset.root_physx_view.get_material_properties().device
        env_ids_i32 = torch.as_tensor(env_ids, device=value_device, dtype=torch.int32)
        self.asset.root_physx_view.set_material_properties(values.contiguous().to(value_device), indices=env_ids_i32)


class RobotMaterialCSADR(MaterialPropertyADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert isinstance(self.asset, Articulation)
        if self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)
                self.num_shapes_per_body.append(link_view.max_shapes)
            if sum(self.num_shapes_per_body) != self.asset.root_physx_view.max_shapes:
                raise ValueError("RobotMaterialCSADR failed to parse shape counts per body.")

    def sample_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        values = super().sample_values(env_ids)
        if self.num_shapes_per_body is None:
            return values
        current_materials = self.asset.root_physx_view.get_material_properties().to(values.device)
        env_ids = torch.as_tensor(env_ids, device=values.device, dtype=torch.long)
        out = current_materials[env_ids].clone()
        for body_id in self.asset_cfg.body_ids:
            start = sum(self.num_shapes_per_body[:body_id])
            end = start + self.num_shapes_per_body[body_id]
            out[:, start:end] = values[:, start:end]
        return out


class ObjectMaterialCSADR(MaterialPropertyADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert isinstance(self.asset, RigidObject)
        initial_materials = self.asset.root_physx_view.get_material_properties()
        initial_static_friction = initial_materials[:, :, 0]
        self.randomizable_shape_mask = initial_static_friction > 1e-4

    def sample_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        values = super().sample_values(env_ids)
        env_ids = torch.as_tensor(env_ids, device=values.device, dtype=torch.long)
        mask = self.randomizable_shape_mask[env_ids].unsqueeze(-1).to(values.device)
        current = self.asset.root_physx_view.get_material_properties().to(values.device)[env_ids]
        return torch.where(mask, values, current)


class RobotMassADR(RobotBodyCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.recompute_inertia = bool(cfg.params.get("recompute_inertia", True))

    def _set_defaults(self):
        self.defaults = self.asset.data.default_mass[:, self.body_ids].clone()

    def set_values(self, env_ids, values):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        body_ids = torch.as_tensor(self.body_ids, dtype=torch.int64, device=self.device)
        masses = self.asset.root_physx_view.get_masses()
        masses[env_ids[:, None], body_ids] = values
        self.asset.root_physx_view.set_masses(masses, env_ids)

        if not self.recompute_inertia:
            return
        ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
        inertias = self.asset.root_physx_view.get_inertias()
        if isinstance(self.asset, Articulation):
            inertias[env_ids[:, None], body_ids] = self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
        else:
            inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
        self.asset.root_physx_view.set_inertias(inertias, env_ids)


class ObjectMassADR(ConsecutiveSuccessADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.recompute_inertia = bool(cfg.params.get("recompute_inertia", True))

    def _set_defaults(self):
        if self.asset_cfg.body_ids == slice(None):
            self._body_ids = torch.arange(self.asset.num_bodies, dtype=torch.long, device=self.device)
        else:
            self._body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.long, device=self.device)
        self.defaults = self.asset.data.default_mass[:, self._body_ids].clone()

    def set_values(self, env_ids, values):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        masses = self.asset.root_physx_view.get_masses()
        masses[env_ids[:, None], self._body_ids] = values
        self.asset.root_physx_view.set_masses(masses, env_ids)

        if not self.recompute_inertia:
            return
        ratios = masses[env_ids[:, None], self._body_ids] / self.asset.data.default_mass[env_ids[:, None], self._body_ids]
        inertias = self.asset.root_physx_view.get_inertias()
        if isinstance(self.asset, Articulation):
            inertias[env_ids[:, None], self._body_ids] = (
                self.asset.data.default_inertia[env_ids[:, None], self._body_ids] * ratios[..., None]
            )
        else:
            inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
        self.asset.root_physx_view.set_inertias(inertias, env_ids)


class ObsNoiseCSADR(ConsecutiveSuccessADR):
    def _set_defaults(self):
        self.obs_key = self.cfg.params.get("obs_key", None)
        if self.obs_key is None:
            raise ValueError("ObsNoiseCSADR requires `obs_key` in curriculum params.")

        obs_dim = self.cfg.params.get("obs_dim", None)
        obs_shape = self.cfg.params.get("obs_shape", None)
        self._shape_func = self.cfg.params.get("shape_func", None)
        self._shape_func_params = self.cfg.params.get("shape_func_params", {})
        if obs_dim is None and obs_shape is None and self._shape_func is None:
            raise ValueError("ObsNoiseCSADR requires `obs_dim` / `obs_shape` or `shape_func` in curriculum params.")
        if obs_dim is not None and obs_shape is not None:
            raise ValueError("ObsNoiseCSADR expects only one of `obs_dim` or `obs_shape`.")

        if obs_dim is None and obs_shape is not None:
            if isinstance(obs_shape, int):
                obs_dim = obs_shape
            else:
                obs_dim = 1
                for dim in obs_shape:
                    obs_dim *= int(dim)

        self.obs_dim = int(obs_dim) if obs_dim is not None else 0
        self._obs_buffer_attr = self.cfg.params.get("obs_buffer_attr", "obs_adr_buffers")
        if not hasattr(self._env, self._obs_buffer_attr):
            setattr(self._env, self._obs_buffer_attr, {})

        if self.obs_dim > 0:
            self.defaults = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
            buffers = getattr(self._env, self._obs_buffer_attr)
            if self.obs_key not in buffers:
                buffers[self.obs_key] = torch.zeros_like(self.defaults)
        else:
            self._defer_adr_init = True
            self.defaults = torch.zeros((self.num_envs, 1), device=self.device)

    def _resolve_feature_dim(self) -> int:
        if self.obs_dim > 0:
            return self.obs_dim
        if self._shape_func is None:
            raise ValueError("ObsNoiseCSADR could not infer observation dimension: `shape_func` missing.")
        data = self._shape_func(self._env, **self._shape_func_params)
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"ObsNoiseCSADR shape_func must return torch.Tensor. Got: {type(data)}.")
        if data.shape[0] != self.num_envs:
            raise ValueError(
                f"ObsNoiseCSADR shape_func returned leading dim {data.shape[0]}, expected num_envs={self.num_envs}."
            )
        self.obs_dim = int(data.reshape(self.num_envs, -1).shape[1])
        self.defaults = torch.zeros((self.num_envs, self.obs_dim), device=self.device, dtype=data.dtype)
        buffers = getattr(self._env, self._obs_buffer_attr)
        if self.obs_key not in buffers or buffers[self.obs_key].shape != (self.num_envs, self.obs_dim):
            buffers[self.obs_key] = torch.zeros_like(self.defaults)
        return self.obs_dim

    def _set_limits_and_stepsize(self):
        low_default = torch.zeros((self.num_envs, self.obs_dim), device=self.device, dtype=self.defaults.dtype)
        high_default = low_default.clone()
        step_default = torch.full((self.num_envs, self.obs_dim), 0.01, device=self.device, dtype=self.defaults.dtype)
        limit_low_default = torch.full((self.num_envs, self.obs_dim), -1.0, device=self.device, dtype=self.defaults.dtype)
        limit_high_default = torch.full((self.num_envs, self.obs_dim), 1.0, device=self.device, dtype=self.defaults.dtype)
        self.initial_low = self._expand_param(self.cfg.params.get("initial_low", None), low_default)
        self.initial_high = self._expand_param(self.cfg.params.get("initial_high", None), high_default)
        self.step_size = self._expand_param(self.cfg.params.get("step_size", None), step_default)
        self.limit_low = self._expand_param(self.cfg.params.get("limit_low", None), limit_low_default)
        self.limit_high = self._expand_param(self.cfg.params.get("limit_high", None), limit_high_default)
        self.limits_set = True

    def set_values(self, env_ids, values):
        buffers = getattr(self._env, self._obs_buffer_attr)
        if self.obs_key not in buffers:
            buffers[self.obs_key] = torch.zeros((self.num_envs, self.obs_dim), device=self.device, dtype=values.dtype)
        buffer = buffers[self.obs_key]
        buffer[env_ids] = values.to(buffer.device, dtype=buffer.dtype)


class ActionNoiseCSADR(ConsecutiveSuccessADR):
    def _set_defaults(self):
        self.action_key = self.cfg.params.get("action_key", None)
        if self.action_key is None:
            raise ValueError("ActionNoiseCSADR requires `action_key` in curriculum params.")

        action_dim = self.cfg.params.get("action_dim", None)
        action_shape = self.cfg.params.get("action_shape", None)
        self._shape_func = self.cfg.params.get("shape_func", None)
        self._shape_func_params = self.cfg.params.get("shape_func_params", {})
        if action_dim is None and action_shape is None and self._shape_func is None:
            raise ValueError("ActionNoiseCSADR requires `action_dim` / `action_shape` or `shape_func` in curriculum params.")
        if action_dim is not None and action_shape is not None:
            raise ValueError("ActionNoiseCSADR expects only one of `action_dim` or `action_shape`.")

        if action_dim is None and action_shape is not None:
            if isinstance(action_shape, int):
                action_dim = action_shape
            else:
                action_dim = 1
                for dim in action_shape:
                    action_dim *= int(dim)

        self.action_dim = int(action_dim) if action_dim is not None else 0
        self._action_buffer_attr = self.cfg.params.get("action_buffer_attr", "action_adr_buffers")
        if not hasattr(self._env, self._action_buffer_attr):
            setattr(self._env, self._action_buffer_attr, {})

        if self.action_dim > 0:
            self.defaults = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            buffers = getattr(self._env, self._action_buffer_attr)
            if self.action_key not in buffers:
                buffers[self.action_key] = torch.zeros_like(self.defaults)
        else:
            self._defer_adr_init = True
            self.defaults = torch.zeros((self.num_envs, 1), device=self.device)

    def _resolve_feature_dim(self) -> int:
        if self.action_dim > 0:
            return self.action_dim
        if self._shape_func is None:
            raise ValueError("ActionNoiseCSADR could not infer action dimension: `shape_func` missing.")
        data = self._shape_func(self._env, **self._shape_func_params)
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"ActionNoiseCSADR shape_func must return torch.Tensor. Got: {type(data)}.")
        if data.shape[0] != self.num_envs:
            raise ValueError(
                f"ActionNoiseCSADR shape_func returned leading dim {data.shape[0]}, expected num_envs={self.num_envs}."
            )
        self.action_dim = int(data.reshape(self.num_envs, -1).shape[1])
        self.defaults = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=data.dtype)
        buffers = getattr(self._env, self._action_buffer_attr)
        if self.action_key not in buffers or buffers[self.action_key].shape != (self.num_envs, self.action_dim):
            buffers[self.action_key] = torch.zeros_like(self.defaults)
        return self.action_dim

    def _set_limits_and_stepsize(self):
        low_default = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=self.defaults.dtype)
        high_default = low_default.clone()
        step_default = torch.full((self.num_envs, self.action_dim), 0.01, device=self.device, dtype=self.defaults.dtype)
        limit_low_default = torch.full((self.num_envs, self.action_dim), -1.0, device=self.device, dtype=self.defaults.dtype)
        limit_high_default = torch.full((self.num_envs, self.action_dim), 1.0, device=self.device, dtype=self.defaults.dtype)
        self.initial_low = self._expand_param(self.cfg.params.get("initial_low", None), low_default)
        self.initial_high = self._expand_param(self.cfg.params.get("initial_high", None), high_default)
        self.step_size = self._expand_param(self.cfg.params.get("step_size", None), step_default)
        self.limit_low = self._expand_param(self.cfg.params.get("limit_low", None), limit_low_default)
        self.limit_high = self._expand_param(self.cfg.params.get("limit_high", None), limit_high_default)
        self.limits_set = True

    def set_values(self, env_ids, values):
        buffers = getattr(self._env, self._action_buffer_attr)
        if self.action_key not in buffers:
            buffers[self.action_key] = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=values.dtype)
        buffer = buffers[self.action_key]
        buffer[env_ids] = values.to(buffer.device, dtype=buffer.dtype)


class ObjectScaleAndPosADR(ManagerTermBase):
    """Compatibility applier for object pool pose updates (no ADR difficulty ownership)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg")
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.inactive_height = float(cfg.params.get("inactive_height", -100.0))
        self._object_pool_state = cfg.params.get("object_pool_state", None)
        if not isinstance(self._object_pool_state, dict) or "names" not in self._object_pool_state:
            raise ValueError("ObjectScaleAndPosADR requires `object_pool_state` with `names`.")
        self.asset_index = int(self._object_pool_state["names"].index(self.asset_cfg.name))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids_t) == 0:
            return

        active_indices = getattr(
            self._env,
            "active_asset_indices",
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
        )
        target_pose = getattr(self._env, "_object_pool_target_pose", None)
        if target_pose is None:
            return

        matches = active_indices[env_ids_t] == self.asset_index
        match_ids = env_ids_t[matches]
        not_match_ids = env_ids_t[~matches]

        if len(not_match_ids) > 0 and (self.asset.data.root_pos_w[not_match_ids, 2] > -50.0).any():
            away_pose = torch.cat(
                (self.asset.data.root_pos_w[not_match_ids].clone(), self.asset.data.root_quat_w[not_match_ids].clone()),
                dim=-1,
            )
            away_pose[:, 2] = self.inactive_height
            self.asset.write_root_pose_to_sim(away_pose, not_match_ids)

        if len(match_ids) > 0:
            self.asset.write_root_pose_to_sim(target_pose[env_ids_t][matches], match_ids)


class ObjectPoolScaleAndPosADR(ConsecutiveSuccessADR):
    """Pool-aware object pose + size ADR with per-dimension low/high difficulty.

    Expected asset naming:
    - ``object`` for difficulty 0
    - ``obj_<m|p><abs_diff>_<idx>`` for nonzero difficulty
    Example: ``obj_m5_0`` means size difficulty 5 on the "smaller" side.

    ADR feature dimensions are ordered as: ``[x, y, z, size]`` and each feature has
    separate low/high difficulty (8 total ADR modes). Size is sampled from existing
    signed levels inside the current interval ``[min_diff, max_diff]``.
    """

    _OBJECT_NAME_PATTERN = re.compile(r"^obj_([mp])(\d+)_(\d+)$")

    def __init__(self, cfg, env):
        self.asset_cfgs = cfg.params.get("asset_cfgs", None)
        if not self.asset_cfgs:
            raise ValueError(f"{self.__class__.__name__} requires non-empty `asset_cfgs`.")
        self.asset_names = [asset_cfg.name for asset_cfg in self.asset_cfgs]

        parsed_levels = []
        for name in self.asset_names:
            if name == "object":
                parsed_levels.append(0)
            else:
                match = self._OBJECT_NAME_PATTERN.match(name)
                if match is None:
                    raise ValueError(
                        f"{self.__class__.__name__} expected asset name 'object' or pattern "
                        f"'obj_<m|p><abs_diff>_<idx>'. Got: '{name}'."
                    )
                sign_token = match.group(1)
                abs_diff = int(match.group(2))
                signed_diff = -abs_diff if sign_token == "m" else abs_diff
                parsed_levels.append(signed_diff)
        self._asset_levels_from_name = parsed_levels
        level_set = set(parsed_levels)
        abs_levels = sorted({abs(level) for level in level_set if level != 0})
        missing_pairs = [level for level in abs_levels if (-level not in level_set or level not in level_set)]
        if len(missing_pairs) > 0:
            raise ValueError(
                f"{self.__class__.__name__} requires both 'small' (-k) and 'large' (+k) assets "
                f"for each absolute difficulty. Missing pairs for: {missing_pairs}."
            )

        self._min_pool_level = int(min(parsed_levels))
        self._max_pool_level = int(max(parsed_levels))

        params = dict(cfg.params)
        if params.get("asset_cfg", None) is None:
            params["asset_cfg"] = self.asset_cfgs[0]
        cfg.params = params

        self.inactive_height = float(cfg.params.get("inactive_height", -100.0))
        self.randomize_quat = bool(cfg.params.get("randomize_quat", False))
        self.randomize_z = bool(cfg.params.get("randomize_z", False))

        self.assets: list[RigidObject] = []
        self.pool_levels = torch.zeros(0, dtype=torch.long, device=env.device)
        self.level_to_asset_indices: dict[int, torch.Tensor] = {}
        self._available_levels = torch.zeros(0, dtype=torch.long, device=env.device)

        super().__init__(cfg, env)

        if not hasattr(self._env, "active_asset_indices"):
            self._env.active_asset_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if not hasattr(self._env, "_object_pool_target_pose"):
            self._env._object_pool_target_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def _set_defaults(self):
        self.assets = [self._env.scene[name] for name in self.asset_names]
        self.pool_levels = torch.tensor(self._asset_levels_from_name, dtype=torch.long, device=self.device)

        self.level_to_asset_indices = {}
        for level_t in torch.unique(self.pool_levels):
            level = int(level_t.item())
            indices = torch.nonzero(self.pool_levels == level_t).squeeze(-1)
            self.level_to_asset_indices[level] = indices
        self._available_levels = torch.tensor(sorted(self.level_to_asset_indices.keys()), dtype=torch.long, device=self.device)

        # Compatibility attributes used by existing wrappers/debug flows.
        self._env._object_scale_pool_names = list(self.asset_names)
        self._env._object_scale_pool_difficulties = self.pool_levels
        xform_scales = []
        for name in self.asset_names:
            view = XformPrimView(self._env.scene[name].cfg.prim_path, device=self.device, validate_xform_ops=False)
            xform_scales.append(view.get_scales()[0].to(self.device))
        self._env._object_scale_pool_scales = torch.stack(xform_scales, dim=0)

        default_pos = self.assets[0].data.root_pos_w.clone()
        default_size = torch.zeros((self.num_envs, 1), device=self.device, dtype=default_pos.dtype)
        self.defaults = torch.cat((default_pos, default_size), dim=-1)

    def _set_limits_and_stepsize(self):
        step_default = torch.zeros_like(self.defaults)
        step_default[:, 0] = 0.01
        step_default[:, 1] = 0.01
        step_default[:, 2] = 0.01 if self.randomize_z else 0.0
        step_default[:, 3] = 1.0

        limit_low_default = self.defaults.clone()
        limit_high_default = self.defaults.clone()
        limit_low_default[:, 0] = -1.0
        limit_low_default[:, 1] = -1.0
        limit_low_default[:, 2] = 0.0
        limit_high_default[:, 0] = 1.0
        limit_high_default[:, 1] = 1.0
        limit_high_default[:, 2] = 1.0
        limit_low_default[:, 3] = float(self._min_pool_level)
        limit_high_default[:, 3] = float(self._max_pool_level)

        initial_low_default = self.defaults.clone()
        initial_high_default = self.defaults.clone()

        self.initial_low = self._expand_param(self.cfg.params.get("initial_low", None), initial_low_default)
        self.initial_high = self._expand_param(self.cfg.params.get("initial_high", None), initial_high_default)
        self.step_size = self._expand_param(self.cfg.params.get("step_size", None), step_default)
        self.limit_low = self._expand_param(self.cfg.params.get("limit_low", None), limit_low_default)
        self.limit_high = self._expand_param(self.cfg.params.get("limit_high", None), limit_high_default)

        # Keep z fixed when requested, regardless of custom step config.
        if not self.randomize_z:
            self.step_size[:, 2] = 0.0

        self.limits_set = True

    def _sample_size_levels(
        self,
        env_ids: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        reset_modes: torch.Tensor,
    ) -> torch.Tensor:
        size_low = torch.ceil(low[:, 3]).long()
        size_high = torch.floor(high[:, 3]).long()
        size_low = torch.clamp(size_low, min=self._min_pool_level, max=self._max_pool_level)
        size_high = torch.clamp(size_high, min=self._min_pool_level, max=self._max_pool_level)

        invalid = size_low > size_high
        if invalid.any():
            fallback = torch.clamp(
                torch.round((low[:, 3] + high[:, 3]) * 0.5).long(),
                min=self._min_pool_level,
                max=self._max_pool_level,
            )
            size_low[invalid] = fallback[invalid]
            size_high[invalid] = fallback[invalid]

        levels = self._available_levels.unsqueeze(0)
        candidate_mask = (levels >= size_low.unsqueeze(1)) & (levels <= size_high.unsqueeze(1))
        has_candidates = candidate_mask.any(dim=1)
        sampled = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)

        if has_candidates.any():
            sampled[has_candidates] = self._sample_levels_from_mask(candidate_mask[has_candidates])
        if (~has_candidates).any():
            sampled[~has_candidates] = self._sample_nearest_levels(
                size_low[~has_candidates], size_high[~has_candidates]
            )

        is_bound_envs = torch.isin(reset_modes, self.mode_range)
        if is_bound_envs.any():
            bound_rows = torch.nonzero(is_bound_envs).squeeze(-1)
            param_idx = (reset_modes[bound_rows] - self.mode_offset).long()
            physical_dim_idx = param_idx // 2
            is_lower_bound = (param_idx % 2) == 0
            size_rows_mask = physical_dim_idx == 3 #check if the the mode is scale
            if size_rows_mask.any():
                size_rows = bound_rows[size_rows_mask]
                size_is_lower = is_lower_bound[size_rows_mask]
                size_low_rows = size_low[size_rows]
                size_high_rows = size_high[size_rows]
                row_levels = self._available_levels.unsqueeze(0)
                row_candidate_mask = (row_levels >= size_low_rows.unsqueeze(1)) & (row_levels <= size_high_rows.unsqueeze(1))
                row_has_candidates = row_candidate_mask.any(dim=1)

                if row_has_candidates.any():
                    valid_mask = row_candidate_mask[row_has_candidates]
                    level_grid = self._available_levels.unsqueeze(0).expand(valid_mask.shape[0], -1)
                    i64max = torch.iinfo(torch.long).max
                    i64min = torch.iinfo(torch.long).min
                    lower_choices = torch.where(valid_mask, level_grid, torch.full_like(level_grid, i64max)).min(dim=1).values
                    upper_choices = torch.where(valid_mask, level_grid, torch.full_like(level_grid, i64min)).max(dim=1).values
                    valid_rows = size_rows[row_has_candidates]
                    valid_is_lower = size_is_lower[row_has_candidates]
                    if valid_is_lower.any():
                        sampled[valid_rows[valid_is_lower]] = lower_choices[valid_is_lower]
                    if (~valid_is_lower).any():
                        sampled[valid_rows[~valid_is_lower]] = upper_choices[~valid_is_lower]

                if (~row_has_candidates).any():
                    fallback_rows = size_rows[~row_has_candidates]
                    fallback_levels = self._sample_nearest_levels(size_low[fallback_rows], size_high[fallback_rows])
                    sampled[fallback_rows] = fallback_levels

        return sampled

    def _sample_levels_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Uniformly sample one available level per row from boolean mask."""
        # mask: (rows, n_levels)
        counts = mask.sum(dim=1)
        if (counts <= 0).any():
            raise ValueError(f"{self.__class__.__name__} got empty candidate rows in _sample_levels_from_mask.")
        rand_rank = torch.floor(torch.rand(mask.shape[0], device=self.device) * counts.to(torch.float32)).long()
        ranks = torch.cumsum(mask.to(torch.long), dim=1) - 1
        chosen_mask = mask & (ranks == rand_rank.unsqueeze(1))
        level_grid = self._available_levels.unsqueeze(0).expand(mask.shape[0], -1)
        return (chosen_mask.to(torch.long) * level_grid).sum(dim=1).long()

    def _sample_nearest_levels(self, size_low: torch.Tensor, size_high: torch.Tensor) -> torch.Tensor:
        """Sample nearest existing levels to interval midpoint (random tie-break)."""
        mid = (size_low.to(torch.float32) + size_high.to(torch.float32)) * 0.5
        level_grid = self._available_levels.to(torch.float32).unsqueeze(0).expand(len(mid), -1)
        diff = torch.abs(level_grid - mid.unsqueeze(1))
        min_diff = diff.min(dim=1).values
        nearest_mask = diff == min_diff.unsqueeze(1)
        return self._sample_levels_from_mask(nearest_mask)

    def _snap_levels_to_available(self, levels: torch.Tensor) -> torch.Tensor:
        levels = levels.clone()
        is_valid = torch.isin(levels, self._available_levels)
        if is_valid.all():
            return levels
        missing_rows = torch.nonzero(~is_valid).squeeze(-1)
        levels[missing_rows] = self._sample_nearest_levels(levels[missing_rows], levels[missing_rows])
        return levels

    def _sample_asset_indices_for_levels(self, sampled_levels: torch.Tensor) -> torch.Tensor:
        selected_indices = torch.zeros_like(sampled_levels, dtype=torch.long, device=self.device)
        unique_levels = torch.unique(sampled_levels)
        for level_t in unique_levels:
            level = int(level_t.item())
            candidates = self.level_to_asset_indices[level]
            level_rows = torch.nonzero(sampled_levels == level_t).squeeze(-1)
            choose_ids = torch.randint(0, candidates.numel(), (len(level_rows),), device=self.device)
            selected_indices[level_rows] = candidates[choose_ids]
        return selected_indices

    def sample_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return torch.zeros((0, 4), device=self.device, dtype=self.defaults.dtype)

        reset_modes = self.central_adr_manager.get_reset_instructions(env_ids).to(self.device)

        low_diff = self.difficulties[0::2]
        high_diff = self.difficulties[1::2]
        low = self.initial_low[env_ids] - (low_diff[None] * self.step_size[env_ids])
        low = torch.clamp(low, min=self.limit_low[env_ids], max=self.limit_high[env_ids])
        high = self.initial_high[env_ids] + (high_diff[None] * self.step_size[env_ids])
        high = torch.clamp(high, min=self.limit_low[env_ids], max=self.limit_high[env_ids])

        values = torch.rand_like(high) * (high - low) + low

        # Enforce exact ADR boundary values for xyz dimensions.
        is_bound_envs = torch.isin(reset_modes, self.mode_range)
        if is_bound_envs.any():
            bound_rows = torch.nonzero(is_bound_envs).squeeze(-1)
            param_idx = (reset_modes[bound_rows] - self.mode_offset).long()
            physical_dim_idx = param_idx // 2
            is_lower_bound = (param_idx % 2) == 0
            is_pos_dim = physical_dim_idx < 3
            if is_pos_dim.any():
                pos_rows = bound_rows[is_pos_dim]
                pos_dims = physical_dim_idx[is_pos_dim]
                pos_is_lower = is_lower_bound[is_pos_dim]
                low_rows = pos_rows[pos_is_lower]
                high_rows = pos_rows[~pos_is_lower]
                low_dims = pos_dims[pos_is_lower]
                high_dims = pos_dims[~pos_is_lower]
                if len(low_rows) > 0:
                    values[low_rows, low_dims] = low[low_rows, low_dims]
                if len(high_rows) > 0:
                    values[high_rows, high_dims] = high[high_rows, high_dims]

        sampled_levels = self._sample_size_levels(env_ids=env_ids, low=low, high=high, reset_modes=reset_modes)
        values[:, 3] = sampled_levels.to(values.dtype)
        return values

    def set_values(self, env_ids, values):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        sampled_levels = torch.round(values[:, 3]).long()
        sampled_levels = torch.clamp(sampled_levels, min=self._min_pool_level, max=self._max_pool_level)
        sampled_levels = self._snap_levels_to_available(sampled_levels)
        selected_asset_indices = self._sample_asset_indices_for_levels(sampled_levels)
        self._env.active_asset_indices[env_ids] = selected_asset_indices

        if self.randomize_quat:
            quat = math_utils.random_orientation(len(env_ids), self.device)
        else:
            quat = self.assets[0].data.root_quat_w[env_ids].clone()
        target_pose = torch.cat((values[:, :3], quat), dim=-1)
        self._env._object_pool_target_pose[env_ids] = target_pose

        for asset_idx, asset in enumerate(self.assets):
            matches = selected_asset_indices == asset_idx
            match_ids = env_ids[matches]
            not_match_ids = env_ids[~matches]

            if len(not_match_ids) > 0 and (asset.data.root_pos_w[not_match_ids, 2] > -50.0).any():
                away_pose = torch.cat(
                    (asset.data.root_pos_w[not_match_ids].clone(), asset.data.root_quat_w[not_match_ids].clone()),
                    dim=-1,
                )
                away_pose[:, 2] = self.inactive_height
                asset.write_root_pose_to_sim(away_pose, not_match_ids)

            if len(match_ids) > 0:
                asset.write_root_pose_to_sim(target_pose[matches], match_ids)


# class AllObjecCSADR(ManagerTermBase):
#     """
#     Centralized object-pool manager:
#     - owns exactly one ADRDifficultyCore with 8 dims (2 * [pos_xyz + shared_scale])
#     - selects active object variant per env
#     - samples object pose and applies it directly to all pool objects
#     """
#
#     def __init__(self, cfg, env):
#         super().__init__(cfg, env)
#         self.asset_cfgs = cfg.params.get("asset_cfgs", [SceneEntityCfg(name=f"object_{i}") for i in range(10)])
#         self.asset_names = [asset_cfg.name for asset_cfg in self.asset_cfgs]
#         self.assets: list[RigidObject] = [env.scene[name] for name in self.asset_names]
#         env.active_asset_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
#         env._object_pool_target_pose = torch.zeros((env.num_envs, 7), device=env.device)
#
#         self.central_adr_manager: CentralADRManager = env.central_adr_manager
#         self.max_difficulty = int(cfg.params.get("max_difficulty", 5))
#         self.upgrade_threshold = float(cfg.params.get("upgrade_threshold", 0.7))
#         self.downgrade_threshold = float(cfg.params.get("downgrade_threshold", 0.1))
#         self.eval_interval = int(cfg.params.get("eval_interval", 256))
#         self.inactive_height = float(cfg.params.get("inactive_height", -100.0))
#         self.randomize_quat = bool(cfg.params.get("randomize_quat", False))
#         self.randomize_z = bool(cfg.params.get("randomize_z", False))
#         self.log_every_resets = max(1, int(cfg.params.get("log_every_resets", 1)))
#         self.log_watch_indices = [int(i) for i in cfg.params.get("log_watch_indices", [])]
#         self._log_counter = 0
#
#         success_path = cfg.params.get(
#             "success_key",
#             "command_manager._terms.object_pose.metrics.consecutive_success",
#         )
#         self.success_getter = build_getter(self._env, success_path)
#
#         self._init_pool_metadata()
#         self._pos_param_count = 6
#         self._scale_low_param_idx = 6
#         self._scale_high_param_idx = 7
#         self.adr_core = ADRDifficultyCore(
#             device=self.device,
#             num_envs=self.num_envs,
#             central_adr_manager=self.central_adr_manager,
#             param_shape=8,
#             eval_interval=self.eval_interval,
#             max_difficulty=self.max_difficulty,
#             upgrade_threshold=self.upgrade_threshold,
#             downgrade_threshold=self.downgrade_threshold,
#         )
#         self._prev_active_asset_indices = env.active_asset_indices.clone()
#
#         self._init_pose_sampling_params(cfg)
#
#     def _parse_asset_difficulty(self, asset_name: str) -> int:
#         parts = asset_name.split("_")
#         if len(parts) >= 2 and parts[1].isdigit():
#             return int(parts[1])
#         return 0
#
#     def _init_pool_metadata(self):
#         names = self.asset_names
#         diffs = torch.tensor([self._parse_asset_difficulty(name) for name in names], dtype=torch.long, device=self.device)
#         scales = []
#         for name in names:
#             view = XformPrimView(self._env.scene[name].cfg.prim_path, device=self.device, validate_xform_ops=False)
#             scales.append(view.get_scales()[0].to(self.device))
#         scales_tensor = torch.stack(scales, dim=0)
#
#         self.pool_names = names
#         self.pool_difficulties = diffs
#         self.pool_scales = scales_tensor
#         self._object_pool_state = {
#             "names": self.pool_names,
#             "difficulties": self.pool_difficulties,
#             "scales": self.pool_scales,
#         }
#
#         # Compatibility with existing wrappers/helpers.
#         self._env._object_scale_pool_names = self.pool_names
#         self._env._object_scale_pool_difficulties = self.pool_difficulties
#         self._env._object_scale_pool_scales = self.pool_scales
#
#     def _expand_pos_param(self, value, default_value: torch.Tensor) -> torch.Tensor:
#         if value is None:
#             tensor = default_value.clone()
#         elif isinstance(value, torch.Tensor):
#             tensor = value.to(default_value.device, dtype=default_value.dtype)
#         else:
#             tensor = torch.tensor(value, device=default_value.device, dtype=default_value.dtype)
#
#         feature_dim = int(default_value.shape[-1])
#         if tensor.ndim == 0:
#             tensor = tensor.repeat(feature_dim)
#         if tensor.ndim == 1:
#             if tensor.shape[0] == 1:
#                 tensor = tensor.repeat(feature_dim)
#             elif tensor.shape[0] != feature_dim:
#                 raise ValueError(
#                     f"AllObjecCSADR position param has wrong shape {tuple(tensor.shape)} for feature dim={feature_dim}."
#                 )
#             return tensor.unsqueeze(0).repeat(self.num_envs, 1)
#         if tensor.ndim == 2:
#             if tensor.shape[1] != feature_dim:
#                 raise ValueError(
#                     f"AllObjecCSADR position param has wrong shape {tuple(tensor.shape)} for feature dim={feature_dim}."
#                 )
#             if tensor.shape[0] == 1:
#                 return tensor.repeat(self.num_envs, 1)
#             if tensor.shape[0] != self.num_envs:
#                 raise ValueError(
#                     f"AllObjecCSADR position param has wrong env dimension {tensor.shape[0]} for num_envs={self.num_envs}."
#                 )
#             return tensor
#         raise ValueError(f"AllObjecCSADR position param rank must be <= 2. Got: {tensor.ndim}.")
#
#     def _init_pose_sampling_params(self, cfg):
#         self.pos_defaults = self.assets[0].data.root_pos_w.clone()
#         step_default = torch.full_like(self.pos_defaults, 0.01)
#         if not self.randomize_z:
#             step_default[:, 2] = 0.0
#
#         limit_low_default = self.pos_defaults.clone()
#         limit_high_default = self.pos_defaults.clone()
#         limit_low_default[:, :3] = torch.tensor((-1.0, -1.0, 0.0), device=self.device, dtype=self.pos_defaults.dtype)
#         limit_high_default[:, :3] = torch.tensor((1.0, 1.0, 1.0), device=self.device, dtype=self.pos_defaults.dtype)
#
#         self.initial_low = self._expand_pos_param(cfg.params.get("initial_low", None), self.pos_defaults)
#         self.initial_high = self._expand_pos_param(cfg.params.get("initial_high", None), self.pos_defaults)
#         self.step_size = self._expand_pos_param(cfg.params.get("step_size", None), step_default)
#         self.limit_low = self._expand_pos_param(cfg.params.get("limit_low", None), limit_low_default)
#         self.limit_high = self._expand_pos_param(cfg.params.get("limit_high", None), limit_high_default)
#
#     def _clamp_to_pool_difficulty(self, level: int) -> int:
#         min_diff = int(self.pool_difficulties.min().item())
#         max_diff = int(self.pool_difficulties.max().item())
#         return max(min(int(level), max_diff), min_diff)
#
#     def _current_scale_low_level(self) -> int:
#         return self._clamp_to_pool_difficulty(int(torch.round(self.adr_core.difficulties[self._scale_low_param_idx]).item()))
#
#     def _current_scale_high_level(self) -> int:
#         return self._clamp_to_pool_difficulty(int(torch.round(self.adr_core.difficulties[self._scale_high_param_idx]).item()))
#
#     def _sample_asset_indices_at_or_below(self, max_level: int, count: int) -> torch.Tensor:
#         max_level = self._clamp_to_pool_difficulty(max_level)
#         candidate_idx = torch.nonzero(self.pool_difficulties <= max_level).squeeze(-1)
#         if candidate_idx.numel() == 0:
#             candidate_idx = torch.nonzero(self.pool_difficulties == self.pool_difficulties.min()).squeeze(-1)
#         sample_ids = torch.randint(0, candidate_idx.numel(), (count,), device=self.device)
#         return candidate_idx[sample_ids]
#
#     def _update_difficulty_from_episode(self, env_ids: torch.Tensor) -> None:
#         env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
#         if len(env_ids) == 0:
#             return
#         if self.adr_core.first_update:
#             self.adr_core.first_update = False
#             return
#
#         success_values = torch.as_tensor(self.success_getter(), device=self.device).reshape(-1)
#         if success_values.shape[0] != self.num_envs:
#             raise ValueError(
#                 f"AllObjecCSADR expected success tensor with leading dim num_envs={self.num_envs}, "
#                 f"got shape {tuple(success_values.shape)}."
#             )
#
#         reset_modes = self.central_adr_manager.get_episode_result_instructions(env_ids).to(self.device)
#         is_bound_envs = torch.isin(reset_modes, self.adr_core.mode_range)
#         if not is_bound_envs.any():
#             return
#
#         bound_env_ids = env_ids[is_bound_envs]
#         bound_results = success_values[bound_env_ids].float()
#         param_idx = (reset_modes[is_bound_envs] - self.adr_core.mode_offset).long()
#         prev_active_indices = self._prev_active_asset_indices[bound_env_ids].clamp(0, self.pool_difficulties.numel() - 1)
#         prev_active_difficulty = self.pool_difficulties[prev_active_indices]
#         current_max_obj_difficulty = self._current_scale_high_level()
#
#         for idx in torch.unique(param_idx):
#             idx_int = int(idx.item())
#             idx_mask = param_idx == idx
#             idx_results = bound_results[idx_mask]
#
#             # Hardest-only credit assignment for all object-related ADR dimensions.
#             hardest_mask = prev_active_difficulty[idx_mask] == current_max_obj_difficulty
#             idx_results = idx_results[hardest_mask]
#
#             num_new = int(idx_results.numel())
#             if num_new == 0:
#                 continue
#
#             start = int(self.adr_core.eval_counts[idx_int].item())
#             insert_indices = (start + torch.arange(num_new, device=self.device)) % self.adr_core.eval_interval
#             self.adr_core.consecutive_success_counter[idx_int, insert_indices] = idx_results
#             self.adr_core.eval_counts[idx_int] += num_new
#
#             if int(self.adr_core.eval_counts[idx_int].item()) < self.adr_core.eval_interval:
#                 continue
#
#             avg_success = float(self.adr_core.consecutive_success_counter[idx_int].mean().item())
#             if avg_success >= self.adr_core.upgrade_threshold:
#                 self.adr_core.difficulties[idx_int] = torch.clamp(
#                     self.adr_core.difficulties[idx_int] + 1.0, 0.0, float(self.adr_core.max_difficulty)
#                 )
#                 self.adr_core.consecutive_success_counter[idx_int] = 0.0
#                 self.adr_core.eval_counts[idx_int] = 0
#             elif avg_success <= self.adr_core.downgrade_threshold:
#                 self.adr_core.difficulties[idx_int] = torch.clamp(
#                     self.adr_core.difficulties[idx_int] - 1.0, 0.0, float(self.adr_core.max_difficulty)
#                 )
#                 self.adr_core.consecutive_success_counter[idx_int] = 0.0
#                 self.adr_core.eval_counts[idx_int] = 0
#
#     def _select_active_assets(self, env_ids: torch.Tensor):
#         env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
#         if len(env_ids) == 0:
#             return
#
#         # Non-boundary policy: sample uniformly from all objects with difficulty <= current scale-high difficulty.
#         selected_indices = self._sample_asset_indices_at_or_below(self._current_scale_high_level(), len(env_ids))
#         reset_modes = self.central_adr_manager.get_reset_instructions(env_ids).to(self.device)
#         is_object_bound = torch.isin(reset_modes, self.adr_core.mode_range)
#
#         if is_object_bound.any():
#             bound_rows = torch.nonzero(is_object_bound).squeeze(-1)
#             bound_param_idx = (reset_modes[bound_rows] - self.adr_core.mode_offset).long()
#             physical_dim_idx = bound_param_idx // 2
#             is_lower_bound = (bound_param_idx % 2) == 0
#             is_scale_dim = physical_dim_idx >= 3
#
#             if is_scale_dim.any():
#                 scale_rows = bound_rows[is_scale_dim]
#                 scale_is_lower = is_lower_bound[is_scale_dim]
#                 lower_rows = scale_rows[scale_is_lower]
#                 upper_rows = scale_rows[~scale_is_lower]
#                 if lower_rows.numel() > 0:
#                     chosen = self._sample_asset_indices_at_or_below(
#                         self._current_scale_low_level(), int(lower_rows.numel())
#                     )
#                     selected_indices[lower_rows] = chosen
#                 if upper_rows.numel() > 0:
#                     chosen = self._sample_asset_indices_at_or_below(
#                         self._current_scale_high_level(), int(upper_rows.numel())
#                     )
#                     selected_indices[upper_rows] = chosen
#
#         self._env.active_asset_indices[env_ids] = selected_indices
#
#     def _sample_target_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
#         env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
#         if len(env_ids) == 0:
#             return torch.zeros((0, 7), device=self.device, dtype=self.pos_defaults.dtype)
#
#         low_diff = self.adr_core.difficulties[0::2][:3]
#         high_diff = self.adr_core.difficulties[1::2][:3]
#         low = self.initial_low[env_ids] - (low_diff[None] * self.step_size[env_ids])
#         low = torch.clamp(low, min=self.limit_low[env_ids], max=self.limit_high[env_ids])
#         high = self.initial_high[env_ids] + (high_diff[None] * self.step_size[env_ids])
#         high = torch.clamp(high, min=self.limit_low[env_ids], max=self.limit_high[env_ids])
#         pos = torch.rand_like(high) * (high - low) + low
#
#         reset_modes = self.central_adr_manager.get_reset_instructions(env_ids).to(self.device)
#         is_bound_envs = torch.isin(reset_modes, self.adr_core.mode_range)
#         if is_bound_envs.any():
#             param_idx = (reset_modes[is_bound_envs] - self.adr_core.mode_offset).long()
#             physical_dim_idx = param_idx // 2
#             is_lower_bound = (param_idx % 2 == 0)
#             is_pos_dim = physical_dim_idx < 3
#             if is_pos_dim.any():
#                 bound_rows = torch.nonzero(is_bound_envs).squeeze(-1)
#                 pos_rows = bound_rows[is_pos_dim]
#                 pos_dims = physical_dim_idx[is_pos_dim]
#                 pos_lower = is_lower_bound[is_pos_dim]
#                 low_rows = pos_rows[pos_lower]
#                 high_rows = pos_rows[~pos_lower]
#                 low_dims = pos_dims[pos_lower]
#                 high_dims = pos_dims[~pos_lower]
#                 if len(low_rows) > 0:
#                     pos[low_rows, low_dims] = low[low_rows, low_dims]
#                 if len(high_rows) > 0:
#                     pos[high_rows, high_dims] = high[high_rows, high_dims]
#
#         if self.randomize_quat:
#             quat = math_utils.random_orientation(len(env_ids), self.device)
#         else:
#             quat = self.assets[0].data.root_quat_w[env_ids].clone()
#         return torch.cat((pos, quat), dim=-1)
#
#     def _apply_target_pose(self, env_ids: torch.Tensor, target_pose: torch.Tensor) -> None:
#         env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
#         selected_indices = self._env.active_asset_indices[env_ids]
#         for asset_idx, asset in enumerate(self.assets):
#             matches = selected_indices == asset_idx
#             match_ids = env_ids[matches]
#             not_match_ids = env_ids[~matches]
#
#             if len(not_match_ids) > 0 and (asset.data.root_pos_w[not_match_ids, 2] > -50.0).any():
#                 away_pose = torch.cat(
#                     (asset.data.root_pos_w[not_match_ids].clone(), asset.data.root_quat_w[not_match_ids].clone()),
#                     dim=-1,
#                 )
#                 away_pose[:, 2] = self.inactive_height
#                 asset.write_root_pose_to_sim(away_pose, not_match_ids)
#
#             if len(match_ids) > 0:
#                 asset.write_root_pose_to_sim(target_pose[matches], match_ids)
#
#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         if env_ids is None:
#             env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
#         else:
#             env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
#
#         # Snapshot previous-episode active object difficulties for hardest-only credit assignment.
#         self._prev_active_asset_indices[env_ids_t] = self._env.active_asset_indices[env_ids_t].clone()
#         self._update_difficulty_from_episode(env_ids_t)
#         self._select_active_assets(env_ids_t)
#         target_pose = self._sample_target_pose(env_ids_t)
#         self._env._object_pool_target_pose[env_ids_t] = target_pose
#         self._apply_target_pose(env_ids_t, target_pose)
#
#     def __call__(self, *args, **kwargs):
#         self._log_counter += 1
#         if self._log_counter % self.log_every_resets != 0:
#             return None
#         out = self.adr_core.summarize(self.max_difficulty)
#         for idx in self.log_watch_indices:
#             if 0 <= idx < self.adr_core.difficulties.numel():
#                 out[f"watch_{idx}"] = self.adr_core.difficulties[idx]
#         self._env.max_difficulty = self.adr_core.difficulties.max()
#         self._env.mean_difficulty = self.adr_core.difficulties.mean()
#         return out


# Backward-compatible alias (older code may use `AllObjectCSADR` spelling).
# AllObjectCSADR = AllObjecCSADR


def active_pool_wrapper(mdp_func, asset_names: list[str], **kwargs):
    def wrapped_obs_func(env: ManagerBasedRLEnv) -> torch.Tensor:
        all_results = []
        active_mask = torch.zeros((env.num_envs, len(asset_names)), dtype=torch.bool, device=env.device)
        for name in asset_names:
            temp_cfg = SceneEntityCfg(name)
            data = mdp_func(env, asset_cfg=temp_cfg, **kwargs)
            all_results.append(data)
        for idx, name in enumerate(asset_names):
            # Pool policy parks inactive objects underground. Verify one active object per env.
            active_mask[:, idx] = env.scene[name].data.root_pos_w[:, 2] > -50.0

        active_count = active_mask.sum(dim=1)
        assert bool((active_count == 1).all().item()), (
            "active_pool_wrapper expected exactly one active object per env, "
            f"but got counts in [{int(active_count.min().item())}, {int(active_count.max().item())}]."
        )

        stacked_data = torch.stack(all_results, dim=1)
        active_idx = getattr(env, "active_asset_indices", None)
        if active_idx is None:
            active_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        batch_idx = torch.arange(env.num_envs, device=env.device)

        return stacked_data[batch_idx, active_idx]

    return wrapped_obs_func


def obs_adr_wrapper(mdp_func, obs_key: str, operation: str = "add", obs_buffer_attr: str = "obs_adr_buffers", **kwargs):
    if operation not in {"add", "scale", "abs"}:
        raise ValueError(f"Unsupported obs_adr operation '{operation}'. Use one of: add, scale, abs.")

    def wrapped_obs_func(env: ManagerBasedRLEnv) -> torch.Tensor:
        data = mdp_func(env, **kwargs)
        buffer_dict = getattr(env, obs_buffer_attr, None)
        if not isinstance(buffer_dict, dict):
            return data

        noise = buffer_dict.get(obs_key, None)
        if noise is None:
            return data

        data_flat = data.reshape(env.num_envs, -1)
        noise = noise.to(data.device, dtype=data.dtype)
        noise_flat = noise.reshape(env.num_envs, -1)

        if noise_flat.shape[1] == 1 and data_flat.shape[1] != 1:
            noise_flat = noise_flat.repeat(1, data_flat.shape[1])
        if noise_flat.shape != data_flat.shape:
            raise ValueError(
                f"Obs ADR shape mismatch for key '{obs_key}': data {tuple(data_flat.shape)} vs noise {tuple(noise_flat.shape)}."
            )

        if operation == "add":
            data_flat = data_flat + noise_flat
        elif operation == "scale":
            data_flat = data_flat * noise_flat
        else:
            data_flat = noise_flat

        return data_flat.reshape_as(data)

    return wrapped_obs_func
