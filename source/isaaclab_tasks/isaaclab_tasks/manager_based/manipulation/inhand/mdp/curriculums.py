from __future__ import annotations
import re
from turtledemo.forest import randomize

from isaaclab.envs.mdp.curriculums import modify_env_param
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.scene import InteractiveScene
import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase
from isaaclab.assets import Articulation, RigidObject

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils


class ConsecutiveSuccessADR(ManagerTermBase):
    """
    Adaptive Domain Randomization (ADR) for joint stiffness.

    This term monitors the consecutive successes of the agents. When the maximum
    consecutive success count across the environment exceeds a specified threshold,
    it widens the uniform distribution range [min, max] from which joint stiffness
    is sampled during reset.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg = cfg.params.get('asset_cfg')
        self.asset: Articulation | RigidObject = env.scene[self.asset_cfg.name]
        

        self.defaults: torch.Tensor
        self._set_defaults()

        self.initial_low = cfg.params.get("initial_low", self.defaults[0])
        self.initial_low = self.initial_low.to(self.defaults.device)

        self.initial_high = cfg.params.get("initial_high", self.defaults[0])
        self.initial_high = self.initial_high.to(self.defaults.device)

        self.step_size = cfg.params.get("step_size", (1 / 100 * self.defaults[0]))  # Amount to widen range by
        self.step_size = self.step_size.to(self.defaults.device)

        self.limit_low = cfg.params.get("limit_low", torch.full_like(self.defaults[0], 0))
        self.limit_low = self.limit_low.to(self.defaults.device)

        self.limit_high = cfg.params.get("limit_high", self.defaults[0] * 2)  # Hard max limit
        self.limit_high = self.limit_high.to(self.defaults.device)

        self.threshold = cfg.params.get("success_threshold", 20)
        self.count_success = cfg.params.get('count_success',
                                            False)  # counts success or use current value as consecutive success

        success_path = cfg.params.get('success_key',
                                      'command_manager._terms.object_pose.metrics.consecutive_success')  # key of success, if None, will try find itself
        self.success_getter = self.process_getter(self._env, success_path)

        self.max_difficulty = cfg.params.get('max_difficulty', 10)
        self.global_success = cfg.params.get('global_success', True)
        unrandomized_ratio = cfg.params.get('unrandomized_ratio', 0.4)
        self.max_unrandomized_idx = int(self._env.num_envs * unrandomized_ratio)

        # Current range state
        self.difficulties = torch.zeros(self._env.num_envs, dtype=torch.int, device=self.defaults.device)

        # Track consecutive successes globally (or per env depending on ADR strategy)
        # Here we track the max consecutive success achieved in the current difficulty block
        self.consecutive_success_counter = torch.zeros(self._env.num_envs if not self.global_success else 1,
                                                       device=self.defaults.device)
        self.values = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = env_ids.to(self.defaults.device)
        level_up_env_ids = self._manage_difficulty(env_ids=env_ids)

        if len(level_up_env_ids) == 0:
            return

        self.values = self._get_new_values(level_up_env_ids)

    def _manage_difficulty(self, env_ids):
        env_ids = env_ids[env_ids > self.max_unrandomized_idx].to(self.defaults.device)

        if len(env_ids) == 0:
            return []

        if self.global_success:
            if self.count_success:
                self.consecutive_success_counter = self.consecutive_success_counter + self.success_getter() if self.success_getter() else 0
            else:
                self.consecutive_success_counter = self.success_getter()
            if max(self.consecutive_success_counter[env_ids]) >= self.threshold:
                self.difficulties[env_ids] = torch.full_like(self.difficulties[env_ids], self.difficulties.max() + 1)
                self.difficulties = self.difficulties.clamp(0, self.max_difficulty)
                return env_ids
            return []
        else:
            if self.count_success:
                current_successes = self.success_getter()[env_ids] > 0

                # Find indices in the subset that failed
                failed_mask = ~current_successes
                failed_env_ids = env_ids[failed_mask]
                if len(failed_env_ids) > 0:
                    self.consecutive_success_counter[failed_env_ids] = 0

                # B. Handle Successes (Increment counter)
                succeeded_mask = current_successes
                succeeded_env_ids = env_ids[succeeded_mask]
                if len(succeeded_env_ids) > 0:
                    self.consecutive_success_counter[succeeded_env_ids] += 1
            else:
                self.consecutive_success_counter[env_ids] = self.success_getter()[env_ids]

            # Check who crossed the threshold
            level_up_mask = self.consecutive_success_counter[env_ids] > self.threshold
            level_up_env_ids = env_ids[level_up_mask]

            if len(level_up_env_ids) > 0:
                # Increment difficulty level, clamped to max_level
                new_levels = self.difficulties[level_up_env_ids] + 1
                new_levels = torch.clamp(new_levels, max=self.max_difficulty)
                self.difficulties[level_up_env_ids] = new_levels
                self.consecutive_success_counter[level_up_env_ids] = 0

        return level_up_env_ids

    def process_getter(self, root, path):
        path_parts: list[str | tuple[str, int]] = []
        for part in path.split("."):
            m = re.compile(r"^(\w+)\[(\d+)\]$").match(part)
            if m:
                path_parts.append((m.group(1), int(m.group(2))))
            else:
                path_parts.append(part)

        # Traverse the parts to find the container
        container = root
        for container_path in path_parts[:-1]:
            if isinstance(container_path, tuple):
                # we are accessing a list element
                name, idx = container_path
                # find underlying attribute
                if isinstance(container_path, dict):
                    seq = container[name]  # type: ignore[assignment]
                else:
                    seq = getattr(container, name)
                # save the container for the next iteration
                container = seq[idx]
            else:
                # we are accessing a dictionary key or an attribute
                if isinstance(container, dict) or (hasattr(container, 'keys') and callable(container.keys())):
                    container = container[container_path]
                else:
                    container = getattr(container, container_path)

        # save the container and the last part of the path
        self._container = container
        self._last_path = path_parts[-1]  # for "a.b[2].c", this is "c", while for "a.b[2]" it is 2

        # build the getter and setter
        if isinstance(self._container, tuple):
            get_value = lambda: self._container[self._last_path]  # noqa: E731


        elif isinstance(self._container, (list, dict)):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

        elif isinstance(self._container, object):
            get_value = lambda: getattr(self._container, self._last_path)  # noqa: E731
        else:
            raise TypeError(
                f"Unable to build accessors for address '{path}'. Unknown type found for access variable:"
                f" '{type(self._container)}'. Expected a list, dict, or object with attributes."
            )

        return get_value

    def _get_new_values(self, env_ids):
        """
        env_ids: the envs that shuold be randomized
        return: values uniformly sampled in the linear increasing range
        """
        if len(env_ids) == 0:
            return

        low = self.initial_low - (self.difficulties[env_ids][:, None] * self.step_size)  # (env_ids, property_shape)
        self.low = torch.clamp(low, min=self.limit_low, max=self.limit_high)

        high = self.initial_high + (self.difficulties[env_ids][:, None] * self.step_size)
        self.high = torch.clamp(high, min=self.limit_low, max=self.limit_high)

        values = torch.rand_like(self.high) * (high - low) + low

        return values

    def __call__(self, *args, **kwargs):
        pass

    def _set_defaults(self):
        raise NotImplementedError

class RobotCSADR(ConsecutiveSuccessADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert isinstance(self.asset, Articulation)
        if not self.asset_cfg.joint_names:
            self.asset_cfg.joint_names = '.*'
        if not self.asset_cfg.body_names:
            self.asset_cfg.body_names = '.*'
        self.joint_ids, self.joint_names = self.asset.find_joints(self.asset_cfg.joint_names)
        self.body_ids, self.body_names = self.asset.find_bodies(self.asset_cfg.body_names)

class RobotJointCSADR(RobotCSADR):
    def _get_new_values(self, env_ids):
        return super()._get_new_values(env_ids)[:, self.joint_ids]
    

class RobotBodyCSADR(RobotCSADR):
    def _get_new_values(self, env_ids):
        return super()._get_new_values(env_ids)[:, self.body_ids]
    

class RobotJointStiffnessCSADR(RobotJointCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        level_up_env_ids = self._manage_difficulty(env_ids)

        if len(level_up_env_ids) == 0:
            return

        values = self._get_new_values(level_up_env_ids)
        self.asset.write_joint_stiffness_to_sim(values, env_ids=level_up_env_ids, joint_ids=self.joint_ids)  # type: ignore

    def _set_defaults(self):
        self.defaults = self.asset.data.joint_stiffness


class RobotDampingADR(RobotJointCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        level_up_env_ids = self._manage_difficulty(env_ids)

        if len(level_up_env_ids) == 0:
            return

        values = self._get_new_values(level_up_env_ids)
        self.asset.write_joint_damping_to_sim(values, env_ids=level_up_env_ids, joint_ids=self.joint_ids)
    
    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_damping

# default_joint_friction is deprecated, use default_joint_friction_coeff
class RobotJointStaticFrictionADR(RobotJointCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        level_up_env_ids = self._manage_difficulty(env_ids)

        if len(level_up_env_ids) == 0:
            return

        values = self._get_new_values(level_up_env_ids)
        self.asset.write_joint_friction_coefficient_to_sim(values, joint_ids=self.joint_ids, 
                                                           env_ids=level_up_env_ids)

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_friction_coeff.clone()

class RobotJointDynamicFrictionADR(RobotJointCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        level_up_env_ids = self._manage_difficulty(env_ids)

        if len(level_up_env_ids) == 0:
            return

        values = self._get_new_values(level_up_env_ids)
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=self.asset.data.joint_friction_coeff[level_up_env_ids[:, None], self.joint_ids],
            joint_dynamic_friction_coeff=values, 
            joint_ids=self.joint_ids, 
            env_ids=level_up_env_ids)

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_dynamic_friction_coeff.clone()


class RobotJointViscousFrictionADR(RobotJointCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        level_up_env_ids = self._manage_difficulty(env_ids)

        if len(level_up_env_ids) == 0:
            return

        values = self._get_new_values(level_up_env_ids)
        self.asset.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=self.asset.data.joint_friction_coeff[level_up_env_ids[:, None], self.joint_ids],
            joint_viscous_friction_coeff=values, 
            joint_ids=self.joint_ids, 
            env_ids=level_up_env_ids)

    def _set_defaults(self):
        self.defaults = self.asset.data.default_joint_viscous_friction_coeff.clone()


class IndividualArmatureADR(ConsecutiveSuccessADR):
    def __init__(self, cfg, env):
        self.actuator = cfg.params.get('actuators', 'joint')
        super().__init__(cfg, env)

    def reset(self, env_ids=None):
        to_set_ids = self._manage_difficulty(env_ids)

        if len(to_set_ids) == 0:
            return

        values = self._get_new_values(to_set_ids)
        self.asset.actuators[self.actuator].armature[to_set_ids] = values

    def _set_defaults(self):
        self.defaults = self.asset.actuators[self.actuator].armature


class RobotArmatureADR(ManagerTermBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get('asset_cfg')
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.actuators = getattr(cfg, 'actuators', list(self.asset.actuators.keys()))
        self.adrs = []
        original_params = cfg.params
        for k in self.actuators:
            cfg.params = original_params | {'actuators': k}
            adr = IndividualArmatureADR(cfg, env)
            self.adrs.append(adr)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for i, adr in enumerate(self.adrs):
            adr.reset(env_ids)

    def __call__(self, *args, **kwargs):
        pass


class MaterialPropertyADR(ConsecutiveSuccessADR, mdp.randomize_rigid_body_material):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None, *args, **kwargs) -> None:
        to_set_ids = self._manage_difficulty(env_ids)

        if len(to_set_ids) == 0:
            return

        values = self._get_new_values(to_set_ids)

        # if to_set_ids is None:
        #     to_set_ids = torch.arange(self._env.scene.num_envs, device=self.defaults.device)
        # else:
            # to_set_ids = to_set_ids.to(self.defaults.device)

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_physx_view.max_shapes

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()
        values = values.to(materials.device)

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[to_set_ids, start_idx:end_idx] = values[:, start_idx:end_idx]
        else:
            materials[to_set_ids] = values[:]

        self.asset.root_physx_view.set_material_properties(materials, to_set_ids)
    
    def _set_defaults(self):
        self.defaults = self.asset.root_physx_view.get_material_properties()

    def _get_new_values(self, env_ids):
        """
        env_ids: the envs that shuold be randomized
        return: values uniformly sampled in the linear increasing range
        """
        if len(env_ids) == 0:
            return

        low = self.initial_low - (self.difficulties[env_ids][:, None, None] * self.step_size)  # (env_ids, property_shape)
        self.low = torch.clamp(low, min=self.limit_low, max=self.limit_high)

        high = self.initial_high + (self.difficulties[env_ids][:, None, None] * self.step_size)
        self.high = torch.clamp(high, min=self.limit_low, max=self.limit_high)

        values = torch.rand_like(self.high) * (high - low) + low

        return values
    

class RobotMassADR(RobotBodyCSADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.recompute_inertia = cfg.params.get("recompute_inertia", True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        to_set_ids = self._manage_difficulty(env_ids)

        if len(to_set_ids) == 0:
            return

        values = self._get_new_values(to_set_ids)
        # self.asset.root_physx_view.set_masses(values, env_ids=to_set_ids)

        # resolve body indices

        body_ids = torch.tensor(self.body_ids, dtype=torch.int, device=self.defaults.device)

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.root_physx_view.get_masses()

        masses[to_set_ids[:, None], body_ids] = values

        # set the mass into the physics simulation
        self.asset.root_physx_view.set_masses(masses, to_set_ids)

        # recompute inertia tensors if needed
        if self.recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[to_set_ids[:, None], body_ids] / self.asset.data.default_mass[to_set_ids[:, None], body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[to_set_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[to_set_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[to_set_ids] = self.asset.data.default_inertia[to_set_ids] * ratios
            # set the inertia tensors into the physics simulation
            self.asset.root_physx_view.set_inertias(inertias, to_set_ids)
    
    def _set_defaults(self):
        self.defaults = self.asset.data.default_mass

class AllObjecCSADR(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfgs = cfg.params.get("asset_cfgs", [SceneEntityCfg(name=f'object_{i}') for i in range(10)])
        self.asset_names = [cfg.name for cfg in self.asset_cfgs]
        self.assets = [env.scene[name] for name in self.asset_names]
        env.active_asset_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        cls = cfg.params.get("cls")

        self.adrs = []
        original_params = cfg.params
        for asset_cfg in self.asset_cfgs:
            cfg.params = original_params | {'asset_cfg': asset_cfg}
            adr = cls(cfg, env)
            self.adrs.append(adr)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for adr in self.adrs:
            adr.reset(env_ids)

    def __call__(self, *args, **kwargs):
        pass

class ObjectMassADR(ConsecutiveSuccessADR):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.recompute_inertia = cfg.params.get("recompute_inertia", True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        to_set_ids = self._manage_difficulty(env_ids)

        if len(to_set_ids) == 0:
            return

        values = self._get_new_values(to_set_ids)
        # self.asset.root_physx_view.set_masses(values, env_ids=to_set_ids)

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device=self.defaults.device)
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device=self.defaults.device)

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.root_physx_view.get_masses()

        masses[to_set_ids[:, None], body_ids] = values

        # set the mass into the physics simulation
        self.asset.root_physx_view.set_masses(masses, to_set_ids)

        # recompute inertia tensors if needed
        if self.recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[to_set_ids[:, None], body_ids] / self.asset.data.default_mass[to_set_ids[:, None], body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[to_set_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[to_set_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[to_set_ids] = self.asset.data.default_inertia[to_set_ids] * ratios
            # set the inertia tensors into the physics simulation
            self.asset.root_physx_view.set_inertias(inertias, to_set_ids)
    
    def _set_defaults(self):
        self.defaults = self.asset.data.default_mass


#TODO: think of better way of randomizing quaternion
class ObjectScaleAndPosADR(ConsecutiveSuccessADR):
    """
    sets the position and quaternion of the object
    if the object's difficulty doesn't match the current difficulty, it will be moved under the ground plane
    and if it matches, the object will be moved to workspace
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.pos_device = self.asset.data.root_pos_w.device
        if len(self.asset_cfg.name.split("_")) == 2:
            self.asset_difficulty = int(self.asset_cfg.name.split("_")[1])
        else:
            self.asset_difficulty = 0

        randomize_z = cfg.params.get("randomize_z", False)
        if not randomize_z:
            self.step_size[2] = 0

        self.limit_low = torch.tensor((-1., -1., 0., -1., -1., -1., -1.), device=self.pos_device)
        self.limit_high = torch.tensor((1., 1., 1.0,  1., 1., 1., 1.), device=self.pos_device)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = env_ids.to(self.defaults.device)
        self._manage_difficulty(env_ids)

        match_ids, not_match_ids = self._get_difficylty_matches_and_unmatches(env_ids, self.difficulties[env_ids])

        if len(not_match_ids) > 0 and (self.asset.data.root_pos_w[not_match_ids, 2] > -50.).any():
            away_pos = self.defaults[not_match_ids]
            away_pos[:, 2] = -100.
            self.asset.write_root_pose_to_sim(away_pos, not_match_ids)

        # if len(to_set_ids) == 0:
        #     return

        if len(match_ids) > 0:
            values = self._get_new_values(match_ids)
            self.asset.write_root_pose_to_sim(values, match_ids)
            self._env.active_asset_indices[match_ids] = self.asset_difficulty

    def _set_defaults(self):
        # self.defaults = self.asset.data.cfg.spawn.scale
        self.defaults = torch.concat((self.asset.data.root_pos_w, self.asset.data.root_quat_w), dim=-1)
        # if (self.defaults[:, 2] < -50.).any():
        #     self.defaults[:, 2] = self.workspace_pos[-1]

    def _get_new_values(self, env_ids):
        values = super()._get_new_values(env_ids)

        # normalize the quaternions
        quats = values[:, 3:]
        values[:, 3:] = quats / quats.sum(dim=-1, keepdim=True)

        return values

    def _get_difficylty_matches_and_unmatches(self, env_ids, difficulties):
        difficulty_matches = difficulties[env_ids] == self.asset_difficulty
        match_ids = env_ids[difficulty_matches]
        not_match_ids = env_ids[~difficulty_matches]
        return match_ids, not_match_ids

    def maybe_get_asset(self):
        match_ids, not_match_ids = self._get_difficylty_matches_and_unmatches(self.difficulties, torch.range(self.num_envs))
        if len(match_ids) > 0:
            return self.asset, match_ids
        return None, None



def active_pool_wrapper(mdp_func, asset_names: list[str]):
    """
    Wraps a standard Isaac Lab mdp observation function.
    """

    # 1. REMOVED **kwargs: We strictly only ask for 'env' now.
    # Isaac Lab's config parser will be perfectly happy with this.
    def wrapped_obs_func(env: ManagerBasedRLEnv) -> torch.Tensor:
        all_results = []

        for name in asset_names:
            temp_cfg = SceneEntityCfg(name)

            # 2. Call the standard mdp function directly with just what it needs
            data = mdp_func(env, asset_cfg=temp_cfg)
            all_results.append(data)

        stacked_data = torch.stack(all_results, dim=1)

        # 3. Read the state saved by our Event Term
        active_idx = getattr(env, "active_asset_indices", None)

        if active_idx is None:
            active_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        batch_idx = torch.arange(env.num_envs, device=env.device)

        return stacked_data[batch_idx, active_idx]

    return wrapped_obs_func
