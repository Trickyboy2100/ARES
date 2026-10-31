"""Explicit altered-physics PlugCharger family; no change to native geometry.

Geometry/visual construction follows Apache-2.0 ManiSkill PlugChargerEnv at
62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3. Physics changes happen during
construction, before GPU initialization. Import this module before gym.make.
"""
import math

import sapien
from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("ARES-AdaptivePlug-v0", max_episode_steps=200)
class AdaptivePlugEnv(PlugChargerEnv):
    """Same initial geometry/success/horizon, configurable tool/socket physics."""

    def __init__(self, *args, charger_density_scale=1.0,
                 socket_friction=None, **kwargs):
        self.charger_density_scale = float(charger_density_scale)
        self.socket_friction = (None if socket_friction is None
                                else float(socket_friction))
        if (not math.isfinite(self.charger_density_scale)
                or self.charger_density_scale <= 0):
            raise ValueError("charger_density_scale must be finite and positive")
        if self.socket_friction is not None and (
                not math.isfinite(self.socket_friction)
                or self.socket_friction < 0):
            raise ValueError("socket_friction must be None or finite nonnegative")
        super().__init__(*args, **kwargs)

    def adaptation_metadata(self):
        """Configuration provenance; actual collider/body readback is separate."""
        return dict(
            task="ARES-AdaptivePlug-v0", native_reference="PlugCharger-v1",
            distribution="CUSTOM_PHYSICS_NOT_OFFICIAL_NATIVE_SCORE",
            geometry_initialization_success="inherited_native_unchanged",
            horizon=200, physics_applied="actor_builder_before_gpu_init",
            charger_density_scale=self.charger_density_scale,
            charger_density_kg_m3=1000.0 * self.charger_density_scale,
            socket_friction=self.socket_friction,
            socket_friction_mode=("native_default_material" if
                                  self.socket_friction is None else
                                  "explicit_static_equals_dynamic"),
            native_control=(self.charger_density_scale == 1.0
                            and self.socket_friction is None),
            proposed_density_scales=[0.5, 1.0, 2.0],
            proposed_socket_frictions=[0.2, 0.5, 1.0],
        )

    def physical_readback(self):
        """Read constructed PhysX components, rather than echo requested values.

        These are the component properties used at GPU initialization. This
        audit does not independently measure a GPU mass through force response.
        Each body is retained separately for vectorized environments.
        """
        def shapes(body):
            result = []
            for index, shape in enumerate(body.get_collision_shapes()):
                material = shape.physical_material
                result.append(dict(
                    shape_index=index,
                    density_kg_m3=float(shape.density),
                    static_friction=float(material.static_friction),
                    dynamic_friction=float(material.dynamic_friction),
                    restitution=float(material.restitution),
                ))
            return result

        charger = []
        for index, body in enumerate(self.charger._bodies):
            charger.append(dict(
                body_index=index,
                mass_kg=float(body.mass),
                inertia_kg_m2=[float(x) for x in body.inertia],
                collisions=shapes(body),
            ))
        socket = [dict(body_index=index, collisions=shapes(body))
                  for index, body in enumerate(self.receptacle._bodies)]
        return dict(
            source="constructed_physx_component_readback",
            gpu_dynamics_independently_identified=False,
            charger=charger,
            socket=socket,
        )

    def _build_charger(self, peg_size, base_size, gap):
        if self.charger_density_scale == 1.0:
            return super()._build_charger(peg_size, base_size, gap)
        builder = self.scene.create_actor_builder()
        density = 1000.0 * self.charger_density_scale
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic, mat.roughness, mat.specular = 1.0, 0.0, 1.0
        for y in (gap, -gap):
            pose = sapien.Pose([peg_size[0], y, 0])
            builder.add_box_collision(pose, peg_size, density=density)
            builder.add_box_visual(pose, peg_size, material=mat)
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic, mat.roughness = 0.0, 0.1
        pose = sapien.Pose([-base_size[0], 0, 0])
        builder.add_box_collision(pose, base_size, density=density)
        builder.add_box_visual(pose, base_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        if self.socket_friction is None:
            return super()._build_receptacle(peg_size, receptacle_size, gap)
        builder = self.scene.create_actor_builder()
        contact = sapien.physx.PhysxMaterial(
            self.socket_friction, self.socket_friction, 0.0)
        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy, dz = peg_size[1] + gap + sy, peg_size[2] + sz
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic, mat.roughness = 0.0, 0.1
        poses = [sapien.Pose([dx, 0, dz]), sapien.Pose([dx, 0, -dz]),
                 sapien.Pose([dx, dy, 0]), sapien.Pose([dx, -dy, 0])]
        half_sizes = [[receptacle_size[0], receptacle_size[1], sz]] * 2 + [
            [receptacle_size[0], sy, receptacle_size[2]]] * 2
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size, material=contact)
            builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size, material=contact)
        builder.add_box_visual(pose, half_size, material=mat)
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic, mat.roughness, mat.specular = 1.0, 0.0, 1.0
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        for y in (-(gap * 0.5 + peg_size[1]), gap * 0.5 + peg_size[1]):
            builder.add_box_visual(sapien.Pose([-receptacle_size[0], y, 0]),
                                   half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")
