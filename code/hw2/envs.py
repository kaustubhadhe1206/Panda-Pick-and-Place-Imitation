"""Pick-and-place environment built on MuJoCo Playground's PandaPickCube, run as a batch.

PandaPickCube asks the robot to lift the cube to a point in the air. For this
project I changed where the cube starts and where it has to go: both are spots
ON THE TABLE, placed around the robot base so that the arm has to swing
45-60 degrees (seen from the top) to carry the cube from one to the other, then
put it down and let go. Everything else (robot, cube, physics, observations,
actions) is PandaPickCube unchanged.

Layout, measured from the robot base (at the origin) as seen from the top:
  angle:     both spots are within 60 degrees left/right of straight ahead (+x),
             and the angle between them (the "turn") is 45-60 degrees, either direction
  distance:  both spots are 0.40-0.60 m from the base (comfortable reach for a
             top-down grasp; the Panda can reach about 0.85 m)
  height:    the cube's resting height on the table

Each episode starts from env.reset(jax.random.PRNGKey(seed)), so any episode
can be reproduced from its seed number alone. JAX/MuJoCo are only imported
inside the functions below, so the other modules can be tested without them.
"""

import numpy as np

EPISODE_LENGTH = 250        # 5 seconds at 0.02 s per step (pick + carry + place + release)
MIN_TURN_DEG = 45.0         # angle between pick and place spots, seen from the top, around the base
MAX_TURN_DEG = 60.0
MAX_ABS_ANGLE_DEG = 60.0    # both spots stay within this angle of straight ahead
MIN_RADIUS = 0.40           # distance of both spots from the robot base (m)
MAX_RADIUS = 0.60


def pick_and_place_from_uniforms(uniforms, xp=np):
  """Turn 5 random numbers in [0, 1) into a pick spot and a place spot (x, y), as described above.

  Works with numpy (for tests) and jax.numpy (inside the env); `uniforms` can
  also have shape (5, N) to make N layouts at once.
  """
  # Only +, -, * and xp functions on `uniforms` (numpy functions would break inside jax.jit).
  degrees_to_radians = np.pi / 180.0
  turn = (MIN_TURN_DEG + uniforms[0] * (MAX_TURN_DEG - MIN_TURN_DEG)) * degrees_to_radians
  lowest_angle = -MAX_ABS_ANGLE_DEG * degrees_to_radians
  highest_first_angle = MAX_ABS_ANGLE_DEG * degrees_to_radians - turn   # so the second spot also stays in range
  first_angle = lowest_angle + uniforms[1] * (highest_first_angle - lowest_angle)
  second_angle = first_angle + turn
  # Randomly pick which of the two spots is the pick spot, so the arm swings left or right.
  swap = uniforms[2] < 0.5
  pick_angle = xp.where(swap, second_angle, first_angle)
  place_angle = xp.where(swap, first_angle, second_angle)
  pick_radius = MIN_RADIUS + uniforms[3] * (MAX_RADIUS - MIN_RADIUS)
  place_radius = MIN_RADIUS + uniforms[4] * (MAX_RADIUS - MIN_RADIUS)
  pick_xy = xp.stack([pick_radius * xp.cos(pick_angle), pick_radius * xp.sin(pick_angle)])
  place_xy = xp.stack([place_radius * xp.cos(place_angle), place_radius * xp.sin(place_angle)])
  return pick_xy, place_xy


def turn_angle_deg(pick_xy, place_xy):
  """Angle (degrees) the arm has to swing around its base, seen from the top. Inputs: (..., 2) arrays."""
  pick_angle = np.arctan2(pick_xy[..., 1], pick_xy[..., 0])
  place_angle = np.arctan2(place_xy[..., 1], place_xy[..., 0])
  return np.degrees(np.abs(place_angle - pick_angle))


def get_env_config():
  """Settings of our pick-and-place env (PandaPickCube defaults + our changes) as a dict."""
  from mujoco_playground._src.manipulation.franka_emika_panda import pick
  config = pick.default_config()
  config.impl = "jax"
  config.episode_length = EPISODE_LENGTH
  return config


def load_pick_place_env():
  """Create the pick-and-place env (a subclass of Playground's PandaPickCube)."""
  import jax
  import jax.numpy as jnp
  from mujoco import mjx
  from mujoco_playground._src.manipulation.franka_emika_panda import pick

  class PandaPickAndPlace(pick.PandaPickCube):
    """PandaPickCube, but the cube starts and must be placed on spots 45-60 degrees apart around the base."""

    def reset(self, random_key):
      random_key, layout_key = jax.random.split(random_key)
      state = super().reset(random_key)   # robot at home pose; we then move the cube and the target

      pick_xy, place_xy = pick_and_place_from_uniforms(jax.random.uniform(layout_key, (5,)), xp=jnp)
      table_height = self._init_obj_pos[2]   # cube center height when it rests on the table
      cube_start = jnp.array([pick_xy[0], pick_xy[1], table_height])
      target_pos = jnp.array([place_xy[0], place_xy[1], table_height])

      cube_slice = slice(self._obj_qposadr, self._obj_qposadr + 3)
      data = state.data.replace(
          qpos=state.data.qpos.at[cube_slice].set(cube_start),
          mocap_pos=state.data.mocap_pos.at[self._mocap_target, :].set(target_pos))
      info = {**state.info, "target_pos": target_pos}
      # PandaPickCube.reset() never runs forward kinematics, so at step 0 the gripper/cube
      # positions (and those parts of the first observation) would all be zeros.
      # mjx.forward fills them in; it does not change the physics.
      data = mjx.forward(self.mjx_model, data)
      return state.replace(data=data, obs=self._get_obs(data, info), info=info)

  return PandaPickAndPlace(config=get_env_config())


class BatchedPickPlaceEnv:
  """Runs many pick-and-place episodes at once on the GPU (jax.vmap + jax.jit).

  reset(seeds) and step(state, actions) work on the whole batch.
  to_numpy(state) copies what the controllers need to the CPU as numpy arrays.
  """

  def __init__(self, env=None):
    import jax
    import jax.numpy as jnp

    self._jax = jax
    self._jnp = jnp
    self.env = env if env is not None else load_pick_place_env()
    self.mj_model = self.env.mj_model
    env_config = get_env_config()
    self.episode_length = int(env_config.episode_length)   # 250 steps
    self.action_scale = float(env_config.action_scale)     # ctrl += 0.04 * action
    self.control_dt = float(env_config.ctrl_dt)            # 0.02 s per step
    self.action_size = int(self.env.action_size)           # 8 = 7 arm joints + gripper

    gripper_site_id = self.mj_model.site("gripper").id
    box_body_id = self.mj_model.body("box").id

    def pick_useful_fields(state):
      return {
          "obs": state.obs,
          "qpos": state.data.qpos,
          "ctrl": state.data.ctrl,
          "grip_pos": state.data.site_xpos[:, gripper_site_id],
          "box_pos": state.data.xpos[:, box_body_id],
          "target_pos": state.info["target_pos"],
          "reward": state.reward,
          "done": state.done,
          "reached_box": state.info["reached_box"],
      }

    self._reset_batch = jax.jit(jax.vmap(self.env.reset))
    self._step_batch = jax.jit(jax.vmap(self.env.step))
    self._pick_useful_fields = jax.jit(pick_useful_fields)

  def reset(self, seeds):
    jax, jnp = self._jax, self._jnp
    seeds = jnp.asarray(np.asarray(seeds), dtype=jnp.int32)
    random_keys = jax.vmap(jax.random.PRNGKey)(seeds)
    return self._reset_batch(random_keys)

  def step(self, state, actions):
    return self._step_batch(state, self._jnp.asarray(actions, dtype=self._jnp.float32))

  def to_numpy(self, state):
    fields = self._jax.device_get(self._pick_useful_fields(state))
    return {name: np.asarray(value) for name, value in fields.items()}
