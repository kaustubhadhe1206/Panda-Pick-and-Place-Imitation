"""Pick-and-place environment built on MuJoCo Playground's PandaPickCube, run as a batch.

PandaPickCube asks the robot to lift the cube to a point in the air. For this
project I changed only the target: it is now a spot ON THE TABLE on the other
side of the workspace, so the robot has to pick the cube up, carry it over,
put it down and let go. Everything else (robot, cube, physics, observations,
actions, cube start positions) is PandaPickCube unchanged.

  cube start:  x = 0.5 +- 0.2 m, y = +-0.2 m (same as PandaPickCube)
  target:      x = 0.5 +- 0.2 m, |y| in [0.05, 0.2] m on the OPPOSITE side of y = 0
               from the cube, at the cube's resting height (on the table)

Each episode starts from env.reset(jax.random.PRNGKey(seed)), so any episode
can be reproduced from its seed number alone. JAX/MuJoCo are only imported
inside the functions below, so the other modules can be tested without them.
"""

import numpy as np

EPISODE_LENGTH = 250        # 5 seconds at 0.02 s per step (pick + carry + place + release)
TARGET_MIN_ABS_Y = 0.05     # target is at least this far on the other side of y = 0
TARGET_MAX_ABS_Y = 0.20
TARGET_X_RANGE = 0.20       # target x = cube's default x +- this


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
    """PandaPickCube, but the target is a spot on the table on the other side of the cube."""

    def reset(self, random_key):
      random_key, target_key = jax.random.split(random_key)
      state = super().reset(random_key)
      cube_start = state.data.qpos[self._obj_qposadr:self._obj_qposadr + 3]

      x_key, y_key = jax.random.split(target_key)
      target_x = self._init_obj_pos[0] + jax.random.uniform(x_key, minval=-TARGET_X_RANGE, maxval=TARGET_X_RANGE)
      distance_from_middle = jax.random.uniform(y_key, minval=TARGET_MIN_ABS_Y, maxval=TARGET_MAX_ABS_Y)
      other_side = jnp.where(cube_start[1] >= 0, -1.0, 1.0)
      target_pos = jnp.array([target_x, other_side * distance_from_middle, cube_start[2]])

      data = state.data.replace(mocap_pos=state.data.mocap_pos.at[self._mocap_target, :].set(target_pos))
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
