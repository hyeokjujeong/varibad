"""
T-MAZE environment for VariBAD.

Based on Ni et al. 2023 (arXiv:2307.03864), "When Do Transformers Shine in RL?
Decoupling Memory from Credit Assignment". Reward formula from paper Section
3.2; observation encoding mirrors Memory-RL's `ambiguous_position=True` scheme
(reference: /PublicSSD/shnoh/Memory-RL/envs/tmaze.py).

Supports both `mode='passive'` (Sprint 1) and `mode='active'` (Sprint 2).
External-facing args: `corridor_length`, `mode`. All other env details are
hard-coded per TMAZE_guideline_v6.md sections 3 and 7.

Coordinate convention (Memory-RL): oracle at x=0, start at x=oracle_length,
junction at x = oracle_length + corridor_length. Passive sets oracle_length=0
(so S=O at x=0); active sets oracle_length=1 (so S=x=1, O=x=0).
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class TMazeEnv(gym.Env):
    metadata = {'render.modes': []}

    # =========================================================================
    # 3a. Skeleton: __init__, seed, reset_task, get_task, reset
    # =========================================================================

    def __init__(self, corridor_length: int = 10, mode: str = 'passive'):
        super().__init__()
        assert corridor_length >= 1, \
            f"corridor_length must be >= 1, got {corridor_length}"
        assert mode in ('passive', 'active'), \
            f"mode must be 'passive' or 'active', got {mode!r}"

        self.corridor_length = corridor_length
        self.mode = mode
        if mode == 'passive':
            self.oracle_length = 0          # S = O at x = 0
            self._max_episode_steps = corridor_length + 1   # T = L + 1
        else:  # active
            self.oracle_length = 1          # S = x=1, O = x=0
            # T = L + 3 follows Memory-RL's episode_length = L + 2*oracle_length + 1.
            # v6 §3 explicitly adopts the code over the paper's "L = T-2".
            self._max_episode_steps = corridor_length + 3

        # Action: Discrete(4). Order per guideline v6 §7-1.
        #   0: right, 1: up, 2: down, 3: left
        # Note: differs from Memory-RL's index order (right/up/left/down); v6's
        # explicit mapping is the authoritative spec for our env.
        self.action_space = spaces.Discrete(4)
        self._action_mapping = np.array([
            [+1,  0],   # 0: right
            [ 0, +1],   # 1: up
            [ 0, -1],   # 2: down
            [-1,  0],   # 3: left
        ], dtype=np.int64)

        # Observation: Memory-RL ambiguous_position=True encoding, shape (2,).
        #   x = 0 (oracle):              [0, exposure]  (first visit only)
        #   0 < x < L (corridor middle): [0, 0]
        #   x = L (junction or arm):     [1, y]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Task: hidden context goal_y in {-1, +1}. task_dim=1 so that
        # VariBadWrapper exposes it correctly through info['task'].
        self.task_dim = 1

        # gym standard RNG. parallel_envs.make_env will call env.seed(seed+rank)
        # after construction; the default seed here just ensures self.np_random
        # is a valid Generator before any reset_task call.
        self.np_random = None
        self.seed()

        # Initialise task and MDP state so the env is usable immediately after
        # construction (some gym tooling probes obs/action spaces by calling
        # reset()).
        self._goal_y = None
        self.x = 0
        self.y = 0
        self.step_count = 0
        self.oracle_visited = False
        self.reset_task()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task=None):
        """Reset only the hidden context; do NOT reset the MDP state.

        VariBAD convention — see environments/example_env.py and the
        VariBadWrapper.reset path. `task=None` samples uniformly from
        {-1, +1}; an explicit array-like with value in {-1, +1} forces it.
        """
        if task is None:
            self._goal_y = int(self.np_random.choice([-1, +1]))
        else:
            task_arr = np.asarray(task).reshape(-1)
            assert task_arr.shape == (1,), \
                f"task must be a scalar / length-1 array, got shape {task_arr.shape}"
            v = int(task_arr.item())
            assert v in (-1, +1), \
                f"task value must be in {{-1, +1}}, got {v}"
            self._goal_y = v
        return self.get_task()

    def get_task(self):
        return np.array([self._goal_y], dtype=np.float32)

    def reset(self):
        """Reset the MDP state. Does not reset the task."""
        self.x = self.oracle_length  # passive: start at x=0
        self.y = 0
        self.step_count = 0
        self.oracle_visited = False
        return self._get_obs()

    # =========================================================================
    # 3b. Dynamics: step, _is_valid, _get_obs, _reward
    # =========================================================================

    def step(self, action):
        # Coerce action to a Python int regardless of caller's container type.
        # PPO/A2C in this repo emit actions as torch tensors → numpy arrays at
        # the env boundary (VecPyTorch in parallel_envs.py).
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action[0])
        assert self.action_space.contains(action), f"invalid action {action!r}"

        self.step_count += 1

        # Attempt move; on wall bump, stay in place.
        dx, dy = self._action_mapping[action]
        nx, ny = self.x + int(dx), self.y + int(dy)
        if self._is_valid(nx, ny):
            self.x, self.y = nx, ny

        done = self.step_count >= self._max_episode_steps
        reward = self._reward(done)
        obs = self._get_obs()
        info = {'task': self.get_task()}
        return obs, float(reward), done, info

    def _is_valid(self, x, y):
        """Walls of the T-maze.

        Corridor lies on y=0, x in [0, L]. Goal arms (y = +/-1) exist only at
        x = L (junction column). Anywhere else is wall → agent stays.
        """
        x_max = self.oracle_length + self.corridor_length  # = L in passive
        if x < 0 or x > x_max:
            return False
        if y == 0:
            return True
        if y in (-1, +1):
            return x == x_max
        return False

    def _get_obs(self):
        """Ambiguous-position observation per Memory-RL.

        The oracle exposure (the goal_y signal) is shown only on the FIRST
        visit to x=0; subsequent visits return 0. In passive mode the agent
        starts at the oracle, so the first `reset()` returns [0, goal_y] and
        all later observations carry no exposure.
        """
        exposure = 0
        if self.x == 0 and not self.oracle_visited:
            exposure = self._goal_y
            self.oracle_visited = True

        x_junction = self.oracle_length + self.corridor_length  # = L
        if self.x == 0:
            obs = np.array([0.0, float(exposure)], dtype=np.float32)
        elif self.x < x_junction:
            obs = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # x == x_junction: junction (y=0) or one of the goal arms (y=±1).
            obs = np.array([1.0, float(self.y)], dtype=np.float32)
        return obs

    def _reward(self, done):
        """Paper Section 3.2, generalized to active via oracle_length grace.

        Intermediate (t < T):  R_t = (1[x_{t+1} >= t - oracle_length] - 1) / (T - 1)
        Terminal     (t = T):  R_T = 1[reached correct goal arm]

        For passive (oracle_length=0) this matches v6 §3 verbatim. For active
        (oracle_length=1) the agent has `oracle_length` step(s) of grace —
        Memory-RL's reward_fn uses the same condition `x < t - oracle_length`.

        Here `self.step_count` is t (already incremented in step()), and
        `self.x` / `self.y` are the positions AFTER the action — i.e. the
        paper's x_{t+1}, y_{t+1}.
        """
        T = self._max_episode_steps
        if done:
            x_max = self.oracle_length + self.corridor_length
            at_correct_arm = (self.x == x_max) and (self.y == self._goal_y)
            return 1.0 if at_correct_arm else 0.0
        t = self.step_count
        return 0.0 if self.x >= t - self.oracle_length else -1.0 / (T - 1)
