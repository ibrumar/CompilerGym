# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from threading import Lock
from typing import Optional

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ObservationType, StepType
from compiler_gym.wrappers.core import CompilerEnvWrapper

_GLOBAL_STEP_LOCK = Lock()


class LockedStep(CompilerEnvWrapper):
    """TODO."""

    def __init__(self, env: CompilerEnv, lock: Optional[Lock] = None):
        """Constructor.

        :param env: The environment to wrap.

        :param lock: The thread lock to acquire before calls to :code:`ste()`.
            If not provided, a default lock is used and shared.
        """
        super().__init__(env)
        self.lock = lock or _GLOBAL_STEP_LOCK

    def reset(self, *args, **kwargs) -> ObservationType:
        with self.lock:
            return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs) -> StepType:
        with self.lock:
            return self.env.step(*args, **kwargs)

    @property
    def observation(self):
        with self.lock:
            return self.env.observation

    @property
    def reward(self):
        with self.lock:
            return self.env.reward

    def fork(self):
        with self.lock:
            fkd = self.env.fork()
            return LockedStep(env=fkd, lock=self.lock)
