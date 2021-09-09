# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

import numpy as np

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service.connection import ServiceError
from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ObservationType
from compiler_gym.wrappers import CompilerEnvWrapper


class LlvmMedianRuntimeReward(CompilerEnvWrapper):
    class RuntimeReward(Reward):
        def __init__(self, n: int):
            super().__init__(
                id="runtime",
                observation_spaces=["Runtime"],
                default_value=0,
                min=None,
                max=None,
                default_negates_returns=True,
                deterministic=False,
                platform_dependent=True,
            )
            self.n = n
            self.previous_runtime: Optional[float] = None
            self.current_benchmark: Optional[str] = None

        def reset(self, benchmark, observation_view) -> None:
            # If we are changing the benchmark then check that it is runnable.
            if benchmark != self.current_benchmark:
                if not observation_view["IsRunnable"]:
                    raise ValueError(f"Benchmark is not runnable: {benchmark}")
                self.current_benchmark = benchmark

            # NOTE(cummins): We don't compute an initial runtime here because we
            # need to send over the set_runtimes_per_observation_param first.
            self.previous_runtime = None

        def update(
            self,
            actions: List[int],
            observations: List[ObservationType],
            observation_view,
        ) -> float:
            del actions  # unused
            del observation_view  # unused
            runtimes = observations[0]
            if len(runtimes) != self.n:
                raise ServiceError(
                    f"Expected {self.n} runtimes but received {len(runtimes)}"
                )
            runtime = np.median(runtimes)

            reward = self.previous_runtime - runtime
            self.previous_runtime = runtime
            return reward

    def __init__(self, env: LlvmEnv, n: int):
        super().__init__(env)
        self.n = n
        self.env.reward.add_space(self.RuntimeReward(self.n))
        self.env.reward_space = "runtime"

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        # Reset the number of runtimes per observation as this is a
        # session-specific parameter.
        self.env.send_param("llvm.set_runtimes_per_observation_count", str(self.n))
        # Reset the reward space's previous runtime measurement now that we have
        # configured the number of observations we want.
        self.env.reward_space.previous_runtime = np.median(
            self.env.observation.Runtime()
        )
        return obs

    def fork(self) -> "LlvmMedianRuntimeReward":
        fkd = LlvmMedianRuntimeReward(env=self.env.fork(), n=self.n)
        fkd.send_param("llvm.set_runtimes_per_observation_count", str(self.n))
        fkd.reward_space.previous_runtime = self.env.reward_space.previous_runtime
        return fkd
