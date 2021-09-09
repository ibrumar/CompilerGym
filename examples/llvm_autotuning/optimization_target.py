# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from enum import Enum
from threading import Lock
from time import time
from typing import List

import gym
import numpy as np
from llvm_autotuning.runtime_reward_wrapper import LlvmMedianRuntimeReward

from compiler_gym.envs import LlvmEnv
from compiler_gym.service.connection import ServiceError

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = Lock()


class OptimizationTarget(str, Enum):
    CODESIZE = "codesize"
    BINSIZE = "binsize"
    RUNTIME = "runtime"

    @property
    def optimization_space_enum_name(self) -> str:
        return {
            OptimizationTarget.CODESIZE: "IrInstructionCount",
            OptimizationTarget.BINSIZE: "ObjectTextSizeBytes",
            OptimizationTarget.RUNTIME: "Runtime",
        }[self.value]

    def make_env(self, benchmark: str) -> LlvmEnv:
        env: LlvmEnv = gym.make("llvm-v0", benchmark=benchmark)
        if self.value == OptimizationTarget.CODESIZE:
            env.reward_space = "IrInstructionCountOz"
        elif self.value == OptimizationTarget.BINSIZE:
            env.reward_space = "ObjectTextSizeOz"
        elif self.value == OptimizationTarget.RUNTIME:
            env = LlvmMedianRuntimeReward(env, n=3)
        else:
            assert False, f"Unknown OptimizationTarget: {self.value}"
        return env

    def final_reward(self, env: LlvmEnv) -> float:
        """Compute the final reward of the environment.

        Note that this may modify the environment state. You should call
        :code:`reset()` before continuing to use the environment after this.
        """
        # Reapply the environment state in a retry loop.
        actions = list(env.actions)
        env.reset()
        for i in range(1, 5 + 1):
            _, _, done, info = env.step(actions)
            if not done:
                break
            logger.warning(
                "Attempt %d to apply actions during final reward failed: %s",
                i,
                info.get("error_details"),
            )
        else:
            raise ValueError("Failed to replay environment's actions")

        if self.value == OptimizationTarget.CODESIZE:
            return (
                env.observation.IrInstructionCountOz()
                / env.observation.IrInstructionCount()
            )

        if self.value == OptimizationTarget.BINSIZE:
            return (
                env.observation.ObjectTextSizeOz()
                / env.observation.ObjectTextSizeBytes()
            )

        if self.value == OptimizationTarget.RUNTIME:
            with _RUNTIME_LOCK:
                final_runtimes = measure_runtimes(env, n=30, min_measurement_seconds=10)
                env.reset()
                env.send_param("llvm.apply_baseline_optimizations", "-O3")
                o3_runtimes = measure_runtimes(env, n=30, min_measurement_seconds=10)
                return np.median(o3_runtimes) / np.median(final_runtimes)

        assert False, f"Unknown OptimizationTarget: {self.value}"


def measure_runtimes(
    env: LlvmEnv, n: int = 30, min_measurement_seconds: float = 10
) -> List[float]:
    # 15 min timeout on RPC calls to accomodate for lengthy slow runtimes.
    old_service_timeout = env.service.opts.rpc_call_max_seconds
    env.service.opts.rpc_call_max_seconds = 900
    try:
        env.send_param("llvm.set_runtimes_per_observation_count", str(n))
        runtimes = []
        end_time = time() + min_measurement_seconds
        while time() < end_time:
            try:
                r = env.observation.Runtime().tolist()
                assert len(r) == n
                runtimes += r
            except ServiceError as e:
                if "Deadline exceeded" in str(e):
                    return runtimes.append(float("inf"))
        return runtimes
    finally:
        env.service.opts.rpc_call_max_seconds = old_service_timeout
