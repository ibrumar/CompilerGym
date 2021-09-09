# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This modules defines an interface for describing autotuners for LLVM environments."""
import tempfile
from pathlib import Path
from typing import Any, Dict

from llvm_autotuning.autotuners.greedy import greedy  # noqa Register autotuners
from llvm_autotuning.autotuners.nevergrad_ import nevergrad  # noqa
from llvm_autotuning.autotuners.opentuner_ import opentuner_ga  # noqa
from llvm_autotuning.autotuners.random_ import random  # noqa
from llvm_autotuning.optimization_target import OptimizationTarget
from pydantic import BaseModel, validator

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.capture_output import capture_output
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.temporary_working_directory import temporary_working_directory
from compiler_gym.util.timer import Timer


class Autotuner(BaseModel):

    algorithm: str

    optimization_target: OptimizationTarget

    search_time_seconds: int

    algorithm_config: Dict[str, Any] = {}

    @property
    def autotune(self):
        """Return the autotuner function for this algorithm.

        An autotuner function takes a single CompilerEnv argument and optional
        keyword configuration arguments (determined by algorithm_config) and
        tunes the environment, returning nothing.
        """
        try:
            return globals()[self.algorithm]
        except KeyError as e:
            raise ValueError(
                f"Unknown autotuner: {self.algorithm}.\n"
                f"Make sure the {self.algorithm}() function definition is available "
                "in the autotuners module."
            ) from e

    @property
    def autotune_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "optimization_target": self.optimization_target,
            "search_time_seconds": self.search_time_seconds,
        }
        kwargs.update(self.algorithm_config)
        return kwargs

    def __call__(self, env: CompilerEnv, seed: int = 0xCC) -> CompilerEnvState:
        # Run the autotuner in a temporary working directory and capture the
        # stdout/stderr.
        with tempfile.TemporaryDirectory(
            dir=transient_cache_path("."), prefix="autotune-"
        ) as tmpdir:
            with temporary_working_directory(Path(tmpdir)):
                with capture_output():
                    with Timer() as timer:
                        self.autotune(env, seed=seed, **self.autotune_kwargs)

        return CompilerEnvState(
            benchmark=env.benchmark.uri,
            commandline=env.commandline(),
            walltime=timer.time,
            reward=self.optimization_target.final_reward(env),
        )

    # === Start of implementation details. ===

    @validator("algorithm_config", pre=True)
    def validate_algorithm_config(cls, value) -> Dict[str, Any]:
        return value or {}
