# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym
import pytest

from compiler_gym.envs.llvm import LlvmEnv

from .runtime_reward_wrapper import LlvmMedianRuntimeReward


@pytest.fixture(scope="function")
def env() -> LlvmEnv:
    """Create an LLVM environment."""
    with gym.make("llvm-v0") as env_:
        yield env_


def test_reward_range(env: LlvmEnv):
    env = LlvmMedianRuntimeReward(env, n=3)
    assert env.reward_range == (-float("inf"), float("inf"))


def test_reward_range_not_runnable_benchmark(env: LlvmEnv):
    env = LlvmMedianRuntimeReward(env, n=3)

    with pytest.raises(
        ValueError, match=r"^Benchmark is not runnable: benchmark://npb-v0/1$"
    ):
        env.reset(benchmark="benchmark://npb-v0/1")


def test_reward_values(env: LlvmEnv):
    env = LlvmMedianRuntimeReward(env, n=3)
    env.reset()

    _, reward_a, done, info = env.step(env.action_space.sample())
    assert not done, info

    _, reward_b, done, info = env.step(env.action_space.sample())
    assert not done, info

    _, reward_c, done, info = env.step(env.action_space.sample())
    assert not done, info

    assert env.episode_reward == reward_a + reward_b + reward_c
    assert reward_a or reward_b or reward_c
