# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.envs import CompilerEnv


def greedy(env: CompilerEnv, **kwargs) -> None:
    """A greedy search policy.

    :param env: The environment to optimize.
    """

    def eval_action(env: CompilerEnv, action: int):
        with env.fork() as fkd:
            return (fkd.step(action)[1], action)

    while True:
        best = max(eval_action(env, action) for action in range(env.action_space.n))
        if best[0] < 0 or env.step(best[1])[2]:
            return
