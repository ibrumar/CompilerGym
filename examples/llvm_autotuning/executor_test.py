# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integration tests for the LLVM autotuners."""
import pytest
from llvm_autotuning.executor import Executor


def _hello_fn():
    return "Hello, world"


# FIXME(cummins): Add "slurm" to list of executor types to test.
@pytest.mark.parametrize("type", ["local", "debug"])
def test_no_args_call(tmpdir, type: str):
    with Executor(type=type, cpus=1).get_executor(logs_dir=tmpdir) as executor:
        job = executor.submit(_hello_fn)

    assert job.result() == "Hello, world"


def _add_fn(a, b, *args, **kwargs):
    return a + b + sum(args) + kwargs["c"]


# FIXME(cummins): Add "slurm" to list of executor types to test.
@pytest.mark.parametrize("type", ["local", "debug"])
def test_call_with_args(tmpdir, type: str):
    with Executor(type=type, cpus=1).get_executor(logs_dir=tmpdir) as executor:
        job = executor.submit(_add_fn, 1, 1, 1, 1, c=1, d=None)

    assert job.result() == 5
