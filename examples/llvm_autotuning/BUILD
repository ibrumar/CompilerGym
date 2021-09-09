# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "benchmarks",
    srcs = ["benchmarks.py"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/envs/llvm",
    ],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    imports = ["examples/llvm_autotuning"],
    deps = [
        ":runtime_reward_wrapper",
        "//compiler_gym/envs/llvm",
    ],
)

py_test(
    name = "executor_test",
    srcs = ["executor_test.py"],
    deps = [
        ":executor",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    visibility = ["//examples/llvm_autotuning:__subpackages__"],
    deps = [
        ":benchmarks",
        ":executor",
        ":optimization_target",
        ":runtime_reward_wrapper",
        "//compiler_gym",
        "//examples/llvm_autotuning/autotuners",
    ],
)

py_binary(
    name = "info",
    srcs = ["info.py"],
    deps = [
        ":experiment",
    ],
)

py_library(
    name = "optimization_target",
    srcs = ["optimization_target.py"],
    visibility = ["//examples/llvm_autotuning:__subpackages__"],
    deps = [
        ":runtime_reward_wrapper",
        "//compiler_gym/envs/llvm",
    ],
)

py_library(
    name = "runtime_reward_wrapper",
    srcs = ["runtime_reward_wrapper.py"],
    deps = [
        "//compiler_gym/envs/llvm",
        "//compiler_gym/service:connection",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
        "//compiler_gym/wrappers",
    ],
)

py_test(
    name = "runtime_reward_wrapper_test",
    srcs = ["runtime_reward_wrapper_test.py"],
    deps = [
        ":runtime_reward_wrapper",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_binary(
    name = "tune",
    srcs = ["tune.py"],
    data = ["//examples/llvm_autotuning/config"],
    deps = [
        ":experiment",
        "//compiler_gym/util",
    ],
)
