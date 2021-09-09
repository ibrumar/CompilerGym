# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "autotuners",
    srcs = [
        "__init__.py",
        "greedy.py",
        "nevergrad_.py",
        "opentuner_.py",
        "random_.py",
    ],
    visibility = ["//examples/llvm_autotuning:__subpackages__"],
    deps = [
        "//compiler_gym:random_search",
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//examples/llvm_autotuning:optimization_target",
    ],
)

py_test(
    name = "greedy_test",
    srcs = ["greedy_test.py"],
    deps = [
        ":autotuners",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "nevergrad_test",
    srcs = ["nevergrad_test.py"],
    deps = [
        ":autotuners",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "opentuner_test",
    srcs = ["opentuner_test.py"],
    deps = [
        ":autotuners",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "random_test",
    srcs = ["random_test.py"],
    deps = [
        ":autotuners",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)
