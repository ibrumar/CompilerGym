import distutils.util

import setuptools

setuptools.setup(
    name="compiler_gym_examples",
    version="0.0.10",
    description="Reinforcement learning environments for compiler research",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/CompilerGym",
    license="MIT",
    packages=[
        "llvm_autotuning",
        "llvm_autotuning.autotuners",
    ],
    python_requires=">=3.8",
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)
