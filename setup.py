# -*- coding: utf-8 -*-

"""Setup conditional musegan Library."""

import os
import sys
from distutils.version import LooseVersion

import pip
from setuptools import find_packages, setup

if LooseVersion(sys.version) < LooseVersion("3.8"):
    raise RuntimeError(
        "conditional_musegan requires Python>=3.8, " "but your Python is {}".format(sys.version)
    )
if LooseVersion(pip.__version__) < LooseVersion("21.0.0"):
    raise RuntimeError(
        "pip>=21.0.0 is required, but your pip is {}. "
        'Try again after "pip install -U pip"'.format(pip.__version__)
    )

requirements = {
    "install": [
        "wheel",
        "torch>=1.9.0",
        "setuptools>=38.5.1",
        "tensorboardX>=2.2",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "h5py>=2.10.0",
        "protobuf<=3.20.0",
        "hydra-core>=1.2",
        "midi2audio>=0.1.1",
        "pypianoroll>=1.0.4",
        "scipy>=1.10.1",
        "tensorboard==2.12.0",
    ],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
}
entry_points = {
    "console_scripts": [
        "conditional_musegan-compute-statistics=conditional_musegan.bin.compute_statistics:main",
        "conditional_musegan-train=conditional_musegan.bin.train:main",
        "conditional_musegan-decode=conditional_musegan.bin.decode:main",
        "conditional_musegan-train_valid_eval_split=conditional_musegan.bin.train_valid_eval_split:main",
        "conditional_musegan-param-count=conditional_musegan.bin.param_count:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="conditional_musegan",
    version="0.0.1",
    url="http://github.com/KateSawada/conditional_musegan",
    author="KateSawada",
    description="Conditional MuseGAN implementation",
    long_description_content_type="text/markdown",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    license="MIT License",
    packages=find_packages(include=["conditional_musegan*"]),
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=[
        "Programming Language :: Python :: 3.9.1",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
