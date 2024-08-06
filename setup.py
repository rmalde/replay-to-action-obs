from setuptools import setup, find_packages
import subprocess
import os

# you'll need to manually install rlgym_sim
# pip install git+https://github.com/AechPro/rocket-league-gym-sim@main

setup(
    name="replay_to_action_obs",  # Replace with your package name
    version="0.1.0",
    description="Download rocket league replays and convert to rlgym action and obs",
    author="rmalde",
    url="https://github.com/rmalde/replay_to_action_obs",  # Replace with your project's URL
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "rich",
        "python-dotenv",
        "requests",
        "numpy",
        "numba",
        "scipy",
        "pandas",
        "pyarrow",
        "zipfile",
        "rlgym[all]>=2.0.0rc0",
        "python-ballchasing",
        "rlgym-tools @ git+https://github.com/RLGym/rlgym-tools.git@v2",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license if different
        "Operating System :: OS Independent",
    ],
)

# additional repo installs
# pip install git+https://github.com/lucas-emery/rocket-league-gym.git@v2
# pip install git+https://github.com/RLGym/rlgym-tools.git@v2
