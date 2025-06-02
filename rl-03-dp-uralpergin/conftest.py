
import pytest
from gridworld import GridworldEnv

def pytest_configure(config):
    pytest.global_env = GridworldEnv()
