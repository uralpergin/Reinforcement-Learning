
import numpy as np
import pytest
from cliff_walking import CliffWalkingEnv

def pytest_configure(config):
    pytest.global_env = CliffWalkingEnv()


