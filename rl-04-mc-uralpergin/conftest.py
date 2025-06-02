
import pytest
from blackjack import BlackjackEnv

def pytest_configure(config):
    pytest.global_env = BlackjackEnv(test=True)
