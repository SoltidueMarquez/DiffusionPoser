import pytest


pytestmark = pytest.mark.skip(reason="Unity 在线模拟本轮明确延期；Python runtime 由 140D smoke 覆盖")


def test_unity_simulation_deferred():
    pass
