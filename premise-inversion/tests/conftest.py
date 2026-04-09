"""Faster defaults for hypothesis smoke tests."""
import pytest

import config


@pytest.fixture(autouse=True)
def fast_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "NUM_STEPS", 120, raising=False)
