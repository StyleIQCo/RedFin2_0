"""Model registry invariants: promotion semantics, rollback, stage exclusivity."""
import tempfile
from pathlib import Path

import pytest

from src.registry import ModelRegistry


class _FakeModel:
    def __init__(self, v): self.v = v


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=tmp_path)


def test_register_creates_candidate(registry):
    meta = registry.register("m", _FakeModel(1), 100, "abc", metrics={"mape": 0.1})
    assert meta.stage == "candidate"
    assert meta.version == 1


def test_promotion_archives_prior_production(registry):
    m1 = registry.register("m", _FakeModel(1), 100, "a")
    m2 = registry.register("m", _FakeModel(2), 100, "b")
    registry.promote("m", m1.version, "production")
    registry.promote("m", m2.version, "production")
    versions = {v.version: v.stage for v in registry.list_versions("m")}
    assert versions[1] == "archived"
    assert versions[2] == "production"


def test_load_current_production(registry):
    m1 = registry.register("m", _FakeModel(1), 100, "a")
    registry.promote("m", m1.version, "production")
    loaded, meta = registry.load("m", "production")
    assert loaded.v == 1
    assert meta.stage == "production"


def test_rollback_restores_last_archived(registry):
    m1 = registry.register("m", _FakeModel(1), 100, "a")
    m2 = registry.register("m", _FakeModel(2), 100, "b")
    registry.promote("m", m1.version, "production")
    registry.promote("m", m2.version, "production")  # m1 → archived
    rolled = registry.rollback("m")
    assert rolled.version == 1
    loaded, meta = registry.load("m", "production")
    assert loaded.v == 1
