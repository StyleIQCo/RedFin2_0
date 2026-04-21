"""A lightweight, filesystem-backed model registry.

In production we'd use MLflow Model Registry, SageMaker Model Registry, or a
homegrown service backed by S3 + Postgres. The interface here mimics those —
the point is to show the concepts:

  * Every model has a **name** and immutable **version**.
  * Each version has **lineage** (training data hash, git sha, params, metrics).
  * Models have **stages**: `candidate` → `staging` → `production` → `archived`.
  * Promotion to `production` requires passing a **validation gate**.
  * Rollback is a single metadata update — the artifact is already on disk.

This makes the "CI/CD for ML" story concrete: a training job produces a
`candidate`, CI validates it (offline eval + golden set), promotes to
`staging`, an A/B shadow deploy promotes to `production`, and any regression
triggers rollback to the previously-tagged production version.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib

from src.config import settings

Stage = Literal["candidate", "staging", "production", "archived"]


@dataclass
class ModelMetadata:
    name: str
    version: int
    stage: Stage
    created_at: str
    # Lineage
    training_data_hash: str
    training_data_rows: int
    git_sha: str | None
    # Params + metrics
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    # Free-form tags (owner, framework, etc.)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMetadata":
        return cls(**d)


class ModelRegistry:
    """Filesystem model registry.

    Layout:
      registry/models/<name>/
        ├── v1/
        │   ├── model.joblib
        │   └── metadata.json
        ├── v2/
        └── _INDEX.json          # maps stage → version
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- layout helpers ---
    def _model_dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _version_dir(self, name: str, version: int) -> Path:
        d = self._model_dir(name) / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _index_path(self, name: str) -> Path:
        return self._model_dir(name) / "_INDEX.json"

    def _read_index(self, name: str) -> dict:
        p = self._index_path(name)
        if not p.exists():
            return {"next_version": 1, "stages": {}}
        return json.loads(p.read_text())

    def _write_index(self, name: str, idx: dict) -> None:
        self._index_path(name).write_text(json.dumps(idx, indent=2))

    # --- public API ---
    def register(
        self,
        name: str,
        model: Any,
        training_df_rows: int,
        training_data_hash: str,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        git_sha: str | None = None,
    ) -> ModelMetadata:
        """Register a new version (as `candidate`)."""
        idx = self._read_index(name)
        version = idx["next_version"]
        vdir = self._version_dir(name, version)

        joblib.dump(model, vdir / "model.joblib")
        meta = ModelMetadata(
            name=name,
            version=version,
            stage="candidate",
            created_at=datetime.now(timezone.utc).isoformat(),
            training_data_hash=training_data_hash,
            training_data_rows=training_df_rows,
            git_sha=git_sha,
            params=params or {},
            metrics=metrics or {},
            tags=tags or {},
        )
        (vdir / "metadata.json").write_text(json.dumps(meta.to_dict(), indent=2))

        idx["next_version"] = version + 1
        self._write_index(name, idx)
        return meta

    def promote(self, name: str, version: int, stage: Stage) -> ModelMetadata:
        """Move a version to a new stage.

        Demotes any prior holder of that stage to `archived` (for production/staging).
        """
        idx = self._read_index(name)
        if stage in ("production", "staging"):
            prior = idx["stages"].get(stage)
            if prior is not None and prior != version:
                self._set_meta_stage(name, prior, "archived")
        idx["stages"][stage] = version
        self._write_index(name, idx)
        return self._set_meta_stage(name, version, stage)

    def _set_meta_stage(self, name: str, version: int, stage: Stage) -> ModelMetadata:
        meta_path = self._version_dir(name, version) / "metadata.json"
        meta = ModelMetadata.from_dict(json.loads(meta_path.read_text()))
        meta.stage = stage
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2))
        return meta

    def load(self, name: str, stage: Stage = "production") -> tuple[Any, ModelMetadata]:
        """Load the model currently pinned to a stage."""
        idx = self._read_index(name)
        version = idx["stages"].get(stage)
        if version is None:
            raise LookupError(f"No model '{name}' in stage '{stage}'")
        return self.load_version(name, version)

    def load_version(self, name: str, version: int) -> tuple[Any, ModelMetadata]:
        vdir = self._version_dir(name, version)
        model = joblib.load(vdir / "model.joblib")
        meta = ModelMetadata.from_dict(json.loads((vdir / "metadata.json").read_text()))
        return model, meta

    def list_versions(self, name: str) -> list[ModelMetadata]:
        mdir = self._model_dir(name)
        out: list[ModelMetadata] = []
        for vdir in sorted(mdir.glob("v*"), key=lambda p: int(p.name[1:])):
            meta_file = vdir / "metadata.json"
            if meta_file.exists():
                out.append(ModelMetadata.from_dict(json.loads(meta_file.read_text())))
        return out

    def rollback(self, name: str) -> ModelMetadata:
        """Promote the most recently archived version back to production.

        This mirrors what we'd do on a production incident: revert to the last
        known-good model. Archival history gives us the candidate.
        """
        versions = [v for v in self.list_versions(name) if v.stage == "archived"]
        if not versions:
            raise RuntimeError(f"No archived version to roll back to for {name}")
        last_good = max(versions, key=lambda m: m.version)
        return self.promote(name, last_good.version, "production")


def data_hash(df_bytes: bytes) -> str:
    """SHA256 hash of a bytes representation of the training data.

    Stable hash makes "was this trained on the same data?" a trivial check.
    """
    return hashlib.sha256(df_bytes).hexdigest()[:16]
