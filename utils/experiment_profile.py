from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from utils.artifact_registry import ArtifactRegistry
from utils.artifact_roots import ArtifactRoots


_TOKEN_PATTERN = re.compile(r"^\$\{(?P<kind>artifact|artifact_parent):(?P<name>[A-Za-z0-9_.-]+)\}$")


@dataclass(frozen=True)
class ExperimentCommand:
    module: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentProfile:
    path: Path
    payload: Mapping[str, Any]
    profile_id: str
    schema_name: str
    artifacts: Mapping[str, str]
    commands: Mapping[str, ExperimentCommand]
    profile_sha256: str

    def command(self, stage: str) -> ExperimentCommand:
        try:
            return self.commands[stage]
        except KeyError as error:
            raise KeyError(f"profile does not define a {stage!r} command: {self.profile_id}") from error


def load_experiment_profile(path: str | Path) -> ExperimentProfile:
    profile_path = Path(path).expanduser().resolve()
    with profile_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"experiment profile must be a JSON object: {profile_path}")

    profile_id = _required_string(payload, "id", profile_path)
    schema_name = _required_string(payload, "schema_name", profile_path)
    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, dict) or not artifacts_raw:
        raise ValueError(f"profile artifacts must be a non-empty object: {profile_path}")
    artifacts: dict[str, str] = {}
    for name, artifact_id in artifacts_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"profile artifact key must be a non-empty string: {profile_path}")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"profile artifact id must be a non-empty string: {profile_path}")
        artifacts[name] = artifact_id

    commands_raw = payload.get("commands", {})
    if not isinstance(commands_raw, dict):
        raise ValueError(f"profile commands must be an object: {profile_path}")
    commands = {stage: _parse_command(stage, command, profile_path) for stage, command in commands_raw.items()}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ExperimentProfile(
        path=profile_path,
        payload=payload,
        profile_id=profile_id,
        schema_name=schema_name,
        artifacts=artifacts,
        commands=commands,
        profile_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def resolve_profile_artifacts(
    profile: ExperimentProfile,
    registry: ArtifactRegistry,
    roots: ArtifactRoots,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, artifact_id in profile.artifacts.items():
        record = registry.get(artifact_id)
        if record.schema_name not in {None, profile.schema_name}:
            raise ValueError(
                f"profile schema mismatch for {name}: {record.schema_name} != {profile.schema_name}"
            )
        resolved[name] = record.resolve(roots)
    return resolved


def render_command_args(
    command: ExperimentCommand,
    artifact_paths: Mapping[str, Path],
) -> list[str]:
    result: list[str] = []
    for value in command.args:
        match = _TOKEN_PATTERN.fullmatch(value)
        if match is None:
            result.append(value)
            continue
        name = match.group("name")
        try:
            path = artifact_paths[name]
        except KeyError as error:
            raise KeyError(f"profile token references unknown artifact key: {name}") from error
        result.append(str(path.parent if match.group("kind") == "artifact_parent" else path))
    return result


def _parse_command(stage: str, payload: Any, profile_path: Path) -> ExperimentCommand:
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"profile command stage must be a non-empty string: {profile_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"profile command must be an object: {stage}")
    module = _required_string(payload, "module", profile_path)
    args = payload.get("args", [])
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise ValueError(f"profile command args must be a string list: {stage}")
    return ExperimentCommand(module=module, args=tuple(args))


def _required_string(payload: Mapping[str, Any], field_name: str, profile_path: Path) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"profile requires a non-empty {field_name}: {profile_path}")
    return value.strip()
