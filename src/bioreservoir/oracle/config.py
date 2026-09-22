"""Loader for `experiments/001-fly-oracle/config.yaml` (seeds, trial/input/readout parameters).

Every seed used anywhere in the oracle pipeline is read from here, never hard-coded at a call
site, so the whole pipeline's randomness is reproducible from one committed file (README.md
Pipeline step 5: "The protocol, seeds and question list are published before any run.").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "001-fly-oracle"
CONFIG_YAML = EXPERIMENT_DIR / "config.yaml"
QUESTIONS_YAML = EXPERIMENT_DIR / "questions.yaml"

GRAPH_VARIANTS = ("real", "rewired", "er")
TEXT_VARIANTS = ("original", "negation", "paraphrase")


@dataclass(frozen=True)
class Seeds:
    projection: int
    balance: int
    trial_base: int
    coin_base: int
    rewire_base: int
    er_base: int
    side_base: int


@dataclass(frozen=True)
class TrialConfig:
    duration_ms: float
    n_trials: int
    min_syn: int


@dataclass(frozen=True)
class InputConfig:
    population: str
    rate_hz_min: float
    rate_hz_max: float
    n_per_side: int | None = None


@dataclass(frozen=True)
class ReadoutConfig:
    name: str
    zero_spike_probability: float


@dataclass(frozen=True)
class ControlsConfig:
    graph_variants: tuple[str, ...]
    rewire_max_repair_rounds: int


@dataclass(frozen=True)
class ScoringConfig:
    cluster_by: str


@dataclass(frozen=True)
class WorkersConfig:
    default: int


@dataclass(frozen=True)
class HandednessConfig:
    reference_file: str  # relative to EXPERIMENT_DIR (oracle.reference.load_reference's default)

    def reference_path(self) -> Path:
        return EXPERIMENT_DIR / self.reference_file


@dataclass(frozen=True)
class OracleConfig:
    seed: Seeds
    trial: TrialConfig
    input: InputConfig
    readout: ReadoutConfig
    controls: ControlsConfig
    scoring: ScoringConfig
    workers: WorkersConfig
    handedness: HandednessConfig
    raw: dict  # the parsed YAML, verbatim, for hashing (see `config_hash`)

    def readout_population_left(self) -> str:
        return f"{self.readout.name}_left"

    def readout_population_right(self) -> str:
        return f"{self.readout.name}_right"


def load_config(path: Path = CONFIG_YAML) -> OracleConfig:
    raw = yaml.safe_load(path.read_text())
    return OracleConfig(
        seed=Seeds(**raw["seed"]),
        trial=TrialConfig(**raw["trial"]),
        input=InputConfig(**raw["input"]),
        readout=ReadoutConfig(**raw["readout"]),
        controls=ControlsConfig(
            graph_variants=tuple(raw["controls"]["graph_variants"]),
            rewire_max_repair_rounds=raw["controls"]["rewire_max_repair_rounds"],
        ),
        scoring=ScoringConfig(**raw["scoring"]),
        workers=WorkersConfig(**raw["workers"]),
        handedness=HandednessConfig(**raw["handedness"]),
        raw=raw,
    )


def config_hash(config: OracleConfig) -> str:
    """Stable hash of the config content that materially affects results.

    Used by the ledger to decide whether a previously-run (question, brain, condition, variant)
    combination is still valid to skip on resume, or stale because the config changed underneath
    it (oracle/ledger.py, oracle/plan.py).
    """
    import hashlib
    import json

    canonical = json.dumps(config.raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
