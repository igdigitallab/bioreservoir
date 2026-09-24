"""Population membership rules for experiment 001 (Fly Oracle).

Rules live in `experiments/001-fly-oracle/populations.yaml`, one entry per population, each
naming a selection rule per dataset written against that dataset's OWN raw annotation vocabulary
(MaleCNS: `type`/`class`/`subclass`/`somaSide`/`receptorType`/`super_class`; BANC: `cell_type`/
`cell_class`/`cell_sub_class`/`cell_function`/`cell_function_detailed`/`side`/`super_class`) —
see the YAML file's header comment for why this module does not use the common schema
(`schema.py`) directly. `super_class` is exposed under the same column name for both datasets
(MaleCNS's own raw name is `superclass`, no underscore, renamed here to match BANC's `super_class`
and the common schema) since it is a broad category (sensory/intrinsic/descending/motor/...) with
per-dataset value spellings anyway, same pattern as `side`/`somaSide`.

Counts are always computed against the harmonized, proofread-only neuron set returned by
`harmonize.load_neurons()`, so a count answers "how many neurons of this kind are in the graph
the experiment actually runs on", not "how many exist in the raw unfiltered table".
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import yaml
from pyarrow import csv, feather

from . import banc, harmonize, malecns

REPO_ROOT = Path(__file__).resolve().parents[3]
POPULATIONS_YAML = REPO_ROOT / "experiments" / "001-fly-oracle" / "populations.yaml"

# Requested in this order; MaleCNS's feather reader does not preserve requested column order
# (see malecns.py's own note on this), so callers must `.select()` before renaming positionally.
# `rootSide` is read only to fill `somaSide`'s nulls (see `_malecns_side_with_root_fallback`)
# and dropped again before the final select/rename below — it is not part of the exposed schema.
MALECNS_RAW_COLUMNS = [
    "bodyId",
    "type",
    "class",
    "subclass",
    "somaSide",
    "rootSide",
    "receptorType",
    "superclass",
    "status",
]
MALECNS_RENAMED_COLUMNS = [
    "neuron_id",
    "type",
    "class",
    "subclass",
    "somaSide",
    "receptorType",
    "super_class",
]
# Raw-name subset matching MALECNS_RENAMED_COLUMNS 1:1 (drops "rootSide" and "status", which
# exist only to compute the merged `somaSide` and to filter Traced, respectively).
MALECNS_KEPT_RAW_COLUMNS = [
    "bodyId",
    "type",
    "class",
    "subclass",
    "somaSide",
    "receptorType",
    "superclass",
]

# pyarrow's CSV reader does preserve `include_columns` order (unlike feather), so this doubles
# as both the read order and the rename order.
BANC_RAW_COLUMNS = [
    "pt_root_id",
    "cell_type",
    "cell_class",
    "cell_sub_class",
    "cell_function",
    "cell_function_detailed",
    "side",
    "super_class",
]
BANC_RENAMED_COLUMNS = [
    "neuron_id",
    "cell_type",
    "cell_class",
    "cell_sub_class",
    "cell_function",
    "cell_function_detailed",
    "side",
    "super_class",
]

VALID_MODES = ("exact", "startswith")


@dataclass(frozen=True)
class Condition:
    """One AND-ed clause of a population rule (see populations.yaml header comment)."""

    column: str
    mode: str
    values: tuple[str, ...]

    def mask(self, table: pa.Table) -> pa.Array:
        col = table.column(self.column)
        if self.mode == "exact":
            return pc.is_in(col, value_set=pa.array(self.values, type=pa.string()))
        if self.mode == "startswith":
            if len(self.values) != 1:
                raise ValueError(f"startswith needs exactly one value, got {self.values!r}")
            return pc.starts_with(pc.fill_null(col, ""), self.values[0])
        raise ValueError(f"unknown condition mode: {self.mode!r} (expected one of {VALID_MODES})")

    def describe(self) -> str:
        op = "==" if self.mode == "exact" else "startswith"
        rhs = self.values[0] if len(self.values) == 1 else list(self.values)
        return f"{self.column} {op} {rhs!r}"


@dataclass(frozen=True)
class PopulationRule:
    """One population's rule for one dataset: either `conditions` or `absent` + `reason`."""

    name: str
    role: str
    description: str
    dataset: str
    conditions: tuple[Condition, ...] | None
    absent: bool
    reason: str | None

    def mask(self, table: pa.Table) -> pa.Array:
        if self.absent or not self.conditions:
            raise ValueError(f"population {self.name!r} is absent in dataset {self.dataset!r}")
        combined = self.conditions[0].mask(table)
        for condition in self.conditions[1:]:
            combined = pc.and_(combined, condition.mask(table))
        return combined

    def describe(self) -> str:
        if self.absent:
            return f"ABSENT: {self.reason}"
        return " AND ".join(c.describe() for c in self.conditions)


def load_rules() -> dict[str, dict[str, PopulationRule]]:
    """Parse populations.yaml into `{population_name: {dataset_name: PopulationRule}}`."""
    raw = yaml.safe_load(POPULATIONS_YAML.read_text())
    rules: dict[str, dict[str, PopulationRule]] = {}
    for name, spec in raw["populations"].items():
        per_dataset: dict[str, PopulationRule] = {}
        for dataset, dataset_spec in spec["datasets"].items():
            if dataset_spec.get("absent"):
                per_dataset[dataset] = PopulationRule(
                    name=name,
                    role=spec["role"],
                    description=spec["description"].strip(),
                    dataset=dataset,
                    conditions=None,
                    absent=True,
                    reason=dataset_spec["reason"].strip(),
                )
                continue
            conditions = tuple(
                Condition(
                    column=c["column"],
                    mode=c["mode"],
                    values=tuple(str(v) for v in c["values"]),
                )
                for c in dataset_spec["conditions"]
            )
            per_dataset[dataset] = PopulationRule(
                name=name,
                role=spec["role"],
                description=spec["description"].strip(),
                dataset=dataset,
                conditions=conditions,
                absent=False,
                reason=None,
            )
        rules[name] = per_dataset
    return rules


def _malecns_side_with_root_fallback(table: pa.Table) -> pa.Array:
    """MaleCNS's `somaSide` (L/R/M) is null for ~9.7% of Traced neurons (16,080/165,122,
    checked 2026-09-18) — almost entirely primary sensory neurons, since a soma side is only
    meaningful for neurons with a soma inside the traced volume (2,639/2,639
    olfactory_receptor_neuron, 672/672 mechanosensory_johnstons_organ and 1,428/1,428
    gustatory_all rows all have `somaSide == null`). `rootSide` (the hemisphere a neuron's
    primary neurite enters on) fills 15,864 of those 16,080 instead, using the same L/R/M
    vocabulary plus its own extra value "unknown" (treated as null here, not a fourth side).
    Coalescing the two into one column here (once, for every rule in populations.yaml, all of
    which are written against `somaSide`) replaces the separate `rootSide` lookup that used to
    live only in `sim/bench.py`'s sanity checks and covered readouts but not sensory inputs.
    """
    soma_side = table.column("somaSide")
    root_side = pc.if_else(
        pc.equal(table.column("rootSide"), "unknown"),
        pa.scalar(None, type=pa.string()),
        table.column("rootSide"),
    )
    return pc.coalesce(soma_side, root_side)


@functools.cache
def raw_annotations(dataset: str) -> pa.Table:
    """Raw per-dataset annotation table with the extra columns population rules need.

    Restricted to the harmonized, proofread-only neuron set (`harmonize.load_neurons`) so
    population counts are always counts within the graph the experiment runs on.

    Cached per dataset (measured 2026-09-18: ~0.47 s/call for malecns, ~0.30 s/call for banc,
    re-reading the raw feather/CSV files from disk every time) — `sim/calibrate.py` calls this
    indirectly (via `sim/bench.population_indices`) up to ~10 times per simulated trial (one per
    named readout population), which was adding several seconds of pure I/O on top of every
    ~20-30 s Brian2 trial before this cache existed. Safe to cache: the underlying files do not
    change during a process's lifetime and this function has no side effects.
    """
    proofread_ids = harmonize.load_neurons(dataset)["neuron_id"]

    if dataset == "malecns":
        table = feather.read_table(malecns.BODY_ANNOTATIONS, columns=MALECNS_RAW_COLUMNS)
        table = table.select(MALECNS_RAW_COLUMNS)  # pin order, see MALECNS_RAW_COLUMNS comment
        table = table.filter(pc.equal(table["status"], "Traced"))
        side = _malecns_side_with_root_fallback(table)
        table = table.set_column(table.schema.get_field_index("somaSide"), "somaSide", side)
        table = table.select(MALECNS_KEPT_RAW_COLUMNS)  # drops rootSide + status, pins order
        table = table.rename_columns(MALECNS_RENAMED_COLUMNS)
    elif dataset == "banc":
        table = csv.read_csv(
            banc.CODEX_ANNOTATIONS,
            parse_options=csv.ParseOptions(delimiter="\t"),
            convert_options=csv.ConvertOptions(
                include_columns=BANC_RAW_COLUMNS,
                strings_can_be_null=True,
                null_values=["NA", ""],
            ),
        )
        table = table.rename_columns(BANC_RENAMED_COLUMNS)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    return table.filter(pc.is_in(table["neuron_id"], value_set=proofread_ids))


def count_population(rule: PopulationRule, table: pa.Table) -> int:
    """Neuron count for `rule` against an already-loaded `raw_annotations(rule.dataset)` table."""
    if rule.absent:
        return 0
    mask = rule.mask(table)
    return pc.sum(mask.cast(pa.int64())).as_py() or 0


@dataclass(frozen=True)
class PopulationReport:
    name: str
    role: str
    dataset: str
    count: int
    rule_summary: str


def report(dataset: str) -> list[PopulationReport]:
    """One row per population defined for `dataset`: name, role, count, and the rule applied."""
    rules = load_rules()
    table = raw_annotations(dataset)
    rows: list[PopulationReport] = []
    for name, per_dataset in rules.items():
        rule = per_dataset.get(dataset)
        if rule is None:
            continue
        rows.append(
            PopulationReport(
                name=name,
                role=rule.role,
                dataset=dataset,
                count=count_population(rule, table),
                rule_summary=rule.describe(),
            )
        )
    return rows
