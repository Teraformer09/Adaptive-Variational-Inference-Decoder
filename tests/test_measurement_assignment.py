"""Step 6 Tests — Connect the PD assignment scheduler (`tests/test_measurement_assignment.py`)."""

from __future__ import annotations

from pathlib import Path

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
PD_YAML = REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml"
REG_YAML = REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml"
ASSIGN_YAML = REPO_ROOT / "configs" / "noise_framework" / "generator_assignment.yaml"


def _setup_scheduler(seed: int = 42) -> tuple[PDRegistry, GeneratorRegistry, PDAssignment, AssignmentScheduler]:
    pd_reg = PDRegistry(PD_YAML)
    gen_reg = GeneratorRegistry(REG_YAML, master_seed=seed)
    generators = gen_reg.instantiate_all(master_seed=seed)
    pd_assign = PDAssignment(
        pd_registry=pd_reg,
        master_seed=seed,
        policy_name="uniform_16",
        assignment_config_path=ASSIGN_YAML,
    )
    sched = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
    sched.initialize(generators=generators, pd_registry=pd_reg)
    return pd_reg, gen_reg, pd_assign, sched


def test_assignment_persists_within_window() -> None:
    _, gen_reg, _, sched = _setup_scheduler(seed=42)
    meas = gen_reg.get("measurement_noise_01")
    initial_pid = sched.assignment_for("measurement_noise_01")
    for ts in (0.0, 0.1, 25.0, 50.0, 99.8, 99.9):
        changed = sched.advance(ts)
        assert changed is False
        assert sched.current_window() == 0
        assert sched.assignment_for("measurement_noise_01") == initial_pid
        assert meas.current_profile is not None
        assert meas.current_profile.profile_id == initial_pid


def test_assignment_changes_at_boundary() -> None:
    _, gen_reg, _, sched = _setup_scheduler(seed=42)
    assert sched.advance(99.9) is False
    assert sched.current_window() == 0

    assert sched.advance(100.0) is True
    assert sched.current_window() == 1
    w1_pid = sched.assignment_for("measurement_noise_01")

    assert sched.advance(100.1) is False
    assert sched.current_window() == 1
    assert sched.assignment_for("measurement_noise_01") == w1_pid

    assert sched.advance(200.0) is True
    assert sched.current_window() == 2

    seen_pids = {
        sched.pd_assignment.assign("measurement_noise_01", w) for w in range(8)
    }
    assert len(seen_pids) > 1
    assert gen_reg.get("measurement_noise_01").current_window_id == 2


def test_each_generator_has_one_assignment_per_window() -> None:
    _, gen_reg, pd_assign, sched = _setup_scheduler(seed=77)
    for w in range(3):
        sched.advance(w * 100.0 + 0.0)
        sched.advance(w * 100.0 + 50.0)
        sched.advance(w * 100.0 + 99.9)

    manifest = pd_assign.assignment_manifest()
    n_gens = len(gen_reg.active_generators())
    assert manifest["total_assignments"] == n_gens * 3
    seen_keys = {(rec["generator_id"], rec["window_id"]) for rec in manifest["history"]}
    assert len(seen_keys) == n_gens * 3


def test_assignments_are_reproducible() -> None:
    _, _, a1, s1 = _setup_scheduler(seed=1234)
    _, _, a2, s2 = _setup_scheduler(seed=1234)
    for ts in (0.0, 100.0, 200.0):
        s1.advance(ts)
        s2.advance(ts)
    assert a1.assignment_manifest()["history"] == a2.assignment_manifest()["history"]


def test_generator_rng_streams_are_independent() -> None:
    pd_reg = PDRegistry(PD_YAML)
    a1 = PDAssignment(pd_registry=pd_reg, master_seed=99)
    a2 = PDAssignment(pd_registry=pd_reg, master_seed=99)

    seq_meas_only = [a1.assign("measurement_noise_01", w) for w in range(5)]

    seq_with_others = []
    for w in range(5):
        for idx in range(5):
            a2.assign(f"other_gen_{idx}", w)
        seq_with_others.append(a2.assign("measurement_noise_01", w))

    assert seq_meas_only == seq_with_others
