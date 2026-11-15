"""Layer 1 IP/OP Validation Suite: Code Model, 7-Stage Noise, Measurement Imperfections & Packet Codec."""
from __future__ import annotations

import numpy as np

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder
from simulation.stimulus.packet_codec import pack_packet_u64, unpack_packet_u64


def test_layer1_ip_to_op_end_to_end():
    # IP 1: Code Geometry (Rotated d=3 CSS)
    code = build_rotated_d3_code()
    assert code.hx.shape == (4, 9) and code.hz.shape == (4, 9)
    assert np.all(((code.hx @ code.hz.T) % 2) == 0)

    # IP 2: 16-PD Registry & Active Generators
    pd_reg = PDRegistry().load()
    assert len(pd_reg.list_profiles()) == 16
    gen_reg = GeneratorRegistry("configs/noise_framework/generator_registry.yaml", master_seed=42)
    gens = gen_reg.instantiate_all(master_seed=42)

    # IP 3: 100s Persistent PD Assignment & 10 Hz Synchronized Sampling
    assigner = PDAssignment(
        pd_registry=pd_reg,
        master_seed=42,
        policy_name="curriculum_burst_enriched",
        assignment_config_path="configs/noise_framework/generator_assignment.yaml",
    )
    sched = AssignmentScheduler(pd_assignment=assigner, window_duration_sec=100.0)
    sched.initialize(generators=gens, pd_registry=pd_reg)
    sampler = SynchronizedSampler(
        generators=gens,
        clock=SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0),
        scheduler=sched,
        composer=NoiseComposer(),
    )
    samples = sampler.generate_trajectory(duration_seconds=5.0)
    assert len(samples) == 50

    # OP 1: Physical Faults + Ideal Syndromes + Measurement-Noisy Syndromes + Validity Masks
    enc = SampleEncoder(rounds_per_composite_sample=1)
    traj = enc.encode_composite_trajectory(samples, master_seed=42, trajectory_id="l1_check")
    assert traj["syndromes_x"].shape == (50, 4)
    assert traj["syndromes_z"].shape == (50, 4)
    assert traj["validity_mask_x"].shape == (50, 4)
    assert traj["validity_mask_z"].shape == (50, 4)

    # OP 2: 64-bit AVID-SYNDROME-PACKET-V1 Binary Roundtrip
    sx_bits = traj["syndromes_x"][0]
    sz_bits = traj["syndromes_z"][0]
    sx_nib = int(sx_bits[0] | (sx_bits[1] << 1) | (sx_bits[2] << 2) | (sx_bits[3] << 3))
    sz_nib = int(sz_bits[0] | (sz_bits[1] << 1) | (sz_bits[2] << 2) | (sz_bits[3] << 3))
    word = pack_packet_u64(s_x=sx_nib, s_z=sz_nib, seq_id=0, epoch_id=1, flags=1)
    pkt = unpack_packet_u64(word)
    assert pkt.s_x == sx_nib and pkt.s_z == sz_nib and pkt.seq_id == 0
