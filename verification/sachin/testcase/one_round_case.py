"""One-round test case from the repo's seven-stage dataset, decoded by the repo's bit-exact BP reference.
Usage: PYTHONPATH=<repo> python3 one_round_case.py <repo> <round_id>"""
import sys, json, struct, time, numpy as np
repo=sys.argv[1]; r=int(sys.argv[2]); sys.path.insert(0,repo)
from simulation.reference.syndrome_stream.causal_stream_runner import CausalStreamRunner
from simulation.stimulus.packet_codec import unpack_packet_u64
tid="test_traj_00_s30000"; d=repo+"/datasets/seven_stage_v1/test/"+tid
z=np.load(d+".npz"); meta=json.load(open(d+".meta.json"))
raw=open(d+".bin","rb").read()
# file layout seen: 16-byte header then little-endian u64 packets (2000*8 = 16000)
hdr=len(raw)-2000*8; words=np.frombuffer(raw[hdr:],dtype="<u8")
pkt=unpack_packet_u64(int(words[r]))
run=CausalStreamRunner(K=3)
w=run.weight_bank.active_mem.copy()
smask=int(pkt.s_x)
tr=run.bp.decode(smask,w,K=3,true_error=z["fault_masks_z"][r])
err=z["fault_masks_z"][r]
out=dict(trajectory_id=tid,master_seed=meta["master_seed"],split=meta["split_id"],round_id=r,
 sample_id=int(z["sample_ids"][r]),packet_hex="0x%016x"%int(words[r]),packet_seq_id=pkt.seq_id,
 s_x_packet=pkt.s_x,s_x_npz_observed=int("".join(map(str,z["syndromes_x"][r][::-1])),2),
 s_x_npz_ideal=int("".join(map(str,z["ideal_syndromes_x"][r][::-1])),2),
 true_z_error_qubits=[i for i in range(9) if err[i]],logical_error_z_label=int(z["logical_error_z"][r]),
 correction_mask=tr.correction_mask,correction_qubits=[i for i in range(9) if (tr.correction_mask>>i)&1],
 residual_syndrome_zero=bool(tr.residual_syndrome_zero),logical_error_scored=bool(tr.logical_error),
 bp_core_cycles=tr.cycles_fused,latency_ns=tr.latency_ns_fused)
print(json.dumps(out,indent=1))
