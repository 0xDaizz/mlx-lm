"""Distributed generate test for Kimi K2.5 using sharded_load.

ALL ranks must call stream_generate — TP requires every rank to
participate in the forward pass (collective all_reduce).  Only rank 0
prints the result.
"""
import sys
import time

sys.path.insert(0, "/Users/hw/mlx-lm-server")

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.utils import sharded_load

MODEL_PATH = "/Users/hw/mlx-lm-server/models/Kimi-K2.5"
PROMPT = "What is 2+2? Answer in one sentence."

group = mx.distributed.init()
rank = group.rank()
world = group.size()

print(f"[Rank {rank}/{world}] Loading model with sharded_load...")
t0 = time.time()

try:
    mx.reset_peak_memory()
except AttributeError:
    pass

model, tokenizer = sharded_load(
    MODEL_PATH,
    tensor_group=group,
)

load_time = time.time() - t0
try:
    active = mx.get_active_memory() / (1024**3)
    peak = mx.get_peak_memory() / (1024**3)
    print(f"[Rank {rank}/{world}] Loaded in {load_time:.1f}s, active={active:.1f} GB, peak={peak:.1f} GB")
except AttributeError:
    print(f"[Rank {rank}/{world}] Loaded in {load_time:.1f}s")

# ALL ranks must participate in generation (TP collective ops)
print(f"[Rank {rank}/{world}] Starting generation...")
t1 = time.time()
response_text = ""
for resp in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    max_tokens=64,
):
    response_text = resp.text

gen_time = time.time() - t1

if rank == 0:
    print(f"[Rank {rank}/{world}] Response: {response_text}")
    print(f"[Rank {rank}/{world}] Generation time: {gen_time:.1f}s")

print(f"[Rank {rank}/{world}] Test PASSED")
