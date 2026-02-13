"""Minimal distributed connectivity test."""
import mlx.core as mx

group = mx.distributed.init()

rank = group.rank()
world = group.size()

# Simple all_sum test
data = mx.array([rank + 1.0])
mx.eval(data)
result = mx.distributed.all_sum(data, group=group)
mx.eval(result)

expected = world * (world + 1) / 2
print(f"[Rank {rank}/{world}] all_sum result: {result.item()} (expected: {expected})")

# Memory info
try:
    active = mx.get_active_memory()
    print(f"[Rank {rank}/{world}] Active memory: {active / (1024**3):.2f} GB")
except AttributeError:
    pass

print(f"[Rank {rank}/{world}] Distributed test PASSED")
