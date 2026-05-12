"""ForgetEval — a benchmark for how well memory systems forget.

Five families of test:
  1. supersession    — newer fact replaces older; recall reflects newer
  2. decay           — info with TTL should fade from default recall
  3. amnesia         — selective forgetting doesn't leak; other memories survive
  4. purge           — content marked purged must not be reachable
  5. drift           — multiple updates on the same topic; the latest wins

Each test is a Python callable `(adapter) -> bool` (pass/fail), plus a
short `id` and `family`. The adapter implements minimal memory ops so
the same tests can run against Lethe, Mem0, MemPalace, Letta...
"""
