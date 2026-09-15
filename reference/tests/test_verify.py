"""The Python verifier accepts real Rust-generated proofs, and refuses what Rust refuses.

Run:  python3 reference/tests/test_verify.py

Passes iff ``pyvega.verify.verify`` accepts ``proof.bin`` against ``vk.bin`` for the
``cubic`` fixture (no shared segment) and the ``shared_cubic`` fixture (x in the
shared segment): every per-instance and per-round challenge re-derivation matches,
the 6 pinned public values equal the native recomputation, relaxed Spartan checks
pass, and the final Hyrax PCS opening verifies. The returned public values must
match meta.json.

It also passes only if each proof is refused when the shared commitment is not
carried exactly once (a copy in a step or core instance, another commitment in a
step, or the proof's own copy removed, moved or extra), as the Rust verifier
refuses them with ``InvalidSharedCommitment``.
"""

import json
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from pyvega.vk import load_vk  # noqa: E402
from pyvega.proof import load_proof  # noqa: E402
from pyvega.verify import verify  # noqa: E402
from pyvega.field import scalar_to_repr  # noqa: E402

FIXTURES = os.path.join(ROOT, "reference", "fixtures")


def with_step(proof, i, comm):
  """The proof with step instance ``i`` carrying ``comm`` as its shared commitment."""
  steps = [replace(u, comm_W_shared=comm) if j == i else u for j, u in enumerate(proof.step_instances)]
  return replace(proof, step_instances=steps)


def with_every_instance(proof, comm):
  """The proof with every step instance and the core carrying ``comm``."""
  return replace(
    proof,
    step_instances=[replace(u, comm_W_shared=comm) for u in proof.step_instances],
    core_instance=replace(proof.core_instance, comm_W_shared=comm),
  )


def assert_refused(what, proof, vk):
  """Assert that verification refuses ``proof`` for its shared commitment."""
  try:
    verify(proof, vk, vk.num_steps)
  except ValueError as e:
    if "shared commitment" not in str(e):
      raise AssertionError(f"{what}: refused for another reason: {e}") from e
    print(f"  refused: {what} ({e})")
    return
  raise AssertionError(f"{what}: accepted")


def check_fixture(name):
  """Verify the fixture's Rust proof, compare public values, and return ``(proof, vk)``."""
  fix = os.path.join(FIXTURES, name)
  vk_bytes = open(os.path.join(fix, "vk.bin"), "rb").read()
  proof_bytes = open(os.path.join(fix, "proof.bin"), "rb").read()
  meta = json.load(open(os.path.join(fix, "meta.json")))

  t0 = time.time()
  vk = load_vk(vk_bytes)
  proof = load_proof(proof_bytes)
  print(f"[{name}] parsed vk ({len(vk_bytes)} B) + proof ({len(proof_bytes)} B) in {time.time()-t0:.1f}s")

  t1 = time.time()
  pv_step, pv_core = verify(proof, vk, vk.num_steps)
  print(f"[{name}] verify() accepted in {time.time()-t1:.1f}s")

  # Compare returned public values against meta.json (32-byte LE hex per scalar).
  def hexify(scalars):
    return [scalar_to_repr(s).hex() for s in scalars]

  got_step = [hexify(pv) for pv in pv_step]
  got_core = hexify(pv_core)
  assert got_step == meta["public_values_step"], (got_step, meta["public_values_step"])
  assert got_core == meta["public_values_core"], (got_core, meta["public_values_core"])

  print(f"[{name}] public_values_step: {got_step}")
  print(f"[{name}] public_values_core: {got_core}")
  return proof, vk


def main():
  proof, vk = check_fixture("cubic")
  assert vk.S_step.num_shared == 0 and proof.comm_W_shared is None
  other = proof.step_instances[0].comm_W_rest
  for i in range(len(proof.step_instances)):
    assert_refused(f"cubic: a commitment in step {i}", with_step(proof, i, other), vk)
  assert_refused(
    "cubic: a commitment in the core",
    replace(proof, core_instance=replace(proof.core_instance, comm_W_shared=other)),
    vk,
  )
  assert_refused("cubic: a commitment as the proof's copy", replace(proof, comm_W_shared=other), vk)

  proof, vk = check_fixture("shared_cubic")
  shared = proof.comm_W_shared
  assert vk.S_step.num_shared > 0 and shared is not None
  other = proof.step_instances[0].comm_W_rest
  for i in range(len(proof.step_instances)):
    assert_refused(f"shared_cubic: its copy in step {i}", with_step(proof, i, shared), vk)
  assert_refused(
    "shared_cubic: its copy in the core",
    replace(proof, core_instance=replace(proof.core_instance, comm_W_shared=shared)),
    vk,
  )
  assert_refused("shared_cubic: its copy in every instance", with_every_instance(proof, shared), vk)
  assert_refused("shared_cubic: another commitment in step 1", with_step(proof, 1, other), vk)
  assert_refused("shared_cubic: the proof's copy removed", replace(proof, comm_W_shared=None), vk)
  assert_refused(
    "shared_cubic: the proof's copy moved into every instance",
    replace(with_every_instance(proof, shared), comm_W_shared=None),
    vk,
  )

  print("\nPASS: Python verifier accepted the Rust proofs and refused every shared commitment not carried once")


if __name__ == "__main__":
  main()
