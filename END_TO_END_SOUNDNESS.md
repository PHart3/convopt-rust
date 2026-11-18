# Handoff: end-to-end soundness of the returned solution

## Problem

The soundness lemmas in `src/feas_spec.rs` (`lemma_approx_sound_reduced` at ~1631,
`lemma_approx_sound_unreduced` at ~1380) hold for **an arbitrary** `z`. Nothing applies
them to the iterate the solver actually returns:

- `sdpad_loop` (`src/sdp_altdir.rs:871`) ensures `outcome_ok` (`src/sdpad_spec.rs:883`),
  which is stated only about the **canonical** (standard-form) problem.
- No lemma takes `outcome_ok` as a hypothesis, and none calls the soundness lemmas on
  `o.states@[o.iterate@].prim_z`.
- `pub fn sdpad` (`src/sdp_altdir.rs:1057`), the user's entry point, lies **outside
  `verus!`**. It has no contract, and nothing requires `sdpad_input_ok` from its caller.

## Goal

### 1. Composition lemma (`src/feas_spec.rs` or a new module after it)

```rust
pub proof fn lemma_outcome_sound(sdp: &SDP, check: bool, s: &SdpadSetup, o: SdpadOutcome)
    requires
        sdpad_input_ok(sdp, check),
        sdpad_setup_ok(sdp, check, *s),
        outcome_ok(sdp, s, check, o),
        sdp.lmi@.len() >= 1,
    ensures
        user_outcome_ok(sdp, check, s, o),
```

Define `user_outcome_ok` as a new `open spec fn` whose clauses mention **`o.result@`,
`o.pinf`, `o.value`** directly, never a free `z`. Let `xs = Seq::new(o.result@.len(), |n| rvs(o.result@[n]@))`,
`bd = bdims_r(sdp.lmi@)`, `vd = vdims_r(sdp.lmi@, n_vars(sdp))`, and let `N` be a bound
on the user-level residuals:

- `check == true`:
  `N = setup_red_const(sdp, s) * rv(TOL) * setup_resid_scale(sdp, s, z)
       + setup_recovery_norm(sdp, s) * (rv(o.pinf) * (1 + norm_r(rvs(s.point@))))`,
  where `z = rvs(o.states@[o.iterate@].prim_z@)`. Keep `z` only inside the
  scale term `setup_resid_scale`, or rewrite that term in terms of `xs` if that's easy.
- `check == false`: `N = rv(o.pinf) * (1 + norm_r(rvs(s.point@)))`, i.e. the norm of the
  full residual `A(z) - b`. You must prove that each entry of `action_residual_r(s, z)` is at
  most this norm (relate `action_residual_r` to `setup_action_r(s, z) - point` and use
  `lemma_entry_le_norm`).

Clauses (both cases):

1. `o.result@.len() == vd.len()` and `xs == vars_of(z, block_sum(bd), vd)`. This is the
   bridge: `solution_ok` uses `dims_of(s.symm_dims@)` / `s.block_size_sum`, and the
   lemmas use `vdims_r` / `block_sum(bdims_r)`. Get the equation from `sdpad_setup_ok`.
2. Every equality row: `|user_row_r(sdp.constraint.0@, vd, e, xs) - rv(sdp.constraint.1@[e])| <= N`.
3. Every LMI `i`: `frob_rc(lmi_link_residual(sdp, z, i), bd[i], bd[i]) <= N`.
4. `rv(o.value) == user_obj_r(sdp.objective@, vd, xs)`.
5. If `o.termination` is `Accuracy` or `Stagnation`: for every LMI `i`,
   `psd_shifted(lmi_slack_r(sdp, z, i), bd[i], psd_check_eps_r(rv(PSD_TOL), z, block_off(bd, i), bd[i]) + frob_rc(lmi_link_residual(sdp, z, i), bd[i], bd[i]))`.
   Use `psd_blocks_ok` from `outcome_ok`, then `lemma_lmi_blocks_checked`, then
   `lemma_lmi_slack_shifted`.
6. If `o.termination` is `Accuracy`: also `rv(o.pinf) < rv(TOL)`, which follows from `delta < TOL`.
7. If `o.termination` is `MaxIter`: assert nothing beyond clauses 1–4. Don't strengthen
   it; that case carries no accuracy guarantee.

If `lmi_slack_r` / `lmi_link_residual` can be restated as functions of `xs` (plus the
slack block of `z`), do so in the spec so the statement reads at the user's level.
Otherwise leave them in terms of `z` and add a one-line doc comment saying so.

**LMI-free case** (`sdp.lmi@.len() == 0`): only the exact lemma
`lemma_action_feasible_sound_single` exists. Add the approximate analogue
(`|row residual| <= N`, objective equality, and `psd_shifted(sym_view(z), d, eps)` from
`psd_blocks_ok`), then fold it into `lemma_outcome_sound` by dropping the
`lmi.len() >= 1` precondition. Do this second, after the LMI case verifies.

### 2. Verified entry point (`src/sdp_altdir.rs`)

Inside `verus!`, add

```rust
pub fn sdpad_verified(sdp: &SDP, check: bool, var: &JacobiVariant) -> (o: SdpadOutcome)
    requires sdpad_input_ok(sdp, check),
    ensures  user_outcome_ok(sdp, check, &s_ghost, o)   // s_ghost: the setup; expose it via a ghost field or an exists
```

Its body is `sdpad_setup` + `sdpad_loop` + `proof { lemma_outcome_sound(...) }`. Choose
how to expose the setup in the postcondition: either `exists|s| sdpad_setup_ok(sdp, check, s) && user_outcome_ok(...)`,
or a ghost field on `SdpadOutcome`. Prefer the `exists` form only if it doesn't hurt
the triggers.

Then make the existing unverified `sdpad` a thin wrapper around `sdpad_verified` that
only prints, keeping its current signature `(Vec<SymMatrix>, f64)`.

## Out of scope

- A user-level **dual** statement. The user problem's dual is never defined, and
  `dinf` and the gap stay canonical-only. Don't add them.
- Bounding `setup_red_const` or `setup_recovery_norm`.
- Changes to executable behaviour or performance. The executable diff must be limited
  to the `sdpad` wrapper.

## Done when

- `lemma_outcome_sound` and `sdpad_verified` verify with no new `assume`, `admit`,
  `external_body`, or trusted axioms (check against `TRUSTED_BASE.md`).
- Add a short entry to `VERIFICATION_PROGRESS.md` giving the final statement of
  `user_outcome_ok`.
