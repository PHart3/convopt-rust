# Trusted Base

All trusted items live in `src/trusted.rs`. No other file contains `axiom`,
`assume_specification`, `external_body`, `assume`, or `admit`. Everything
else in the crate is proved relative to the items below.

**Model.** `rv(x)` is the exact real value of an `f64`. Primitive `f64`
operations are modeled as exact operations on SMT reals. Rounding, overflow,
NaN, signed zero and infinities are not modeled.

**Consistency.** In Verus's SMT encoding `f64` is an uninterpreted sort and
`rv` a fresh uninterpreted function, so the axioms are satisfiable: interpret
`rv` as a bijection onto the real algebraic numbers and the result hooks as
the exact operations. They are unrelated to Verus's IEEE `f64 as real` cast,
which is interpreted only in `bit_vector` mode and never used here.

## Trusted items

| Id | Item | Statement | Justification |
|----|------|-----------|---------------|
| T1 | `axiom_sqrt_r` | For `x >= 0`: `sqrt_r(x) >= 0` and `sqrt_r(x)^2 == x`. | Nonnegative square roots exist in the reals but are not expressible in SMT real arithmetic. `sqrt_r` is unspecified for `x < 0`. Uniqueness and monotonicity are proved in `src/real_spec.rs`. |
| T2 | `axiom_add_req`, `axiom_sub_req`, `axiom_mul_req`, `axiom_div_req` | vstd's operator preconditions hold for all `f64` pairs. | vstd leaves them uninterpreted for floats. The operators are total on `f64`; division by zero only has an unspecified value (T3). |
| T3 | `axiom_add`, `axiom_sub`, `axiom_mul`, `axiom_div` | The result `o` of `a + b` (resp. `-`, `*`, `/`) has `rv(o) == rv(a) + rv(b)` (resp. `-`, `*`, and `/` when `rv(b) != 0`). | The exact-real arithmetic abstraction, stated on vstd's result hooks so the code keeps its operator syntax. |
| T4 | `axiom_eq`, `axiom_ne`, `axiom_lt`, `axiom_le`, `axiom_gt`, `axiom_ge` | Executable `f64` comparisons return the exact-real comparison of the `rv` values. | Differs from IEEE only on NaN and signed zeros. Ghost comparisons of `f64` are not covered; specifications compare `rv` values. |
| T5 | `axiom_literals` | Each literal used by the verified code (`0.0`, `0.5`, `1.0`, `1.6`, `2.0`, `5.0`, `1e4`, `1e-3`, `1e-4`, `1e-5`, `1e-8`, `1e-9`, `1e-10`, `1e-12`) denotes its decimal value. | Tolerances and tuning constants are read as the decimals written. The list is exactly the literals in the non-test sources. |
| T6 | `assume_specification` for `f64::abs`, `max`, `min`, `sqrt`, `mul_add`, `hypot`, `recip`, `signum`, `is_finite` | `abs`, `max`, `min` exact; `sqrt(x) == sqrt_r(x)` for `x >= 0`; `mul_add(x, a, b) == x*a + b`; `hypot(x, y) == sqrt_r(x^2 + y^2)`; `recip(x) == 1/x` for `x != 0`; `signum(x)` is `1` or `-1`, matching the sign of nonzero `x`; `is_finite` unconstrained. | vstd gives these methods no specification; each contract is the exact-real reading of the std documentation. `signum(0)` depends on the sign bit, so it is left open. `is_finite` is inexpressible in the model; its one use guards an abort. |
| T7 | `neg`, `pos_infinity` (`external_body`, `inline(always)`) | `rv(neg(x)) == -rv(x)`; `pos_infinity()` unconstrained. | Verus rejects unary `-x` on floats and the std float constants. The wrappers' bodies are exactly the rejected expressions. |
| T8 | `abort(msg) -> !` (`external_body`) | Diverges; no pre- or postcondition. | Replaces `panic!` on data-dependent failures, which vstd forbids unless proved unreachable. Postconditions cover normal returns only; reachability of aborts is not analyzed. |
| T9 | `assume_specification` for `usize::isqrt` | `r * r <= x < (r + 1) * (r + 1)`. | vstd gives `isqrt` no specification; the contract is the std documentation. |

The `external_body` functions of T7 and T8 are not in Verus's verified count.

## Where the trusted items are used

**T1**: the square-root lemmas of `real_spec.rs`; the norms `norm_r`,
`row_scale_r`, `offdiag_norm_r`, `jacobi_eps_r`; the `sqrt` and `hypot`
contracts of T6.

**T3, division.** Why each divisor is nonzero, or why an unspecified
quotient is harmless:

| Division | Reason |
|----------|--------|
| `diag_solver`, `forw_subst`, `back_subst`: `vect[j] / d` | the preceding tolerance check gives `\|rv(d)\| >= TOL * scale >= 1e-10`, else `abort` |
| `ldlt_decomp`: `/ d[j]` | the pivot check gives `rv(d[j]) >= TOL * max(max_i \|A_ii\|, 1) >= 1e-10` |
| `jacobi_rot`: `/ (2 a_jk)`, `/ (\|tau\| + sqrt(1 + tau^2))`, `recip(sqrt(1 + t^2))` | nonzero-pivot precondition; square roots of arguments `>= 1` are `>= 1` |
| `gauss_elim`: `A_jn / A_rn` | the pivot is nonzero unless the remaining column is zero, in which case `A_jn == 0` and the quotient's value does not matter |
| `linear_remove_redundant_sym`: `/ scale` | the Frobenius norm of a pivot row, positive since the pivot has magnitude `>= eps > 0` |
| `relative_residual_is_small`: `num / den` | `dual_solve_ok` claims the bound only when `den != 0` |
| `wgy_update`: `step / penalty` | precondition `rv(penalty) != 0` (penalty in `[PEN_MIN, PEN_MAX]`) |
| `penalty_update`: `pinf / dinf`, `/ PEN_FACT` | the contract uses the ratio only when `dinf != 0`; `PEN_FACT = 0.5` |
| `kkt_quantities`: `/ (1 + \|\|b\|\|)`, `/ (1 + \|\|C_F\|\|_1)`, `/ (1 + \|dual\| + \|prim\|)` | each denominator is `>= 1` |

**T4**: the comparison functions of `src/lin_alg/comparison.rs` and the
executable comparisons in the solvers, eigensolvers and SDP loop.

**T5**: `TOL = 1e-10` (`matrix.rs`); `1e-9` (`psd_block_check`,
`dual_solve`); `1e-12` (Jacobi stopping tests); `1e-8` (`negeigendecomp`);
`0.5` (`constraint_lin_comb`, `scale_off_diag`, `PEN_FACT`); `1.6` (`STEP`);
`5.0` (`PEN_INIT`); `1e4`, `1e-4` (`PEN_MAX`, `PEN_MIN`); `1e-3`, `1e-4`,
`1e-5` (stagnation tests of `sdpad_loop`); `0.0`, `1.0`, `2.0` throughout.

**T6**: `abs`, `max` (pivot scales, stopping tests, residuals); `min`
(`penalty_update`); `sqrt` (norms and distances, `jacobi_rot`, the factor
`2.0.sqrt()` of `linear_remove_redundant_sym`); `mul_add` (dot and Frobenius
products, `ldlt_decomp`); `hypot` (row norms of `linear_remove_redundant_sym`);
`recip`, `signum` (`jacobi_rot`); `is_finite` (an abort guard in
`linear_remove_redundant_sym`).

**T7**: `neg` in `is_significantly_negative` and `negscal_vect`;
`pos_infinity` for the five sentinels of `sdpad_loop` (`best_delta`,
`best_feas_delta`, `best_pinf`, `best_dinf`, `best_gap`: the recorded
quantities of the best candidate so far), whose contracts hold for any
sentinel value.

**T8**: 19 `abort` calls, all data-dependent exits of the original code:
- three "division by zero" exits of the triangular and diagonal solvers;
- the failed factorization in `pos_def_solver`;
- `row > col` in `sym_matrix_proj`;
- inconsistency and a non-finite row norm in `linear_remove_redundant_sym`;
- `index_max_in_col` on column 0 (unreachable under its precondition);
- non-convergence after `MAX_SWEEPS` sweeps in each eigensolver;
- the failed residual test of the regularized Gram solve in `dual_solve`;
- eight input-shape checks in `sdp_to_standard` and its helpers.

**T9**: `lmi_dims` reads an LMI's block size `k` from the length `k^2` of
its constant matrix; the following check `k * k == len` aborts otherwise.

## Library specifications relied upon

vstd is a library, not part of the project's trusted base. The code uses its
contracts for `Vec`/slice indexing and `index_mut` (including `RangeTo`),
`push`, `append`, `extend_from_slice`, `clear`, `vec![x; n]`,
`vstd::slice::slice_to_vec`, `usize::saturating_mul`, `Option` methods, range
iterators, and the `arithmetic::div_mod` lemmas.

## Unverified code

- `sdpad` (`src/sdp_altdir.rs`) is outside `verus!` because it prints the
  termination report (`println!` has no vstd contract). It only calls the
  verified `sdpad_setup` and `sdpad_loop`, and informally must satisfy
  `sdpad_input_ok(sdp, check_constraints)`.
- `src/main.rs` and `src/sdp_tests.rs` are ordinary Rust.
