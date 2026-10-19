# Trusted Base

All trusted items live in `src/trusted.rs`. No other file contains `axiom`,
`assume_specification`, `external_body`, `assume`, or `admit` (checked by
grep at the final audit). Everything else in the crate is proved relative to
the items listed here.

The model: `rv(x)` is the exact real value denoted by an `f64`. Primitive
`f64` operations are abstracted as exact operations on SMT reals. Rounding,
overflow, NaN, signed zero, and infinities are not modeled. All verification
results are relative to this abstraction (CLAUDE.md, section 1).

Consistency note: in Verus's main SMT encoding `f64` is an uninterpreted
sort and `rv` is a fresh uninterpreted function, so the axioms below are
satisfiable (interpret `rv` as a bijection onto the real algebraic numbers,
and the result hooks as the exact operations). They are not connected to
Verus's IEEE `f64 as real` cast, which is only interpreted in `bit_vector`
mode and is never used in this project.

## Trusted items

| Id | Item | Statement | Justification |
|----|------|-----------|---------------|
| T1 | `axiom_sqrt_r` | For `x >= 0`: `sqrt_r(x) >= 0` and `sqrt_r(x)^2 == x`. | Existence of nonnegative square roots in the reals; not expressible in SMT real arithmetic without an uninterpreted symbol. `sqrt_r` is unspecified for `x < 0`. Uniqueness and monotonicity are proved in `src/real_spec.rs`. |
| T2 | `axiom_add_req`, `axiom_sub_req`, `axiom_mul_req`, `axiom_div_req` | The vstd operator preconditions `add_req`, `sub_req`, `mul_req`, `div_req` hold for all `f64` pairs. | vstd leaves these preconditions uninterpreted for floats. The primitive operators are total on `f64` (no panics); the abstraction does not restrict their domain. Division by zero is total, only its value is unspecified (T3). |
| T3 | `axiom_add`, `axiom_sub`, `axiom_mul`, `axiom_div` | If `o` is the result of `a + b` (resp. `-`, `*`, `/`) then `rv(o) == rv(a) + rv(b)` (resp. `-`, `*`, and `/` when `rv(b) != 0`). | The exact-real abstraction of primitive `f64` arithmetic. Stated on vstd's result hooks `add_ensures`, etc., so the executable code keeps its original operator syntax. Division by zero is left unspecified; see the division audit below. |
| T4 | `axiom_eq`, `axiom_ne`, `axiom_lt`, `axiom_le`, `axiom_gt`, `axiom_ge` | Executable `f64` comparisons return the corresponding exact-real comparison of `rv` values. | The exact-real abstraction of comparisons. Under IEEE this differs only on NaN (all comparisons false) and signed zeros (equal under both). Ghost comparisons of `f64` values are not interpreted; specifications compare `rv` values. |
| T5 | `axiom_literals` | Each `f64` literal used by the verified code (`0.0`, `0.5`, `1.0`, `1.6`, `2.0`, `5.0`, `1e4`, `1e-3`, `1e-4`, `1e-5`, `1e-8`, `1e-9`, `1e-10`, `1e-12`) has `rv` equal to its decimal value. | Literals are the entry points of the abstraction: the tolerance and tuning constants in the source are read as the exact decimals written. The list is exactly the set of literals occurring in the non-test sources (audited by grep). |
| T6 | `assume_specification` for `f64::abs`, `max`, `min`, `sqrt`, `mul_add`, `hypot`, `recip`, `signum`, `is_finite` | `abs`, `max`, `min` are exact. `sqrt(x) == sqrt_r(rv(x))` for `rv(x) >= 0`, else unspecified. `mul_add(x,a,b) == x*a + b`. `hypot(x,y) == sqrt_r(x^2 + y^2)`. `recip(x) == 1/x` for `x != 0`, else unspecified. `signum(x)` is `1` or `-1`, matching the sign of a nonzero `x`. `is_finite` is unconstrained. | vstd gives these std methods no specification. Each contract is the exact-real reading of the std documentation. `signum(0)` may return either sign (IEEE: depends on the sign bit), so the contract leaves it open. `is_finite` is a property of floats the exact-real model cannot express; its one use is a data-dependent abort. |
| T7 | `neg`, `pos_infinity` (`external_body`, `inline(always)`) | `rv(neg(x)) == -rv(x)`. `pos_infinity()` is unconstrained. | Verus rejects unary `-x` on floats and the std float constants. The wrappers are one-instruction inline functions whose bodies are the rejected expressions, so executable code and performance are unchanged. |
| T8 | `abort(msg) -> !` (`external_body`) | Diverges. No precondition, no postcondition. | The solver panics on data-dependent failures. vstd forbids `panic!` unless proved unreachable, which is impossible for these checks. `abort` makes such exits explicit: a function that aborts establishes nothing, and verified postconditions only cover normal returns. Reachability of aborts is deliberately not analyzed. |
| T9 | `assume_specification` for `usize::isqrt` | `r * r <= x < (r + 1) * (r + 1)` (integer arithmetic). | vstd gives `isqrt` no specification. The contract is the std documentation ("the square root rounded down"). |

The three `external_body` functions of T7 and T8 are not included in Verus's
verified-function count.

## Where the trusted items are used

**T1** (`sqrt_r`): the square-root lemmas of `src/real_spec.rs`; the norms
`norm_r`, `row_scale_r`, `offdiag_norm_r`, `jacobi_eps_r`; the `sqrt` and
`hypot` contracts of T6.

**T3, division by zero.** Every executable `f64` division of the verified
code and the fact that makes its divisor nonzero (so the quotient axiom
applies), or the reason an unspecified quotient is harmless:

| Division | Divisor is nonzero because |
|----------|----------------------------|
| `diag_solver`, `forw_subst`, `back_subst`: `vect[j] / d` | the preceding tolerance check establishes `\|rv(d)\| >= TOL * scale >= 1e-10` (else `abort`). |
| `ldlt_decomp`: division by `d[j]` | the pivot check of row `j` established `rv(d[j]) >= TOL * max(max_i \|A_ii\|, 1) >= 1e-10`. |
| `jacobi_rot`: `/ (2 a_jk)`, `/ (\|tau\| + sqrt(1 + tau^2))`, `recip(sqrt(1 + t^2))` | the nonzero-pivot precondition; square roots of arguments `>= 1` are `>= 1` (T1). |
| `gauss_elim`: multiplier `A_jn / A_rn` | the pivot is nonzero unless the whole remaining column is zero; then `A_jn == 0` and the annihilated entry is `0` for any value of the quotient, so the postconditions hold. |
| `linear_remove_redundant_sym`: row `/ scale` | the scale is the Frobenius norm of a pivot row, positive because the pivot has magnitude `>= eps > 0` (precondition `abs_max_r(input) > 0`). |
| `relative_residual_is_small`: `num / den` | the contract `dual_solve_ok` claims the residual bound only when `den != 0`. |
| `wgy_update`: `step / penalty` | precondition `rv(penalty) != 0` (the penalty stays in `[PEN_MIN, PEN_MAX]`). |
| `penalty_update`: `pinf / dinf`, `/ PEN_FACT` | the contract names the ratio only when `dinf != 0`; `PEN_FACT = 0.5`. |
| `kkt_quantities`: `/ (1 + \|\|b\|\|)`, `/ (1 + \|\|C_F\|\|_1)`, `/ (1 + \|dual\| + \|prim\|)` | each denominator is `>= 1` (`lemma_dot_self_nonneg` with T1, `lemma_sym_one_norm_nonneg`, `abs_r >= 0`). |

**T4**: the six comparison functions of `src/lin_alg/comparison.rs` and the
executable comparisons of the solvers, eigensolvers and the SDP loop.

**T5**: `TOL = 1e-10` (`matrix.rs`); `1e-9` (`psd_block_check`, `dual_solve`);
`1e-12` (both Jacobi stopping tests); `1e-8` (`negeigendecomp`); `0.5`
(`constraint_lin_comb`, `scale_off_diag`, `PEN_FACT`); `1.6` (`STEP`); `1e4`,
`1e-4` (`PEN_MAX`, `PEN_MIN`); `5.0`, `1e-3`, `1e-5` (`sdpad_loop` termination
and stagnation tests); `0.0`, `1.0`, `2.0` throughout.

**T6**: `abs`, `max` (pivot scales, stopping tests, residuals); `min`
(`penalty_update`); `sqrt` (`euclid_norm`, `euclid_distance`, `frob_norm_sym`,
`frob_dist_sym`, `jacobi_rot`, the factor `2.0.sqrt()` of
`linear_remove_redundant_sym`); `mul_add` (dot products, Frobenius products,
`ldlt_decomp`); `hypot` (the row-norm fold of `linear_remove_redundant_sym`);
`recip`, `signum` (`jacobi_rot`); `is_finite` (the abort guard of
`linear_remove_redundant_sym`).

**T7**: `neg` in `is_significantly_negative` and `negscal_vect`;
`pos_infinity` for the five bookkeeping sentinels of `sdpad_loop`
(`best_delta` and the previous residuals), whose contracts
(`stagnation_ok`, `best_feasible_ok`) hold for any sentinel value.

**T8**: the 19 `abort` calls, all data-dependent exits of the original code:
the three "division by zero" exits of the triangular and diagonal solvers; the
failed factorization in `pos_def_solver`; `row > col` in `sym_matrix_proj`;
inconsistency and a non-finite row norm in `linear_remove_redundant_sym`;
`index_max_in_col` on column 0 (unreachable under its precondition) and
non-convergence after `MAX_SWEEPS` sweeps in both eigensolvers; the failed
residual test of the regularized Gram solve in `dual_solve`; and the eight
input-shape checks of `sdp_to_standard` and its helpers.

**T9**: `lmi_dims` reads the block size `k` of an LMI from the length `k^2`
of its constant matrix; the executable check `k * k == len` that follows
aborts otherwise, so only the existence of the integer square root is used.

## Library specifications relied upon

vstd is used as a library and is not part of this project's trusted base.
Its contracts used by the executable code: `Vec`/slice indexing and
`index_mut` (including `RangeTo` on slices), `push`, `append`,
`extend_from_slice`, `clear`, `vec![x; n]`, `vstd::slice::slice_to_vec`
(vstd's `external_body` wrapper of `<[T]>::to_vec`), `usize::saturating_mul`,
`Option` methods, range iterators, and the `arithmetic::div_mod` lemmas.

## Unverified code

- `sdpad` (`src/sdp_altdir.rs`) is outside `verus!` because it prints the
  termination report (`println!` has no vstd contract). It only calls the
  verified `sdpad_setup` and `sdpad_loop` and must (informally) provide the
  precondition `sdpad_input_ok(sdp, check_constraints)`: the shape
  requirements of `sdp_to_standard` plus `dim_fits(total_rows(sdp))`.
- `src/main.rs` and the tests `src/sdp_tests.rs` are ordinary Rust.
