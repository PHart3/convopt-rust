# Trusted Base

All trusted items live in `src/trusted.rs`. No other file may contain
`axiom`, `assume_specification`, `external_body`, `assume`, or `admit`.
Everything else in the crate is proved relative to the items listed here.

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

| Id | Item | Statement | Justification |
|----|------|-----------|---------------|
| T1 | `axiom_sqrt_r` | For `x >= 0`: `sqrt_r(x) >= 0` and `sqrt_r(x)^2 == x`. | Existence of nonnegative square roots in the reals; not expressible in SMT linear/nonlinear real arithmetic without an uninterpreted symbol. `sqrt_r` is unspecified for `x < 0`. Uniqueness and monotonicity are proved in `src/real_spec.rs`. |
| T2 | `axiom_add_req`, `axiom_sub_req`, `axiom_mul_req`, `axiom_div_req` | The vstd operator preconditions `add_req`, `sub_req`, `mul_req`, `div_req` hold for all `f64` pairs. | vstd leaves these preconditions uninterpreted for floats. The primitive operators are total on `f64` (no panics); the abstraction does not restrict their domain. Division by zero is covered: the operation is total, only its value is unspecified (T3). |
| T3 | `axiom_add`, `axiom_sub`, `axiom_mul`, `axiom_div` | If `o` is the result of `a + b` (resp. `-`, `*`, `/`) then `rv(o) == rv(a) + rv(b)` (resp. `-`, `*`, and `/` when `rv(b) != 0`). | The exact-real abstraction of primitive `f64` arithmetic (CLAUDE.md, section 1). Stated on vstd's result hooks `add_ensures`, etc., so the executable code keeps its original operator syntax. Division by zero is left unspecified. |
| T4 | `axiom_eq`, `axiom_ne`, `axiom_lt`, `axiom_le`, `axiom_gt`, `axiom_ge` | Executable `f64` comparisons return the corresponding exact-real comparison of `rv` values. | The exact-real abstraction of comparisons. Under IEEE this differs only on NaN (all comparisons false) and signed zeros (equal under both). |
| T5 | `axiom_literals` | Each `f64` literal used by the verified code (`0.0`, `0.5`, `1.0`, `1.6`, `2.0`, `5.0`, `1e3`, `1e4`, `1e-3`, `1e-4`, `1e-5`, `1e-6`, `1e-8`, `1e-9`, `1e-10`, `1e-12`) has `rv` equal to its decimal value. | Literals are the entry points of the abstraction: the tolerance and tuning constants in the source are read as the exact decimals written. Only literals that appear in the code are listed; add an entry when a new literal is verified. |
| T6 | `assume_specification` for `f64::abs`, `max`, `min`, `sqrt`, `mul_add`, `hypot`, `recip`, `signum`, `is_finite` | `abs`, `max`, `min` are exact. `sqrt(x) == sqrt_r(rv(x))` for `rv(x) >= 0`, else unspecified. `mul_add(x,a,b) == x*a + b`. `hypot(x,y) == sqrt_r(x^2 + y^2)`. `recip(x) == 1/x` for `x != 0`, else unspecified. `signum(x)` is `1` or `-1`, matching the sign of a nonzero `x`. `is_finite` is unconstrained. | vstd gives these std methods no specification. Each contract is the exact-real reading of the std documentation. `signum` on zero may return either sign (IEEE: depends on the sign bit), so the contract leaves it open. `is_finite` is a genuine property of floats that the exact-real model cannot express, so nothing is claimed; its one use (an assertion in `linear_remove_redundant_sym`) will be treated as a data-dependent abort. |
| T7 | `neg`, `pos_infinity`, `neg_infinity` (`external_body`, `inline(always)`) | `rv(neg(x)) == -rv(x)`. The infinities are unconstrained. | Verus rejects unary `-x` on floats and the std float constants. The wrappers are one-instruction inline functions whose bodies are the rejected expressions, so executable code and performance are unchanged. The infinities are only used as initial sentinels in the solver's bookkeeping; the model makes no claim about them. |
| T8 | `abort(msg) -> !` (`external_body`) | Diverges. No precondition, no postcondition. | The solver panics on data-dependent failures (inconsistent constraints, non-positive-definite systems, non-convergence). vstd forbids `panic!` unless proved unreachable, which is impossible for these data-dependent checks. `abort` makes such exits explicit: a function that aborts establishes nothing, and verified postconditions only cover normal returns. Reachability of aborts is deliberately not analyzed. |
| T9 | `assume_specification` for `usize::isqrt` | `r * r <= x < (r + 1) * (r + 1)` (integer arithmetic). | vstd gives `isqrt` no specification. The contract is the std documentation ("the square root rounded down"). It is used once, to read the block size `k` of an LMI from the length `k^2` of its constant matrix; the executable check `k * k == len` that follows aborts otherwise, so only the existence of the integer square root is relied upon. |

Items T7 and T8 are counted by Verus as "verified" functions (4 in total)
because `external_body` functions are reported as such.

Layer history: L01 added no trusted items. It uses only `neg` (T7, in
`negscal_vect`), the literals `0.0` and `1.0` (T5), and the `mul_add` and
`sqrt` contracts (T6). L02 added no trusted items; it uses the literals
`0.0`, `1.0`, `2.0` (T5) and the `abs`, `mul_add`, `sqrt` contracts (T6).
L03 (`src/lin_alg/mat_spec.rs`) is spec- and proof-only over Verus reals and
added no trusted items; it uses no `f64` literal or contract.
L04 added no trusted items; it uses the literals `0.0`, `0.5` (T5) and no
method contract. Its executable code calls vstd's `slice_to_vec` wrapper
(vstd's own `external_body` specification of `<[T]>::to_vec`, `out@ == slice@`)
in place of the unsupported method call; vstd is used as a library and is not
part of this project's trusted base.
L05 added no trusted items. Its solvers use the literals `0.0`, `1.0` and
`1e-10` (the latter through the constant `TOL`, now declared inside the
`verus!` block of `src/lin_alg/matrix.rs` so that Verus can read its value;
T5), the `abs` and `max` contracts (T6), and `abort` (T8) for the three
data-dependent "division by zero" exits that were `panic!` calls. The
division `vect[j] / d` in each solver is covered by T3 because the preceding
tolerance check establishes `|rv(d)| >= TOL * scale >= 1e-10 > 0`.
L06 added no trusted items. `ldlt_decomp` uses the literals `0.0`, `1.0`, `TOL`
(T5) and the `max`, `mul_add` contracts (T6); its division by `d[j]` is covered
by T3 because the pivot check of the earlier row established
`rv(d[j]) >= TOL * max(max_i |A_ii|, 1) >= 1e-10 > 0`. `pos_def_solver` replaces
its data-dependent `panic!` on a failed factorization by `abort` (T8).
L07 added no trusted items. `psd_block_check` uses the literals `1e-9`, `1.0` (T5) and
the `max` contract (T6); its positivity argument rests on the pivot bound of
`ldlt_decomp` (L06) and the constant `TOL > 0` (T5).
L08 added no trusted items. `jacobi_rot` uses the literals `0.0`, `1.0`, `2.0` (T5) and
the `signum`, `abs`, `sqrt`, `recip` contracts (T6). Its three divisions are covered by
T3: by `2 a_jk` (nonzero by the precondition), by `|tau| + sqrt(1 + tau^2)` (positive
since the square root is nonnegative with square `>= 1`, T1), and `recip` of
`sqrt(1 + t^2)` (positive for the same reason). The `sqrt` contract applies because
both arguments are `>= 1`.
L09 (`src/lin_alg/eigen_spec.rs`) is spec- and proof-only over Verus reals and added
no trusted items; its square-root facts (`lemma_sqrt_mul`, monotonicity, `sqrt(x^2) = |x|`)
are proved from T1. The threshold `1e-12` appears as the real constant
`0.000000000001real`, matching the T5 literal that L10/L11 will use.
L14 added no trusted items. `linear_remove_redundant_sym` uses the literal `0.0` (T5),
the `hypot` and `sqrt` contracts (T6: each fold step `scale' = sqrt(scale^2 + y^2)` has a
nonnegative argument, so T1 applies; the off-diagonal factor `2.0.sqrt()` has square 2),
and `abort` (T8) for the two data-dependent exits of the original code: the inconsistency
`panic!` and the `assert!(scale.is_finite(), ..)` (`is_finite` is unconstrained, T6). Its
division by the row scale is covered by T3: the scale is proved positive from the pivot
bound of `gauss_elim` (L13) and the precondition `abs_max_r(input) > 0`. The
`frac_1_sqrt_2` wrapper that L00 had provided for this function was removed as unused
(T7 now has three wrappers).
L13 added no trusted items. `gauss_elim` uses the literals `0.0` and `TOL` (T5), the
`abs`, `max` contracts (T6) and the comparison `scaled_near_zero` (T4). Its division
`A_jn / A_rn` is covered by T3 when the pivot is nonzero; a zero pivot (all-zero input)
leaves the multiplier unspecified, and the verified postconditions hold for any value.
L15 added no trusted items. The twelve constructors and converters use the literals `0.0`,
`1.0` (T5) and no method contract; `sym_matrix_proj` replaces its data-dependent `panic!`
on `row > col` by `abort` (T8).
L10 added no trusted items. `jacobi_eigen_cach` and its helpers use the literals `0.0`,
`1.0`, `1e-12` (T5), the `abs` and `max` contracts (T6) and `abort` (T8) for
the two data-dependent exits (`index_max_in_col` on column 0, unreachable under its
precondition, and non-convergence after `MAX_SWEEPS` sweeps). The rotation count limit
is computed with vstd's `usize::saturating_mul` contract (a library specification, not
part of this project's trusted base). The nonzero-pivot precondition of `jacobi_rot` is
discharged from the failed stopping test: `|pivot| >= 1e-12 * max(1, max_diag) > 0`.
L11 added no trusted items. `jacobi_eigen_cyc` uses the literals `0.0`, `1.0`, `1e-12` (T5),
the `abs`, `max` contracts (T6), the comparison `!= 0.0` (T4, which is exactly the
nonzero-pivot precondition of `jacobi_rot`) and `abort` (T8) for non-convergence after
`MAX_SWEEPS` sweeps.
L16 added the trusted item T9 (`usize::isqrt`). `sdp_to_standard` and its helpers use the
literals `0.0` and `1.0` (T5) and `abort` (T8) for the data-dependent exits of the original code
(wrong objective or constraint-map counts, non-square or empty constant matrices, empty map
rows, map sizes not matching the constant matrix, a nonempty right-hand side without maps).
The constraint reduction's nonzero-input precondition is discharged from the `-1` entries of
the LMI rows (with LMIs) and from the precondition `single_constraint_nonzero` (without). The
runtime `assert_eq!` checks of the original (`start == total_dim`, `zeros.len() == zeros_len`)
are now static assertions.
