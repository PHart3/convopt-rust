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
| T7 | `neg`, `pos_infinity`, `neg_infinity`, `frac_1_sqrt_2` (`external_body`, `inline(always)`) | `rv(neg(x)) == -rv(x)`. The infinities are unconstrained. `frac_1_sqrt_2()` is positive with square `1/2`. | Verus rejects unary `-x` on floats and the std float constants. The wrappers are one-instruction inline functions whose bodies are the rejected expressions, so executable code and performance are unchanged. The infinities are only used as initial sentinels in the solver's bookkeeping; the model makes no claim about them. |
| T8 | `abort(msg) -> !` (`external_body`) | Diverges. No precondition, no postcondition. | The solver panics on data-dependent failures (inconsistent constraints, non-positive-definite systems, non-convergence). vstd forbids `panic!` unless proved unreachable, which is impossible for these data-dependent checks. `abort` makes such exits explicit: a function that aborts establishes nothing, and verified postconditions only cover normal returns. Reachability of aborts is deliberately not analyzed. |

Items T7 and T8 are counted by Verus as "verified" functions (5 in total)
because `external_body` functions are reported as such.

Layer history: L01 added no trusted items. It uses only `neg` (T7, in
`negscal_vect`), the literals `0.0` and `1.0` (T5), and the `mul_add` and
`sqrt` contracts (T6). L02 added no trusted items; it uses the literals
`0.0`, `1.0`, `2.0` (T5) and the `abs`, `mul_add`, `sqrt` contracts (T6).
L03 (`src/lin_alg/mat_spec.rs`) is spec- and proof-only over Verus reals and
added no trusted items; it uses no `f64` literal or contract.
