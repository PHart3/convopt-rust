# Two gaps in the Verus specifications

Full verification passes (866 verified, 0 errors). These findings concern what
the contracts establish, rather than failed proofs or undocumented assumptions.

## 1. The equality solve is not connected to the full constraint Gram operator

A zero-pattern constraint reads an off-diagonal entry: `A_k(X) = X_ij`.
Because the Frobenius inner product counts off-diagonal entries twice, its
adjoint places `y_k / 2` in that entry. Thus `A_k(A_k*(y_k)) = y_k / 2`.
The verified definitions in [constr_spec.rs](src/lin_alg/constr_spec.rs)
correctly include this factor.

However, `dual_solve_ok` in [sdpad_spec.rs](src/sdpad_spec.rs) specifies
`y_bottom = rhs_bottom`. For distinct zero-pattern constraints orthogonal to
the dense rows, the full `AA*` system instead requires
`y_bottom = 2 * rhs_bottom`. For example, a right-hand side of 1 requires
a multiplier of 2; the current solve returns 1.

The proof establishes the implemented block solve without proving that it is
the required `AA*` solve. A zero right-hand side hides the mismatch. This
finding does not establish that a normal solver run reaches a nonzero one:
an invariant proving it always stays zero could justify the current behavior.
Otherwise, the solve needs the factor of two. Either justification must connect
the block system to the full constraint operator.

## 2. “Iterate” does not mean reachable from initialization

`iterate_ok` in [sdpad_spec.rs](src/sdpad_spec.rs) says that a state is the
result of one valid update from **some arbitrary predecessor**, using some
penalty within the allowed bounds. It does not require that predecessor to
come from initialization or an earlier solver iteration.

Consequently, `outcome_ok` does not establish that its witness state belongs
to the actual iteration sequence. The recorded history contains residual
values and flags, without linking them to a chain of states and penalty
updates.

Individual update calls are verified; this is a missing end-to-end guarantee,
not evidence that the executable loop skips updates. Closing the gap requires
a reachability predicate or ghost history linking initialization, successive
state and penalty updates, and the selected return state.
