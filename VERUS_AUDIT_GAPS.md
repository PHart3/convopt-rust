# Two gaps in the Verus specifications

Full verification passes (866 verified, 0 errors). These findings concern
what the contracts establish, not failed proofs or undocumented assumptions.

## 1. The equality solve is not connected to the full constraint Gram operator

A zero-pattern constraint reads an off-diagonal entry: `A_k(X) = X_ij`.
Because the Frobenius inner product counts off-diagonal entries twice, the
adjoint places `y_k / 2` in that entry, so `A_k(A_k*(y_k)) = y_k / 2`. The
definitions in [constr_spec.rs](src/lin_alg/constr_spec.rs) include this
factor correctly.

However, `dual_solve_ok` in [sdpad_spec.rs](src/sdpad_spec.rs) specifies
`y_bottom = rhs_bottom`. For distinct zero-pattern constraints orthogonal to
the dense rows, the full `AA*` system requires `y_bottom = 2 * rhs_bottom`:
a right-hand side of 1 needs a multiplier of 2, but the solve returns 1.

The proof thus verifies the implemented block solve without showing that it
is the required `AA*` solve. A zero right-hand side hides the mismatch, and
this finding does not show that a normal run reaches a nonzero one. Either an
invariant proves it stays zero, or the solve needs the factor of two. Either
way, the block system must be connected to the full constraint operator.

## 2. "Iterate" does not mean reachable from initialization

`iterate_ok` in [sdpad_spec.rs](src/sdpad_spec.rs) says a state results from
one valid update of **some arbitrary predecessor**, with some penalty in the
allowed bounds. It does not require that predecessor to come from
initialization or an earlier iteration.

So `outcome_ok` does not establish that its witness state belongs to the
actual iteration sequence. The recorded history holds residuals and flags
but no chain of states and penalty updates.

Each update call is verified, so this is a missing end-to-end guarantee, not
evidence that the loop skips updates. Closing it requires a reachability
predicate or ghost history linking initialization, successive state and
penalty updates, and the returned state.
