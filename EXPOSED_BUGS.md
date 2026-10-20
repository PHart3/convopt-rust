# Bugs Exposed by Verification

Genuine errors in the original Rust code found while verifying it with Verus.
`cargo test` passes after each fix.

## 1. Row norms in `linear_remove_redundant_sym` counted entries twice

**Where:** `src/lin_alg/sym_matrix.rs`, the loop computing the norm `scale`
of each kept constraint row (L14).

**Bug.** A row stores a symmetric matrix in packed form, column by column:
column `k` occupies indices `k(k+1)/2` to `k(k+1)/2 + k`, ending with the
diagonal entry `diag`. The loop advanced with `start += k`, but the next
column begins at `diag + 1 = start + k + 1`. From the third column on,
`start` fell further behind and the loop re-added entries of the previous
column. For a 3 x 3 matrix, entry `(1, 1)` was counted one and a half times.

**How Verus exposed it.** The loop invariant requires `start` to be the
packed start of column `k`. `start += k` does not preserve it.

**Fix.** `start = diag + 1`. Any positive scale gives the same solution set,
so only the conditioning of the reduced system changed.

## 2. Cached Jacobi: stale bound and out-of-column cache read

**Where:** `src/lin_alg/jacobi.rs`, the cache update of `jacobi_eigen_cach`
(L10).

**Background.** For each column `m >= 1`, `maxima[m - 1]` stores the row of
the largest-magnitude entry above the diagonal. A rotation at `(j, k)`
changes only rows and columns `j` and `k`, so the cache is refreshed with a
few comparisons against `biggest`, the magnitude of the pivot just rotated:
every unrotated entry is at most that large.

**Bug (a): stale `biggest`.** The rotation's return value went into
`biggest_new` and reached `biggest` only at the next rotation. The update
after rotation `t` therefore used the pivot of rotation `t - 1`. Rotations
can grow off-diagonal entries by up to `sqrt 2`, so the cache could keep a
row that was no longer maximal.

**Bug (b): out-of-column read.** The last branch compared the entries at
`m(m+1)/2 + j` and `m(m+1)/2 + k`, which belong to column `m` only when
`j < k < m`. The branch was also reached for `j < m < k`, where it read a
later column and could cache the row `k >= m`. The next pivot search then
called `jacobi_rot` with its indices in the wrong order, which panics.

**How Verus exposed it.** `jacobi_rot` requires `j < k`, and the invariant
says `maxima[m - 1]` is a maximal row `< m` of column `m`. Proving this
needs `biggest` to bound every unrotated entry, which the stale value does
not, and the last branch cannot be proved for `m < k`. Random tests of the
original code confirmed both: invalid caches in a few hundred of 2000 runs,
and crashes from rotations with `j >= k`.

**Fix.** `biggest` is the value returned by the latest `jacobi_rot`. The
last branch is split by the position of `m`: for `k < m` compare both
rotated entries; for `j < m < k` only row `j` changed, so it replaces the
cached row only if it is now larger; for `m < j` the column is unchanged.
The regression test `test_jacobi_cach_cache_update` (`src/sdp_tests.rs`)
crashes the original solver.

## 3. No PSD check for single-variable problems

**Where:** `src/sdp_altdir.rs`, the block list `psd_blocks` that
`psd_block_check` examines before returning a solution (L17).

**Bug.** `psd_blocks` was built from the LMI and user-variable block sizes.
Without LMIs both lists are empty, so the check was vacuously true. The
solver could return, and record as best feasible, an iterate that is not
PSD, although `Z >= 0` is a constraint of the standard form.

**How Verus exposed it.** The setup contract says `psd_blocks` partitions
the standard-form variable: the block sizes sum to the order of `Z`. In the
no-LMI case there was no block to list, contradicting that sum.

**Fix.** `psd_block_list` returns the single block `(0, var_dim)` when there
are no LMIs, matching the author's fix on `main`. The list is now built once
before the main loop.

## 4. The dual solve used `I` instead of `I/2` for the zero constraints

**Where:** `src/lin_alg/sym_matrix.rs`, `pos_def_solver_upperleft`, called
by `dual_solve` in `src/sdp_altdir.rs` (L17).

**Bug.** Each iteration solves `(A A*) y = rhs`, where `A` has the dense
constraint rows followed by one row per off-block-diagonal entry that must
vanish. Because the Frobenius product counts off-diagonal entries twice,
`A*` places `y_k / 2` at a zero-pattern entry, so the Gram block of these
rows is `I/2`. The solver passed their right-hand sides through unchanged,
solving `(G, I) y = rhs` and halving their multipliers. In exact arithmetic
these right-hand sides stay zero, but the eigensolver does not preserve
block diagonality, so in practice they are small but nonzero, and the
off-block entries of `Z` were damped by `1 - rho/2` instead of `1 - rho`.

**How Verus exposed it.** The verified contract `dual_solve_ok` stated what
the code did, `y_bottom = rhs_bottom`, while the verified adjoint identity
fixed the factor `1/2` in `A*`. A specification audit found the two agree
with the `A A*` system only when `rhs_bottom = 0`.

**Fix.** `pos_def_solver_upperleft` returns `2 * rhs[t]` for these rows, and
the contracts state the system `(G, I/2) y = rhs`. `lemma_wgy_dual_gram`
now proves that every iteration's multipliers solve the `A A*` system of
the full constraint operator.
