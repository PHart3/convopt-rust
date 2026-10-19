# Bugs Exposed by Verification

Genuine errors in the original Rust code that came to light while verifying it
with Verus. Each entry says what was wrong, how the verification effort exposed
it, and what was changed. This file is for human review only.

## Row normalization in `linear_remove_redundant_sym` double-counted packed entries

**Where:** `src/lin_alg/sym_matrix.rs`, the loop that computes the norm `scale` of
each kept constraint row (layer L14, 2026-10-18).

**What was wrong.** A constraint row stores a symmetric matrix in packed form,
column by column: column `k` occupies packed indices `k(k+1)/2` through
`k(k+1)/2 + k`, the last of which is the diagonal entry `(k, k)`. The loop walked
the columns, adding the off-diagonal entries from `start` up to the diagonal index
`diag`, then the diagonal entry, and then advanced with

    start += k;

The next column actually begins at `diag + 1`, which is `start + k + 1`. So from the
third column on, `start` lagged further behind each time, and the loop re-added the
last `k - 1` entries of the previous column. For a 3 x 3 matrix the diagonal entry
`(1, 1)` was counted one and a half times.

**How Verus exposed it.** The specification of the norm is a sum over the packed
triangle, column by column (`row_sq_upto_r` in `src/lin_alg/reduce_spec.rs`). The
loop invariant needs `start` to equal the packed start index of column `k`. That
invariant is established for `k = 1` but is not preserved by `start += k`; the
proof could not be closed without first identifying what the loop actually
summed, which made the overlap obvious.

**Fix.** `start = diag + 1`. The solution set of the reduced system never depended
on the scale (any positive scale works), so only the conditioning of the reduced
system changes. `cargo test` passes.

## Cached Jacobi: the cache update used a stale maximum and an out-of-column index

**Where:** `src/lin_alg/jacobi.rs`, the cache update loop of `jacobi_eigen_cach`
(layer L10, 2026-10-18).

**Background.** The cached solver keeps, for every column `m >= 1` of the packed
symmetric matrix, the row `maxima[m - 1]` of the entry of largest magnitude among the
rows `0..m`. After a rotation at the pivot `(j, k)` only rows and columns `j` and `k`
change, so the cache is refreshed column by column with a few comparisons instead of a
full rescan. The refresh compares the rotated entries against `biggest`, meant to be the
magnitude of the pivot just rotated: every entry that was not rotated is at most that
large, so if the cached entry still has magnitude at least `biggest` after the rotation,
only the rotated rows can beat it.

**What was wrong (1): stale `biggest`.** The rotation returns the magnitude of the
pivot it annihilated, but the solver stored that value in `biggest_new` and copied it
into `biggest` only at the *next* rotation:

    (eigenvals, eigenvects, biggest) = (eigenvals_new, eigenvects_new, biggest_new);
    (eigenvals_new, eigenvects_new, biggest_new) = jacobi_rot(eigenvals, eigenvects, dim, k, j);

So during the cache update after rotation `t`, `biggest` held the pivot magnitude of
rotation `t - 1`. A rotation can make an off-diagonal entry up to `sqrt 2` times larger
than the previous maximum, so the stale value could be smaller than the true bound on
the unrotated entries, and the "cached entry still dominates" shortcut could then keep a
row that was no longer maximal.

**What was wrong (2): out-of-column index.** The last branch of the update,

    } else if |A[m(m+1)/2 + j]| > |A[m(m+1)/2 + k]| { *i = j } else { *i = k }

reads the entries `(j, m)` and `(k, m)` of column `m`, which is only correct when
`j < k < m`. The branch is also reached when `j < m < k` (and, with a tie, when
`m < j`). Then `m(m+1)/2 + k` lies in a later column, the comparison is meaningless, and
the cache may receive the row index `k >= m`, which is not a row of column `m`. The next
pivot search then reads the same wrong position and can select `(k, m)` with `k > m` as
the pivot, calling `jacobi_rot` with the pivot indices in the wrong order. The original
`jacobi_rot` panics in that case (`split_at_mut` / slice range out of bounds).

**How Verus exposed it.** `jacobi_rot` requires `j < k`, and the loop invariant of the
solver states that `maxima[m - 1]` is a maximal row `< m` of column `m`. Proving the
invariant through the update loop requires, for the "cached entry still dominates"
branches, that `biggest` bounds every unrotated entry of the matrix *before the last
rotation*; the stale value gives no such bound. The last branch could not be proved at
all for `m < k`, since `m(m+1)/2 + k` is not the packed index of an entry of column `m`.
An instrumented run of the original code on random symmetric matrices (orders 3 to 10)
confirmed both: invalid caches after a few hundred of 2000 runs, and dozens of rotations
called with `j >= k`, which crash. The matrix of `test_jacobi_cach_cache_update` in
`src/sdp_tests.rs` crashes the original solver.

**Fix.** `biggest` is now the value returned by the last `jacobi_rot` call (the pivot
magnitude of the matrix just rotated; `biggest_new` is gone). The last branch is split:
for `k < m` it compares the two rotated entries as before; for `j < m < k` only row `j`
was rotated, and it replaces the cached row only if it now exceeds the cached entry; for
`m < j` the column is unrotated and the cache is kept. The verified invariant states that
after the update every `maxima[m - 1]` is a maximal row of column `m`, and the pivot
search then returns a position of maximal off-diagonal magnitude. `cargo test` passes,
including the new regression test.
