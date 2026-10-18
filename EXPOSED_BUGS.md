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
