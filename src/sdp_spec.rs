// Spec-level support for the standard-form conversion `sdp_to_standard` (not trusted).
//
// The user formulation (sdp.rs) stores every linear map on symmetric matrices as
// a MatrixSym: a list of partitioned symmetric matrices, each acting on a packed
// symmetric matrix through the packed dot product. The standard-form variable
// Z is one large packed symmetric matrix whose diagonal blocks are, in order,
// the LMI blocks B_0, .., B_{s-1} and the pairs B_{s+2n}, B_{s+2n+1} holding the
// user variables Z_n = B_{s+2n} - B_{s+2n+1}.
//
// This module defines the packed views of user maps (`symrow_r`, `part_r`),
// the block extraction `blk_r` and the variable `zvar_r`, the integer-square-root
// and block-layout arithmetic, the key identity behind every embedding the
// conversion performs,
//
//     <embed(w, st), z> = <w, blk(z, st, d)>        (`lemma_embed_dot`),
//
// its row form for column-major matrices, the linearity facts of row dot
// products, and the layout (`zeros_ok`) and row-wise (`std_rows_ok`) predicates
// that make up the contract of the conversion.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::lemma_dot_sum_fn;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::solve_spec::abs_max_r;
#[cfg(verus_keep_ghost)]
use crate::real_spec::abs_r;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_mod_bound, lemma_fundamental_div_mod_converse};

verus! {

broadcast use crate::trusted::group_real_model;



/* ---------- packed views of user maps ---------- */

/// A partitioned d x d symmetric matrix (column c holds the entries (0, c), .., (c, c)) as a packed vector.
pub open spec fn part_r(v: Seq<Vec<f64>>, d: int) -> Seq<real> {
    Seq::new(pk_len(d) as nat, |t: int| rv(v[pk_unpack(t).1]@[pk_unpack(t).0]))
}

/// Row k of a MatrixSym as a packed vector.
pub open spec fn symrow_r(m: Seq<Vec<Vec<f64>>>, k: int, d: int) -> Seq<real> {
    part_r(m[k]@, d)
}

/// The d x d diagonal block at offset st of a packed matrix z, as a packed vector.
pub open spec fn blk_r(z: Seq<real>, st: int, d: int) -> Seq<real> {
    Seq::new(pk_len(d) as nat, |t: int| z[pk_idx(st + pk_unpack(t).0, st + pk_unpack(t).1)])
}

/// The user variable read from z: the block at st minus the block at st + d.
pub open spec fn zvar_r(z: Seq<real>, st: int, d: int) -> Seq<real> {
    vsub_r(blk_r(z, st, d), blk_r(z, st + d, d))
}

pub proof fn lemma_part_entry(v: Seq<Vec<f64>>, d: int, r: int, c: int)
    requires
        0 <= r <= c < d,
    ensures
        0 <= pk_idx(r, c) < pk_len(d),
        part_r(v, d)[pk_idx(r, c)] == rv(v[c]@[r]),
{
    lemma_pk_idx_bound(r, c, d);
    lemma_pk_unpack_idx(r, c);
}

pub proof fn lemma_blk_entry(z: Seq<real>, st: int, d: int, r: int, c: int)
    requires
        0 <= r <= c < d,
    ensures
        0 <= pk_idx(r, c) < pk_len(d),
        blk_r(z, st, d)[pk_idx(r, c)] == z[pk_idx(st + r, st + c)],
{
    lemma_pk_idx_bound(r, c, d);
    lemma_pk_unpack_idx(r, c);
}

/* ---------- triangle sums supported on a diagonal block ---------- */

/// A triangle sum of order sd whose summand vanishes outside the d x d block at st
/// is the triangle sum of the shifted summand over the block.
pub proof fn lemma_tri_sum_block(f: spec_fn(int, int) -> real, sd: int, st: int, d: int)
    requires
        0 <= st,
        0 <= d,
        st + d <= sd,
        forall|r: int, c: int| 0 <= r <= c < sd && !(st <= r && c < st + d) ==> #[trigger] f(r, c) == 0.0real,
    ensures
        tri_sum_r(f, sd as nat) == tri_sum_r(|a: int, b: int| f(st + a, st + b), d as nat),
{
    let cols = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    let g = |a: int, b: int| f(st + a, st + b);
    let gcols = |b: int| sum_fn_r(|a: int| g(a, b), (b + 1) as nat);
    let zero = |i: int| 0.0real;
    // columns before the block and after it sum to zero
    assert forall|c: int| 0 <= c < sd && !(st <= c < st + d) implies #[trigger] cols(c) == 0.0real by {
        let colf = |r: int| f(r, c);
        assert forall|r: int| 0 <= r < c + 1 implies #[trigger] colf(r) == zero(r) by {
            assert(f(r, c) == 0.0real);
        }
        lemma_sum_fn_ext(colf, zero, (c + 1) as nat);
        lemma_sum_fn_zero((c + 1) as nat);
    }
    // a block column: the rows before the block vanish, the rest is the shifted column
    assert forall|b: int| 0 <= b < d implies #[trigger] cols(st + b) == gcols(b) by {
        let colf = |r: int| f(r, st + b);
        lemma_sum_fn_split(colf, st as nat, (b + 1) as nat);
        assert forall|r: int| 0 <= r < st implies #[trigger] colf(r) == zero(r) by {
            assert(f(r, st + b) == 0.0real);
        }
        lemma_sum_fn_ext(colf, zero, st as nat);
        lemma_sum_fn_zero(st as nat);
        let shifted = |i: int| colf(st + i);
        let gcol = |a: int| g(a, b);
        assert forall|a: int| 0 <= a < b + 1 implies #[trigger] shifted(a) == gcol(a) by {}
        lemma_sum_fn_ext(shifted, gcol, (b + 1) as nat);
    }
    // assemble the outer sum: [0, st) + [st, st + d) + [st + d, sd)
    lemma_sum_fn_split(cols, st as nat, (sd - st) as nat);
    let tail = |i: int| cols(st + i);
    lemma_sum_fn_split(tail, d as nat, (sd - st - d) as nat);
    lemma_sum_fn_ext(cols, zero, st as nat);
    lemma_sum_fn_zero(st as nat);
    let tail2 = |i: int| tail(d + i);
    assert forall|i: int| 0 <= i < sd - st - d implies #[trigger] tail2(i) == zero(i) by {
        assert(cols(st + (d + i)) == 0.0real);
    }
    lemma_sum_fn_ext(tail2, zero, (sd - st - d) as nat);
    lemma_sum_fn_zero((sd - st - d) as nat);
    assert forall|b: int| 0 <= b < d implies #[trigger] tail(b) == gcols(b) by {
        assert(cols(st + b) == gcols(b));
    }
    lemma_sum_fn_ext(tail, gcols, d as nat);
}

/* ---------- the embedding identity ---------- */

/// A packed sd x sd matrix e holding w on the d x d block at st and zero elsewhere
/// satisfies <e, z> = <w, blk(z, st, d)> (packed dot products).
pub proof fn lemma_embed_dot(e: Seq<real>, w: Seq<real>, z: Seq<real>, sd: int, st: int, d: int)
    requires
        0 <= st,
        0 <= d,
        st + d <= sd,
        e.len() == pk_len(sd),
        z.len() == pk_len(sd),
        w.len() == pk_len(d),
        forall|a: int, b: int| 0 <= a <= b < d ==> e[#[trigger] pk_idx(st + a, st + b)] == w[pk_idx(a, b)],
        forall|t: int| 0 <= t < pk_len(sd) && !is_block_idx(t, st, d) ==> #[trigger] e[t] == 0.0real,
    ensures
        dot_r(e, z) == dot_r(w, blk_r(z, st, d)),
{
    let blk = blk_r(z, st, d);
    let big = |t: int| e[t] * z[t];
    let small = |t: int| w[t] * blk[t];
    lemma_dot_sum_fn(e, z);
    lemma_dot_sum_fn(w, blk);
    lemma_tri_sum_packed(big, sd as nat);
    lemma_tri_sum_packed(small, d as nat);
    let f = |r: int, c: int| big(pk_idx(r, c));
    let g = |a: int, b: int| small(pk_idx(a, b));
    // f vanishes outside the block
    assert forall|r: int, c: int| 0 <= r <= c < sd && !(st <= r && c < st + d) implies #[trigger] f(r, c) == 0.0real by {
        lemma_pk_idx_bound(r, c, sd);
        if is_block_idx(pk_idx(r, c), st, d) {
            let (a, b) = choose|a: int, b: int| 0 <= a <= b < d && pk_idx(r, c) == #[trigger] pk_idx(st + a, st + b);
            lemma_pk_idx_injective(r, c, st + a, st + b);
        }
        assert(e[pk_idx(r, c)] == 0.0real);
    }
    lemma_tri_sum_block(f, sd, st, d);
    // on the block, f is the shifted g
    let fs = |a: int, b: int| f(st + a, st + b);
    assert forall|a: int, b: int| 0 <= a <= b < d implies #[trigger] fs(a, b) == g(a, b) by {
        lemma_blk_entry(z, st, d, a, b);
        assert(f(st + a, st + b) == e[pk_idx(st + a, st + b)] * z[pk_idx(st + a, st + b)]);
    }
    lemma_tri_sum_ext(fs, g, d as nat);
}

/// Row form: row i of a column-major matrix m (row_dim rows, pk_len(sd) columns) that holds w on
/// the block at st and zero elsewhere has row dot product <w, blk(z, st, d)>.
pub proof fn lemma_embed_row_dot(m: Seq<real>, row_dim: int, i: int, w: Seq<real>, z: Seq<real>, sd: int, st: int, d: int)
    requires
        0 <= st,
        0 <= d,
        st + d <= sd,
        0 <= i < row_dim,
        m.len() == row_dim * pk_len(sd),
        z.len() == pk_len(sd),
        w.len() == pk_len(d),
        forall|a: int, b: int| 0 <= a <= b < d ==> m[cm_idx(row_dim, i, #[trigger] pk_idx(st + a, st + b))] == w[pk_idx(a, b)],
        forall|t: int| 0 <= t < pk_len(sd) && !is_block_idx(t, st, d) ==> m[#[trigger] cm_idx(row_dim, i, t)] == 0.0real,
    ensures
        row_dot_r(m, row_dim, i, z) == dot_r(w, blk_r(z, st, d)),
{
    let e = Seq::new(pk_len(sd) as nat, |t: int| cm_entry(m, row_dim, i, t));
    assert(row_dim * pk_len(sd) == pk_len(sd) * row_dim);
    assert forall|a: int, b: int| 0 <= a <= b < d implies e[#[trigger] pk_idx(st + a, st + b)] == w[pk_idx(a, b)] by {
        lemma_pk_idx_bound(st + a, st + b, sd);
    }
    lemma_embed_dot(e, w, z, sd, st, d);
    lemma_dot_sum_fn(e, z);
    lemma_sum_fn_ext(|t: int| e[t] * z[t], |n: int| constr_entry(m, row_dim, i, n) * z[n], pk_len(sd) as nat);
}

/* ---------- linearity of row dot products ---------- */

/// Row dot products of a matrix built by matrix_add_diff: m = m1 + (m2 - m3) entrywise.
pub proof fn lemma_row_dot_add_sub(m: Seq<real>, m1: Seq<real>, m2: Seq<real>, m3: Seq<real>, row_dim: int, i: int, z: Seq<real>)
    requires
        0 <= i < row_dim,
        m1.len() == m2.len() == m3.len(),
        m1.len() == row_dim * z.len(),
        m == vadd_r(m1, vsub_r(m2, m3)),
    ensures
        row_dot_r(m, row_dim, i, z) == row_dot_r(m1, row_dim, i, z) + row_dot_r(m2, row_dim, i, z) - row_dot_r(m3, row_dim, i, z),
{
    let f1 = |n: int| constr_entry(m1, row_dim, i, n) * z[n];
    let f2 = |n: int| constr_entry(m2, row_dim, i, n) * z[n];
    let f3 = |n: int| constr_entry(m3, row_dim, i, n) * z[n];
    let f = |n: int| constr_entry(m, row_dim, i, n) * z[n];
    let f23 = |n: int| f2(n) - f3(n);
    let n3 = |n: int| -f3(n);
    assert(row_dim * z.len() == z.len() * row_dim);
    assert forall|n: int| 0 <= n < z.len() implies #[trigger] f(n) == f1(n) + f23(n) by {
        lemma_cm_idx_bound(row_dim, i, n, z.len() as int);
        let t = cm_idx(row_dim, i, n);
        assert(m[t] == m1[t] + (m2[t] - m3[t]));
        assert((m1[t] + (m2[t] - m3[t])) * z[n] == m1[t] * z[n] + (m2[t] * z[n] - m3[t] * z[n])) by (nonlinear_arith);
    }
    assert forall|n: int| 0 <= n < z.len() implies #[trigger] f23(n) == f2(n) + n3(n) by {}
    assert forall|n: int| 0 <= n < z.len() implies #[trigger] n3(n) == (-1.0real) * f3(n) by {}
    lemma_sum_fn_add(f1, f23, f, z.len());
    lemma_sum_fn_add(f2, n3, f23, z.len());
    lemma_sum_fn_scale(f3, n3, -1.0real, z.len());
}

/// Row dot products of the zero matrix vanish.
pub proof fn lemma_row_dot_zero(m: Seq<real>, row_dim: int, i: int, z: Seq<real>)
    requires
        0 <= i < row_dim,
        m.len() == row_dim * z.len(),
        forall|t: int| 0 <= t < m.len() ==> #[trigger] m[t] == 0.0real,
    ensures
        row_dot_r(m, row_dim, i, z) == 0.0real,
{
    let f = |n: int| constr_entry(m, row_dim, i, n) * z[n];
    let zero = |n: int| 0.0real;
    assert(row_dim * z.len() == z.len() * row_dim);
    assert forall|n: int| 0 <= n < z.len() implies #[trigger] f(n) == zero(n) by {
        lemma_cm_idx_bound(row_dim, i, n, z.len() as int);
    }
    lemma_sum_fn_ext(f, zero, z.len());
    lemma_sum_fn_zero(z.len());
}

/// Adding delta to entry (i0, t0) adds delta * z[t0] to row i0's dot product and leaves the other rows.
pub proof fn lemma_row_dot_update(m: Seq<real>, m2: Seq<real>, row_dim: int, i0: int, t0: int, delta: real, i: int, z: Seq<real>)
    requires
        0 <= i0 < row_dim,
        0 <= i < row_dim,
        0 <= t0 < z.len(),
        m.len() == row_dim * z.len(),
        m2 == m.update(cm_idx(row_dim, i0, t0), m[cm_idx(row_dim, i0, t0)] + delta),
    ensures
        row_dot_r(m2, row_dim, i, z) == row_dot_r(m, row_dim, i, z) + (if i == i0 { delta * z[t0] } else { 0.0real }),
{
    let f = |n: int| constr_entry(m, row_dim, i, n) * z[n];
    let g = |n: int| constr_entry(m2, row_dim, i, n) * z[n];
    assert(row_dim * z.len() == z.len() * row_dim);
    lemma_cm_idx_bound(row_dim, i0, t0, z.len() as int);
    assert forall|n: int| 0 <= n < z.len() && !(i == i0 && n == t0) implies #[trigger] g(n) == f(n) by {
        lemma_cm_idx_bound(row_dim, i, n, z.len() as int);
        lemma_cm_idx_injective(row_dim, i, n, i0, t0);
    }
    if i == i0 {
        let x = m[cm_idx(row_dim, i0, t0)];
        assert(m2[cm_idx(row_dim, i0, t0)] == x + delta);
        assert((x + delta) * z[t0] == x * z[t0] + delta * z[t0]) by (nonlinear_arith);
        assert(g(t0) == f(t0) + delta * z[t0]);
        lemma_sum_fn_update(f, g, z.len(), t0, delta * z[t0]);
    } else {
        lemma_sum_fn_ext(f, g, z.len());
    }
}

/// Packed dot product of a vector built by `v = v1 + (v2 - v3)` entrywise.
pub proof fn lemma_dot_add_sub(v: Seq<real>, v1: Seq<real>, v2: Seq<real>, v3: Seq<real>, z: Seq<real>)
    requires
        v1.len() == v2.len() == v3.len() == z.len(),
        v.len() == z.len(),
        forall|t: int| 0 <= t < z.len() ==> #[trigger] v[t] == v1[t] + (v2[t] - v3[t]),
    ensures
        dot_r(v, z) == dot_r(v1, z) + dot_r(v2, z) - dot_r(v3, z),
{
    lemma_dot_sum_fn(v, z);
    lemma_dot_sum_fn(v1, z);
    lemma_dot_sum_fn(v2, z);
    lemma_dot_sum_fn(v3, z);
    let f1 = |t: int| v1[t] * z[t];
    let f2 = |t: int| v2[t] * z[t];
    let f3 = |t: int| v3[t] * z[t];
    let f = |t: int| v[t] * z[t];
    let f23 = |t: int| f2(t) - f3(t);
    let n3 = |t: int| -f3(t);
    assert forall|t: int| 0 <= t < z.len() implies #[trigger] f(t) == f1(t) + f23(t) by {
        assert((v1[t] + (v2[t] - v3[t])) * z[t] == v1[t] * z[t] + (v2[t] * z[t] - v3[t] * z[t])) by (nonlinear_arith);
    }
    assert forall|t: int| 0 <= t < z.len() implies #[trigger] f23(t) == f2(t) + n3(t) by {}
    assert forall|t: int| 0 <= t < z.len() implies #[trigger] n3(t) == (-1.0real) * f3(t) by {}
    lemma_sum_fn_add(f1, f23, f, z.len());
    lemma_sum_fn_add(f2, n3, f23, z.len());
    lemma_sum_fn_scale(f3, n3, -1.0real, z.len());
}

/// The packed dot product vanishes when the left factor is zero.
pub proof fn lemma_dot_zero(v: Seq<real>, z: Seq<real>)
    requires
        v.len() == z.len(),
        forall|t: int| 0 <= t < z.len() ==> #[trigger] v[t] == 0.0real,
    ensures
        dot_r(v, z) == 0.0real,
{
    lemma_dot_sum_fn(v, z);
    lemma_sum_fn_all_zero(|t: int| v[t] * z[t], z.len());
}

/* ---------- integer square root and block layout arithmetic ---------- */

/// k is the integer square root of x.
pub open spec fn is_isqrt(k: int, x: int) -> bool {
    0 <= k && k * k <= x < (k + 1) * (k + 1)
}

/// The integer square root of x >= 0 (unspecified for x < 0).
pub open spec fn isqrt_r(x: int) -> int {
    choose|k: int| is_isqrt(k, x)
}

/// The integer square root exists.
pub proof fn lemma_isqrt_exists(x: int)
    requires
        x >= 0,
    ensures
        0 <= isqrt_r(x),
        isqrt_r(x) * isqrt_r(x) <= x < (isqrt_r(x) + 1) * (isqrt_r(x) + 1),
    decreases x,
{
    if x == 0 {
        assert(is_isqrt(0, 0));
    } else {
        lemma_isqrt_exists(x - 1);
        let k = isqrt_r(x - 1);
        if (k + 1) * (k + 1) <= x {
            assert((k + 2) * (k + 2) > x) by (nonlinear_arith)
                requires k >= 0, x - 1 < (k + 1) * (k + 1);
            assert(is_isqrt(k + 1, x));
        } else {
            assert(is_isqrt(k, x));
        }
    }
}

/// Sum of the first n entries of an integer sequence.
pub open spec fn sum_upto(s: Seq<int>, n: int) -> int
    decreases n,
{
    if n <= 0 { 0 } else { sum_upto(s, n - 1) + s[n - 1] }
}

/// Sum of the squares of the first n entries.
pub open spec fn sq_sum_upto(s: Seq<int>, n: int) -> int
    decreases n,
{
    if n <= 0 { 0 } else { sq_sum_upto(s, n - 1) + s[n - 1] * s[n - 1] }
}

/// Prefix sums of a nonnegative sequence are nondecreasing.
pub proof fn lemma_sum_upto_monotone(s: Seq<int>, n1: int, n2: int)
    requires
        n1 <= n2,
        forall|i: int| 0 <= i < s.len() ==> #[trigger] s[i] >= 0,
        n2 <= s.len(),
    ensures
        sum_upto(s, n1) <= sum_upto(s, n2),
    decreases n2 - n1,
{
    if n1 < n2 {
        lemma_sum_upto_monotone(s, n1, n2 - 1);
    }
}

pub proof fn lemma_sq_sum_upto_monotone(s: Seq<int>, n1: int, n2: int)
    requires
        n1 <= n2,
        n2 <= s.len(),
    ensures
        sq_sum_upto(s, n1) <= sq_sum_upto(s, n2),
    decreases n2 - n1,
{
    if n1 < n2 {
        lemma_sq_sum_upto_monotone(s, n1, n2 - 1);
        assert(s[n2 - 1] * s[n2 - 1] >= 0) by (nonlinear_arith);
    }
}

/// The diagonal blocks of the standard-form variable, in order: the s LMI blocks (sizes bd),
/// then two blocks of size vd[n] for every user variable n.
pub open spec fn block_sizes(bd: Seq<int>, vd: Seq<int>) -> Seq<int> {
    bd + Seq::new(2 * vd.len(), |j: int| var_block_size(vd, j))
}

/// Size of the j-th variable block (the two blocks of variable n have size vd[n]).
pub open spec fn var_block_size(vd: Seq<int>, j: int) -> int {
    vd[j / 2]
}

/// Offset of LMI block i.
pub open spec fn block_off(bd: Seq<int>, i: int) -> int {
    sum_upto(bd, i)
}

/// Total size of the LMI blocks.
pub open spec fn block_sum(bd: Seq<int>) -> int {
    sum_upto(bd, bd.len() as int)
}

/// Offset of the first block of user variable n, after the LMI blocks of total size bsum
/// (the second block of the variable follows at var_off + vd[n]).
pub open spec fn var_off(bsum: int, vd: Seq<int>, n: int) -> int {
    bsum + 2 * sum_upto(vd, n)
}

/// Order of the standard-form variable.
pub open spec fn std_dim(bd: Seq<int>, vd: Seq<int>) -> int {
    var_off(block_sum(bd), vd, vd.len() as int)
}

/// First row of LMI i in the stacked constraint matrix (LMI i has bd[i]^2 rows).
pub open spec fn row_off(bd: Seq<int>, i: int) -> int {
    sq_sum_upto(bd, i)
}

/// Number of LMI rows.
pub open spec fn lmi_rows(bd: Seq<int>) -> int {
    sq_sum_upto(bd, bd.len() as int)
}

/* ---------- the user maps applied to the standard-form variable ---------- */

/// Row e of the n-th map of a list of maps, applied to the user variable Z_n read from z.
pub open spec fn map_term_r(maps: Seq<Vec<Vec<Vec<f64>>>>, bsum: int, vd: Seq<int>, e: int, z: Seq<real>, n: int) -> real {
    dot_r(symrow_r(maps[n]@, e, vd[n]), zvar_r(z, var_off(bsum, vd, n), vd[n]))
}

/// Row e of the sum of a list of maps (one per user variable), applied to the variables read from z:
/// sum_n maps[n][e](Z_n).
pub open spec fn maps_row_r(maps: Seq<Vec<Vec<Vec<f64>>>>, bsum: int, vd: Seq<int>, e: int, z: Seq<real>) -> real {
    sum_fn_r(|n: int| map_term_r(maps, bsum, vd, e, z, n), vd.len())
}

/// Objective term n: C_n(Z_n).
pub open spec fn obj_term_r(objective: Seq<Vec<Vec<f64>>>, bsum: int, vd: Seq<int>, z: Seq<real>, n: int) -> real {
    dot_r(part_r(objective[n]@, vd[n]), zvar_r(z, var_off(bsum, vd, n), vd[n]))
}

/// The objective C_1(Z_1) + .. + C_n(Z_n) read from z.
pub open spec fn obj_r(objective: Seq<Vec<Vec<f64>>>, bsum: int, vd: Seq<int>, z: Seq<real>) -> real {
    sum_fn_r(|n: int| obj_term_r(objective, bsum, vd, z, n), vd.len())
}

/* ---------- the zero pattern ---------- */

/// (r, c) is an upper off-block-diagonal position: r lies in a block that ends at or before c.
pub open spec fn in_zero_region(bs: Seq<int>, r: int, c: int) -> bool {
    exists|j: int| 0 <= j < bs.len() && #[trigger] sum_upto(bs, j) <= r < sum_upto(bs, j + 1) <= c
}

/// The number of upper off-block-diagonal positions: the upper triangle minus the block triangles.
pub open spec fn zeros_len(bs: Seq<int>, sd: int) -> int {
    pk_len(sd) - sum_fn_int(|j: int| pk_len(bs[j]), bs.len())
}

/// Sum of an integer-valued index function.
pub open spec fn sum_fn_int(f: spec_fn(int) -> int, n: nat) -> int
    decreases n,
{
    if n == 0 { 0 } else { sum_fn_int(f, (n - 1) as nat) + f(n - 1) }
}

/// The pair (r, c) occurs in the list.
pub open spec fn listed(list: Seq<(usize, usize)>, r: int, c: int) -> bool {
    exists|q: int| 0 <= q < list.len() && (#[trigger] list[q]).0 == r && list[q].1 == c
}

/// The zero-pattern list of the standard form: exactly the upper off-block-diagonal positions
/// of the sd x sd variable, each listed once.
pub open spec fn zeros_ok(list: Seq<(usize, usize)>, bs: Seq<int>, sd: int) -> bool {
    &&& valid_zero_list(list, sd)
    &&& forall|q: int| 0 <= q < list.len() ==> in_zero_region(bs, (#[trigger] list[q]).0 as int, list[q].1 as int)
    &&& forall|r: int, c: int| 0 <= r < c < sd && in_zero_region(bs, r, c) ==> #[trigger] listed(list, r, c)
    &&& forall|q1: int, q2: int| 0 <= q1 < list.len() && 0 <= q2 < list.len() && q1 != q2
            ==> #[trigger] list[q1] != #[trigger] list[q2]
}

/// Pushing a pair keeps every listed pair listed.
pub proof fn lemma_listed_push(list: Seq<(usize, usize)>, p: (usize, usize), r: int, c: int)
    requires
        listed(list, r, c) || (p.0 == r && p.1 == c),
    ensures
        listed(list.push(p), r, c),
{
    if listed(list, r, c) {
        let q = choose|q: int| 0 <= q < list.len() && (#[trigger] list[q]).0 == r && list[q].1 == c;
        assert(list.push(p)[q] == list[q]);
    } else {
        assert(list.push(p)[list.len() as int] == p);
    }
}

/* ---------- the standard-form rows ---------- */

/// The LMI rows and equality rows of the augmented standard-form matrix aug (column-major,
/// R = lmi_rows + m rows, pk_len(sd) + 1 columns, the last column holding the right-hand side):
/// row row_off(i) + e of LMI i reads H_i(Z_1, .., Z_n)[e] - B_i[e] with right-hand side M_i[e],
/// and row lmi_rows + e reads A_1(Z_1) + .. + A_n(Z_n) with right-hand side b[e].
pub open spec fn std_rows_ok(
    lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>,
    constr: Seq<Vec<Vec<Vec<f64>>>>,
    b: Seq<f64>,
    bd: Seq<int>,
    vd: Seq<int>,
    aug: Seq<real>,
) -> bool {
    let sd = std_dim(bd, vd);
    let cols = pk_len(sd);
    let rows = lmi_rows(bd) + b.len() as int;
    &&& aug.len() == rows * (cols + 1)
    &&& forall|i: int, e: int, z: Seq<real>| 0 <= i < bd.len() && 0 <= e < bd[i] * bd[i] && z.len() == cols
            ==> #[trigger] row_dot_r(aug, rows, row_off(bd, i) + e, z)
                == maps_row_r(lmis[i].0@, block_sum(bd), vd, e, z) - sym_entry(z, block_off(bd, i) + lmi_row_lo(bd[i], e), block_off(bd, i) + lmi_row_hi(bd[i], e))
    &&& forall|i: int, e: int| 0 <= i < bd.len() && 0 <= e < bd[i] * bd[i]
            ==> #[trigger] cm_entry(aug, rows, row_off(bd, i) + e, cols) == rv(lmis[i].1@[e])
    &&& forall|e: int, z: Seq<real>| 0 <= e < b.len() && z.len() == cols
            ==> #[trigger] row_dot_r(aug, rows, lmi_rows(bd) + e, z) == maps_row_r(constr, block_sum(bd), vd, e, z)
    &&& forall|e: int| 0 <= e < b.len()
            ==> #[trigger] cm_entry(aug, rows, lmi_rows(bd) + e, cols) == rv(b[e])
}

/// Entry e = c * k + r of a column-major k x k matrix: the smaller and the larger of (r, c).
pub open spec fn lmi_row_lo(k: int, e: int) -> int {
    if e % k <= e / k { e % k } else { e / k }
}

pub open spec fn lmi_row_hi(k: int, e: int) -> int {
    if e % k <= e / k { e / k } else { e % k }
}

/// The no-LMI case: the rows of the augmented matrix (m rows, pk_len(d) + 1 columns) read A(Z) with
/// right-hand side b, for the single user variable Z of order d.
pub open spec fn single_rows_ok(constr: Seq<Vec<Vec<f64>>>, b: Seq<f64>, d: int, aug: Seq<real>) -> bool {
    let cols = pk_len(d);
    let rows = b.len() as int;
    &&& aug.len() == rows * (cols + 1)
    &&& forall|e: int, z: Seq<real>| 0 <= e < rows && z.len() == cols
            ==> #[trigger] row_dot_r(aug, rows, e, z) == dot_r(symrow_r(constr, e, d), z)
    &&& forall|e: int| 0 <= e < rows ==> #[trigger] cm_entry(aug, rows, e, cols) == rv(b[e])
}

/* ---------- further sum identities ---------- */

/// <w, a - b> = <w, a> - <w, b>.
pub proof fn lemma_dot_sub_right(w: Seq<real>, a: Seq<real>, b: Seq<real>)
    requires
        w.len() == a.len() == b.len(),
    ensures
        dot_r(w, vsub_r(a, b)) == dot_r(w, a) - dot_r(w, b),
{
    let d = vsub_r(a, b);
    lemma_dot_sum_fn(w, d);
    lemma_dot_sum_fn(w, a);
    lemma_dot_sum_fn(w, b);
    let f = |t: int| w[t] * d[t];
    let fa = |t: int| w[t] * a[t];
    let fb = |t: int| w[t] * b[t];
    let nb = |t: int| -fb(t);
    assert forall|t: int| 0 <= t < w.len() implies #[trigger] f(t) == fa(t) + nb(t) by {
        assert(w[t] * (a[t] - b[t]) == w[t] * a[t] - w[t] * b[t]) by (nonlinear_arith);
    }
    assert forall|t: int| 0 <= t < w.len() implies #[trigger] nb(t) == (-1.0real) * fb(t) by {}
    lemma_sum_fn_add(fa, nb, f, w.len());
    lemma_sum_fn_scale(fb, nb, -1.0real, w.len());
}

/// pk_len(n) - pk_len(n - k) = k (n - k) + pk_len(k): the packed entries of the last k columns.
pub proof fn lemma_pk_len_diff(n: int, k: int)
    requires
        0 <= k <= n,
    ensures
        pk_len(n) == pk_len(n - k) + k * (n - k) + pk_len(k),
    decreases k,
{
    if k == 0 {
        lemma_pk_col_start_zero();
    } else {
        lemma_pk_len_diff(n, k - 1);
        lemma_pk_col_start_step(n - k);
        lemma_pk_col_start_step(k - 1);
        assert((k - 1) * (n - (k - 1)) + (n - k + 1) == k * (n - k + 1)) by (nonlinear_arith);
        assert(k * (n - k + 1) - k == k * (n - k)) by (nonlinear_arith);
    }
}

/// Number of rows of the first b blocks of a block list (each block carries its right-hand side).
pub open spec fn rows_upto(blocks: Seq<(Vec<f64>, Vec<f64>)>, b: int) -> int
    decreases b,
{
    if b <= 0 { 0 } else { rows_upto(blocks, b - 1) + blocks[b - 1].1@.len() }
}

pub proof fn lemma_rows_upto_monotone(blocks: Seq<(Vec<f64>, Vec<f64>)>, b1: int, b2: int)
    requires
        b1 <= b2,
    ensures
        rows_upto(blocks, b1) <= rows_upto(blocks, b2),
    decreases b2 - b1,
{
    if b1 < b2 {
        lemma_rows_upto_monotone(blocks, b1, b2 - 1);
    }
}

/// When block i has bd[i]^2 rows, the row counts are the LMI row offsets.
pub proof fn lemma_rows_upto_sq(blocks: Seq<(Vec<f64>, Vec<f64>)>, bd: Seq<int>, b: int)
    requires
        0 <= b <= bd.len(),
        bd.len() <= blocks.len(),
        forall|i: int| 0 <= i < bd.len() ==> (#[trigger] blocks[i]).1@.len() == bd[i] * bd[i],
    ensures
        rows_upto(blocks, b) == row_off(bd, b),
    decreases b,
{
    if b > 0 {
        lemma_rows_upto_sq(blocks, bd, b - 1);
    }
}

/// The integer square root is unique.
pub proof fn lemma_isqrt_unique(k: int, x: int)
    requires
        is_isqrt(k, x),
    ensures
        isqrt_r(x) == k,
{
    let q = isqrt_r(x);
    assert(is_isqrt(q, x));
    assert(q < k ==> (q + 1) * (q + 1) <= k * k) by (nonlinear_arith)
        requires q >= 0;
    assert(q > k ==> (k + 1) * (k + 1) <= q * q) by (nonlinear_arith)
        requires k >= 0;
}

/// The block triangles of the first j blocks fit in the triangle of their total size.
pub proof fn lemma_block_triangles_le(bs: Seq<int>, j: int)
    requires
        0 <= j <= bs.len(),
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] bs[i] >= 0,
    ensures
        sum_fn_int(|i: int| pk_len(bs[i]), j as nat) <= pk_len(sum_upto(bs, j)),
        sum_fn_int(|i: int| pk_len(bs[i]), j as nat) >= 0,
    decreases j,
{
    lemma_pk_col_start_zero();
    if j > 0 {
        lemma_block_triangles_le(bs, j - 1);
        let (s0, k) = (sum_upto(bs, j - 1), bs[j - 1]);
        lemma_sum_upto_monotone(bs, 0, j - 1);
        lemma_pk_len_diff(s0 + k, k);
        assert(k * (s0 + k - k) >= 0);
        lemma_pk_col_start_monotone(0, k);
    }
}

/// The largest magnitude of a sequence dominates each entry.
pub proof fn lemma_abs_max_entry(v: Seq<real>, t: int)
    requires
        0 <= t < v.len(),
    ensures
        abs_max_r(v) >= abs_r(v[t]),
{
    let f = |i: int| abs_r(v[i]);
    assert(abs_max_r(v) == max0_fn_r(f, v.len()));
    lemma_max0_fn_bound(f, v.len());
    assert(f(t) == abs_r(v[t]));
}

/// A prefix sum over a concatenation that stays within the first part.
pub proof fn lemma_sum_upto_prefix(a: Seq<int>, b: Seq<int>, j: int)
    requires
        0 <= j <= a.len(),
    ensures
        sum_upto(a + b, j) == sum_upto(a, j),
    decreases j,
{
    if j > 0 {
        lemma_sum_upto_prefix(a, b, j - 1);
    }
}

/// The sizes of the first s + 2n diagonal blocks add up to the LMI blocks plus two copies of the
/// first n user variables.
pub proof fn lemma_block_sizes_sum(bd: Seq<int>, vd: Seq<int>, n: int)
    requires
        0 <= n <= vd.len(),
    ensures
        sum_upto(block_sizes(bd, vd), bd.len() + 2 * n) == var_off(block_sum(bd), vd, n),
    decreases n,
{
    let bs = block_sizes(bd, vd);
    let s = bd.len() as int;
    if n == 0 {
        let tail = Seq::new(2 * vd.len(), |j: int| var_block_size(vd, j));
        assert(bs == bd + tail);
        lemma_sum_upto_prefix(bd, tail, s);
    } else {
        lemma_block_sizes_sum(bd, vd, n - 1);
        assert(bs[s + 2 * (n - 1)] == vd[n - 1]);
        assert(bs[s + 2 * (n - 1) + 1] == vd[n - 1]);
        assert(sum_upto(bs, s + 2 * n) == sum_upto(bs, s + 2 * n - 1) + bs[s + 2 * n - 1]);
        assert(sum_upto(bs, s + 2 * n - 1) == sum_upto(bs, s + 2 * (n - 1)) + bs[s + 2 * (n - 1)]);
        assert(sum_upto(vd, n) == sum_upto(vd, n - 1) + vd[n - 1]);
    }
}

/// With nonempty LMI blocks, there are at least as many LMI rows as LMIs.
pub proof fn lemma_lmi_rows_ge(bd: Seq<int>, i: int)
    requires
        0 <= i <= bd.len(),
        forall|j: int| 0 <= j < bd.len() ==> #[trigger] bd[j] >= 1,
    ensures
        row_off(bd, i) >= i,
    decreases i,
{
    if i > 0 {
        lemma_lmi_rows_ge(bd, i - 1);
        assert(bd[i - 1] * bd[i - 1] >= 1) by (nonlinear_arith)
            requires bd[i - 1] >= 1;
    }
}

/// The rows of a flattened MatrixSym (m rows, pk_len(d) columns) followed by the right-hand side
/// form the augmented single-variable system.
pub proof fn lemma_single_rows(constr: Seq<Vec<Vec<f64>>>, b: Seq<f64>, d: int, flat: Seq<f64>)
    requires
        d >= 0,
        constr.len() == b.len(),
        flat.len() == b.len() * pk_len(d),
        forall|r: int, i: int, j: int| 0 <= r < b.len() && 0 <= j <= i < d
            ==> rv(#[trigger] flat[cm_idx(b.len() as int, r, pk_idx(j, i))]) == rv(constr[r]@[i]@[j]),
    ensures
        single_rows_ok(constr, b, d, rvs(flat) + rvs(b)),
{
    let (m, cols) = (b.len() as int, pk_len(d));
    let aug = rvs(flat) + rvs(b);
    lemma_pk_col_start_monotone(0, d);
    assert(m * cols + m == m * (cols + 1)) by (nonlinear_arith);
    assert(cols * m == m * cols) by (nonlinear_arith);
    assert forall|e: int, z: Seq<real>| 0 <= e < m && z.len() == cols
        implies #[trigger] row_dot_r(aug, m, e, z) == dot_r(symrow_r(constr, e, d), z) by {
        let w = symrow_r(constr, e, d);
        lemma_dot_sum_fn(w, z);
        let f = |n: int| constr_entry(aug, m, e, n) * z[n];
        let g = |n: int| w[n] * z[n];
        assert forall|n: int| 0 <= n < cols implies #[trigger] f(n) == g(n) by {
            lemma_cm_idx_bound(m, e, n, cols);
            lemma_pk_unpack(n);
            let (j, i) = pk_unpack(n);
            if i >= d {
                lemma_pk_col_start_monotone(d, i);
            }
            assert(aug[cm_idx(m, e, n)] == rv(flat[cm_idx(m, e, pk_idx(j, i))]));
        }
        lemma_sum_fn_ext(f, g, cols as nat);
    }
    assert forall|e: int| 0 <= e < m implies #[trigger] cm_entry(aug, m, e, cols) == rv(b[e]) by {
        assert(cm_idx(m, e, cols) == cols * m + e);
    }
}

/// A row of the stacked matrix whose entries are those of row e of a block has the block's row
/// dot product.
pub proof fn lemma_stacked_row(aug: Seq<real>, mat: Seq<real>, blk: Seq<real>, rows: int, brows: int, roff: int, e: int, z: Seq<real>)
    requires
        0 <= e < brows,
        0 <= roff,
        roff + brows <= rows,
        mat.len() == rows * z.len(),
        blk.len() == brows * z.len(),
        forall|t: int| 0 <= t < mat.len() ==> #[trigger] aug[t] == mat[t],
        forall|t: int| 0 <= t < z.len() ==> mat[#[trigger] cm_idx(rows, roff + e, t)] == blk[cm_idx(brows, e, t)],
    ensures
        row_dot_r(aug, rows, roff + e, z) == row_dot_r(blk, brows, e, z),
{
    let f = |n: int| constr_entry(aug, rows, roff + e, n) * z[n];
    let g = |n: int| constr_entry(blk, brows, e, n) * z[n];
    assert(rows * z.len() == z.len() * rows);
    assert forall|n: int| 0 <= n < z.len() implies #[trigger] f(n) == g(n) by {
        lemma_cm_idx_bound(rows, roff + e, n, z.len() as int);
    }
    lemma_sum_fn_ext(f, g, z.len());
}

/// The entry e of a column-major k x k block addresses the rows e % k and e / k, both below k.
pub proof fn lemma_lmi_row_bounds(k: int, e: int)
    requires
        k >= 1,
        0 <= e < k * k,
    ensures
        0 <= e % k < k,
        0 <= e / k < k,
        0 <= lmi_row_lo(k, e) <= lmi_row_hi(k, e) < k,
{
    lemma_fundamental_div_mod(e, k);
    lemma_mod_bound(e, k);
    let (q, r) = (e / k, e % k);
    assert(q >= 0);
    assert(q < k) by (nonlinear_arith)
        requires k >= 1, e == k * q + r, 0 <= r < k, e < k * k;
}

/// The entry c k + r of a column-major k x k block addresses the rows r and c.
pub proof fn lemma_lmi_row_of(k: int, c: int, r: int)
    requires
        k >= 1,
        0 <= r < k,
        0 <= c < k,
    ensures
        (c * k + r) / k == c,
        (c * k + r) % k == r,
        lmi_row_lo(k, c * k + r) == (if r <= c { r } else { c }),
        lmi_row_hi(k, c * k + r) == (if r <= c { c } else { r }),
{
    lemma_fundamental_div_mod_converse(c * k + r, k, c, r);
}

/// A sequence of sizes at least 1 sums to at least its length.
pub proof fn lemma_sum_upto_ge(s: Seq<int>, j: int)
    requires
        0 <= j <= s.len(),
        forall|i: int| 0 <= i < s.len() ==> #[trigger] s[i] >= 1,
    ensures
        sum_upto(s, j) >= j,
    decreases j,
{
    if j > 0 {
        lemma_sum_upto_ge(s, j - 1);
    }
}

/// Every entry of block_sizes(bd, vd) is one of the given sizes.
pub proof fn lemma_block_sizes_entry(bd: Seq<int>, vd: Seq<int>, j: int)
    requires
        0 <= j < bd.len() + 2 * vd.len(),
    ensures
        j < bd.len() ==> block_sizes(bd, vd)[j] == bd[j],
        j >= bd.len() ==> 0 <= (j - bd.len()) / 2 < vd.len() && block_sizes(bd, vd)[j] == vd[(j - bd.len()) / 2],
{
    if j >= bd.len() {
        let i = j - bd.len();
        lemma_fundamental_div_mod(i, 2);
        lemma_mod_bound(i, 2);
    }
}

/* ---------- the dense rows vanish on the zero pattern ---------- */

/// A row applied to the unit packed vector e_t0 reads its entry t0.
pub proof fn lemma_row_dot_basis(m: Seq<real>, row_dim: int, i: int, t0: int, len: nat)
    requires
        0 <= t0 < len,
    ensures
        row_dot_r(m, row_dim, i, std_basis_r(t0, len)) == constr_entry(m, row_dim, i, t0),
{
    let z = std_basis_r(t0, len);
    let f = |n: int| constr_entry(m, row_dim, i, n) * z[n];
    assert forall|n: int| 0 <= n < len && n != t0 implies #[trigger] f(n) == 0.0real by {
        assert(z[n] == 0.0real);
    }
    lemma_sum_fn_single(f, len, t0);
    assert(f(t0) == constr_entry(m, row_dim, i, t0) * 1.0real);
}

/// The packed dot product vanishes when the right factor is zero.
pub proof fn lemma_dot_zero_right(v: Seq<real>, z: Seq<real>)
    requires
        v.len() == z.len(),
        forall|t: int| 0 <= t < z.len() ==> #[trigger] z[t] == 0.0real,
    ensures
        dot_r(v, z) == 0.0real,
{
    lemma_dot_comm(v, z);
    lemma_dot_zero(z, v);
}

/// A packed index below pk_len(d) addresses a column below d.
pub proof fn lemma_pk_idx_lt_len(r: int, c: int, d: int)
    requires
        0 <= r <= c,
        d >= 0,
        pk_idx(r, c) < pk_len(d),
    ensures
        c < d,
{
    if c >= d {
        lemma_pk_col_start_monotone(d, c);
    }
}

/// A position inside one diagonal block of the partition bs is not off-block-diagonal.
pub proof fn lemma_block_not_zero_region(bs: Seq<int>, j: int, r: int, c: int)
    requires
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] bs[i] >= 0,
        0 <= j < bs.len(),
        sum_upto(bs, j) <= r < sum_upto(bs, j + 1),
        sum_upto(bs, j) <= c < sum_upto(bs, j + 1),
    ensures
        !in_zero_region(bs, r, c),
{
    if in_zero_region(bs, r, c) {
        let j2 = choose|j2: int| 0 <= j2 < bs.len() && #[trigger] sum_upto(bs, j2) <= r < sum_upto(bs, j2 + 1) <= c;
        if j2 < j {
            lemma_sum_upto_monotone(bs, j2 + 1, j);
        } else if j2 > j {
            lemma_sum_upto_monotone(bs, j + 1, j2);
        }
    }
}

/// The unit packed vector at an off-block-diagonal position (r0, c0) reads as zero on every
/// diagonal block of the partition bs.
pub proof fn lemma_blk_basis_zero(bs: Seq<int>, j: int, r0: int, c0: int, sd: int, t: int)
    requires
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] bs[i] >= 0,
        0 <= j < bs.len(),
        sum_upto(bs, bs.len() as int) <= sd,
        in_zero_region(bs, r0, c0),
        0 <= r0 <= c0 < sd,
        0 <= t < pk_len(bs[j]),
    ensures
        blk_r(std_basis_r(pk_idx(r0, c0), pk_len(sd) as nat), sum_upto(bs, j), bs[j])[t] == 0.0real,
{
    let (st, d) = (sum_upto(bs, j), bs[j]);
    let z = std_basis_r(pk_idx(r0, c0), pk_len(sd) as nat);
    let (r, c) = (pk_unpack(t).0, pk_unpack(t).1);
    lemma_pk_unpack(t);
    lemma_pk_idx_lt_len(r, c, d);
    lemma_sum_upto_monotone(bs, 0, j);
    lemma_sum_upto_monotone(bs, j + 1, bs.len() as int);
    assert(sum_upto(bs, j + 1) == st + d);
    lemma_pk_idx_bound(st + r, st + c, sd);
    lemma_pk_idx_bound(r0, c0, sd);
    assert(blk_r(z, st, d)[t] == z[pk_idx(st + r, st + c)]);
    lemma_pk_idx_injective(st + r, st + c, r0, c0);
    if st + r == r0 && st + c == c0 {
        lemma_block_not_zero_region(bs, j, r0, c0);
    }
}

/// The user maps applied to the unit packed vector at an off-block-diagonal position vanish:
/// every user variable is read from diagonal blocks, where that vector is zero.
pub proof fn lemma_maps_row_basis_zero(maps: Seq<Vec<Vec<Vec<f64>>>>, bd: Seq<int>, vd: Seq<int>, e: int, r0: int, c0: int)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        in_zero_region(block_sizes(bd, vd), r0, c0),
        0 <= r0 <= c0 < std_dim(bd, vd),
    ensures
        maps_row_r(maps, block_sum(bd), vd, e, std_basis_r(pk_idx(r0, c0), pk_len(std_dim(bd, vd)) as nat)) == 0.0real,
{
    let bs = block_sizes(bd, vd);
    let bsum = block_sum(bd);
    let sd = std_dim(bd, vd);
    let z = std_basis_r(pk_idx(r0, c0), pk_len(sd) as nat);
    let s = bd.len() as int;
    assert(bs.len() == s + 2 * vd.len());
    lemma_block_sizes_sum(bd, vd, vd.len() as int);
    assert forall|i: int| 0 <= i < bs.len() implies #[trigger] bs[i] >= 0 by {
        lemma_block_sizes_entry(bd, vd, i);
    }
    let f = |n: int| map_term_r(maps, bsum, vd, e, z, n);
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] f(n) == 0.0real by {
        let (st, d) = (var_off(bsum, vd, n), vd[n]);
        let (j1, j2) = (s + 2 * n, s + 2 * n + 1);
        lemma_block_sizes_sum(bd, vd, n);
        lemma_block_sizes_entry(bd, vd, j1);
        lemma_block_sizes_entry(bd, vd, j2);
        assert(bs[j1] == d && bs[j2] == d);
        assert(sum_upto(bs, j2) == st + d);
        lemma_pk_col_start_monotone(0, d);
        let zv = zvar_r(z, st, d);
        assert forall|t: int| 0 <= t < pk_len(d) implies #[trigger] zv[t] == 0.0real by {
            lemma_blk_basis_zero(bs, j1, r0, c0, sd, t);
            lemma_blk_basis_zero(bs, j2, r0, c0, sd, t);
        }
        lemma_dot_zero_right(symrow_r(maps[n]@, e, d), zv);
    }
    lemma_sum_fn_ext(f, |n: int| 0.0real, vd.len());
    lemma_sum_fn_zero(vd.len());
}

/// The LMI entry read by an LMI row lies in that LMI's diagonal block, where the unit packed
/// vector at an off-block-diagonal position is zero.
pub proof fn lemma_lmi_entry_basis_zero(bd: Seq<int>, vd: Seq<int>, i0: int, e: int, r0: int, c0: int)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        0 <= i0 < bd.len(),
        0 <= e < bd[i0] * bd[i0],
        in_zero_region(block_sizes(bd, vd), r0, c0),
        0 <= r0 <= c0 < std_dim(bd, vd),
    ensures
        sym_entry(std_basis_r(pk_idx(r0, c0), pk_len(std_dim(bd, vd)) as nat),
            block_off(bd, i0) + lmi_row_lo(bd[i0], e), block_off(bd, i0) + lmi_row_hi(bd[i0], e)) == 0.0real,
{
    let bs = block_sizes(bd, vd);
    let sd = std_dim(bd, vd);
    let z = std_basis_r(pk_idx(r0, c0), pk_len(sd) as nat);
    let k = bd[i0];
    assert(k >= 1) by (nonlinear_arith)
        requires k >= 0, 0 <= e < k * k;
    lemma_lmi_row_bounds(k, e);
    let (p, q) = (block_off(bd, i0) + lmi_row_lo(k, e), block_off(bd, i0) + lmi_row_hi(k, e));
    let tail = Seq::new(2 * vd.len(), |j: int| var_block_size(vd, j));
    assert(bs == bd + tail);
    lemma_sum_upto_prefix(bd, tail, i0);
    lemma_sum_upto_prefix(bd, tail, i0 + 1);
    assert(bs[i0] == k);
    assert(sum_upto(bs, i0 + 1) == sum_upto(bs, i0) + k);
    assert forall|i: int| 0 <= i < bs.len() implies #[trigger] bs[i] >= 0 by {
        lemma_block_sizes_entry(bd, vd, i);
    }
    lemma_block_sizes_sum(bd, vd, vd.len() as int);
    lemma_sum_upto_monotone(bs, 0, i0);
    lemma_sum_upto_monotone(bs, i0 + 1, bs.len() as int);
    lemma_pk_idx_bound(p, q, sd);
    lemma_pk_idx_bound(r0, c0, sd);
    assert(sym_entry(z, p, q) == z[pk_idx(p, q)]);
    lemma_pk_idx_injective(p, q, r0, c0);
    if p == r0 && q == c0 {
        lemma_block_not_zero_region(bs, i0, r0, c0);
    }
}

/// Every LMI row index below lmi_rows(bd) restricted to the first n LMIs is row e of some LMI i0 < n.
pub proof fn lemma_lmi_row_decomp(bd: Seq<int>, n: int, i: int)
    requires
        0 <= n <= bd.len(),
        0 <= i < sq_sum_upto(bd, n),
    ensures
        exists|i0: int| 0 <= i0 < n && #[trigger] row_off(bd, i0) <= i < row_off(bd, i0) + bd[i0] * bd[i0],
    decreases n,
{
    if i < sq_sum_upto(bd, n - 1) {
        lemma_lmi_row_decomp(bd, n - 1, i);
    } else {
        assert(row_off(bd, n - 1) == sq_sum_upto(bd, n - 1));
        assert(row_off(bd, n - 1) <= i < row_off(bd, n - 1) + bd[n - 1] * bd[n - 1]);
    }
}

/// The stacked standard-form rows have no coefficient at an off-block-diagonal position: the
/// LMI rows and the equality rows read the variable through its diagonal blocks only.
pub proof fn lemma_std_rows_vanish(
    lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>,
    constr: Seq<Vec<Vec<Vec<f64>>>>,
    b: Seq<f64>,
    bd: Seq<int>,
    vd: Seq<int>,
    aug: Seq<real>,
    r0: int,
    c0: int,
)
    requires
        std_rows_ok(lmis, constr, b, bd, vd, aug),
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        in_zero_region(block_sizes(bd, vd), r0, c0),
        0 <= r0 <= c0 < std_dim(bd, vd),
    ensures
        forall|i: int| 0 <= i < lmi_rows(bd) + b.len()
            ==> #[trigger] cm_entry(aug, lmi_rows(bd) + b.len() as int, i, pk_idx(r0, c0)) == 0.0real,
{
    let sd = std_dim(bd, vd);
    let cols = pk_len(sd);
    let rows = lmi_rows(bd) + b.len() as int;
    let bsum = block_sum(bd);
    let t0 = pk_idx(r0, c0);
    lemma_pk_idx_bound(r0, c0, sd);
    let z = std_basis_r(t0, cols as nat);
    assert forall|i: int| 0 <= i < rows implies #[trigger] cm_entry(aug, rows, i, t0) == 0.0real by {
        lemma_row_dot_basis(aug, rows, i, t0, cols as nat);
        if i < lmi_rows(bd) {
            lemma_lmi_row_decomp(bd, bd.len() as int, i);
            let i0 = choose|i0: int| 0 <= i0 < bd.len() && #[trigger] row_off(bd, i0) <= i < row_off(bd, i0) + bd[i0] * bd[i0];
            let e = i - row_off(bd, i0);
            assert(row_dot_r(aug, rows, row_off(bd, i0) + e, z)
                == maps_row_r(lmis[i0].0@, bsum, vd, e, z)
                    - sym_entry(z, block_off(bd, i0) + lmi_row_lo(bd[i0], e), block_off(bd, i0) + lmi_row_hi(bd[i0], e)));
            lemma_maps_row_basis_zero(lmis[i0].0@, bd, vd, e, r0, c0);
            lemma_lmi_entry_basis_zero(bd, vd, i0, e, r0, c0);
        } else {
            let e = i - lmi_rows(bd);
            assert(row_dot_r(aug, rows, lmi_rows(bd) + e, z) == maps_row_r(constr, bsum, vd, e, z));
            lemma_maps_row_basis_zero(constr, bd, vd, e, r0, c0);
        }
    }
}

} // verus!
