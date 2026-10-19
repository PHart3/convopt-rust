// Solver setup specification (not trusted).
//
// sdpad (sdp_altdir.rs) prepares the standard-form problem (C, A, b) of
// sdp_to_standard for the Wen-Goldfarb-Yin iteration:
//
//   - the objective is reweighted so that it acts through the Frobenius
//     product, <C_F, Z>_F = C . Z, by halving the off-diagonal packed entries
//     (`half_off_diag`, `lemma_frob_half_dot`);
//   - the packed Gram matrix G of the dense constraint rows is formed
//     (constraint_gram, L04);
//   - the diagonal-block partition of Z that the termination test checks for
//     positive semidefiniteness is listed (`psd_partition`, `block_list_ok`);
//   - every iteration solves the block system (G, I) y = b, either directly
//     (reduced constraints) or after regularizing G by TOL * max(max_i |G_ii|, 1)
//     on the diagonal, with the residual of the unregularized system bounded
//     relative to the data (`dual_solve_ok`);
//   - the initial iterates are Z = I and S = 0 (`initial_iterates_ok`).

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_mod_bound, lemma_div_multiples_vanish};
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::real_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::ldlt_spec::blocks_fit;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::reduce_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_matrix::*;
#[cfg(verus_keep_ghost)]
use crate::sdp_spec::*;
#[cfg(verus_keep_ghost)]
use crate::sdp::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- the solver's input and the standard form ---------- */

/// The contract of sdp_to_standard for either kind of input.
pub open spec fn std_form_ok(sdp: &SDP, res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>), check: bool) -> bool {
    if sdp.lmi@.len() == 0 { std_form_single_ok(sdp, res, check) } else { std_form_lmi_ok(sdp, res, check) }
}

/// The number of rows of the stacked constraint system before any reduction.
pub open spec fn total_rows(sdp: &SDP) -> int {
    if sdp.lmi@.len() == 0 {
        sdp.constraint.1@.len() as int
    } else {
        lmi_rows(bdims_r(sdp.lmi@)) + sdp.constraint.1@.len() as int
    }
}

/// The solver's input requirements: those of the standard-form conversion, and the
/// constraint row count fitting the packed index arithmetic of the Gram matrix.
pub open spec fn sdpad_input_ok(sdp: &SDP, check: bool) -> bool {
    &&& sdp.lmi@.len() == 0 ==> sdp_single_input_ok(sdp)
            && (check && sdp.constraint.0@.len() == 1 ==> single_constraint_nonzero(sdp))
    &&& sdp.lmi@.len() >= 1 ==> sdp_lmi_input_ok(sdp)
    &&& dim_fits(total_rows(sdp))
}

/// The reduced constraint system has at most as many rows as the stacked system
/// (exactly as many without reduction).
pub proof fn lemma_reduced_rows(aug: Seq<real>, mat: Seq<f64>, point: Seq<f64>, rows: int, total: int, dim: int, check: bool)
    requires
        rows >= 0,
        total >= 0,
        dim >= 0,
        mat.len() == rows * pk_len(dim),
        point.len() >= rows,
        aug.len() == total * (pk_len(dim) + 1),
        if check { reduced_from(aug, result_aug(mat, point, rows), dim) } else { result_aug(mat, point, rows) == aug },
    ensures
        rows <= total,
{
    lemma_pk_col_start_zero();
    lemma_pk_col_start_monotone(0, dim);
    let cd = red_col_dim(dim);
    let res = result_aug(mat, point, rows);
    assert(res.len() == rows * pk_len(dim) + rows);
    assert(rows * pk_len(dim) + rows == rows * (pk_len(dim) + 1)) by (nonlinear_arith);
    if check {
        let e = choose|e: Seq<real>| reduced_system(aug, e, result_aug(mat, point, rows), dim);
        let pivots = choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(aug, e, result_aug(mat, point, rows), dim, pivots);
        let rank = pivots.len() as int;
        // res.len() == cd * rank == rows * cd, hence rank == rows
        assert(rank == rows) by (nonlinear_arith)
            requires cd >= 1, cd * rank == rows * cd;
        // the echelon form has rank <= aug.len() / cd == total
        assert(aug.len() as int == cd * total) by (nonlinear_arith)
            requires aug.len() as int == total * cd;
        lemma_div_multiples_vanish(total, cd);
    } else {
        assert(rows == total) by (nonlinear_arith)
            requires cd >= 1, rows * cd == total * cd;
    }
}

/// The facts of the standard form that the solver setup relies on: the order d >= 1 of the
/// variable, the fitting of d and of the number of dense constraint rows, the lengths of the
/// returned data, the zero pattern, the zero right-hand sides of the zero constraints, and the
/// block sizes (whose partition has order d).
pub proof fn lemma_std_form_facts(sdp: &SDP, res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>), check: bool)
    requires
        sdpad_input_ok(sdp, check),
        std_form_ok(sdp, res, check),
    ensures
        res.3 >= 1,
        dim_fits(res.3 as int),
        res.0@.len() == pk_len(res.3 as int),
        res.2@.len() >= res.1.1@.len(),
        res.1.0@.len() == (res.2@.len() - res.1.1@.len()) * pk_len(res.3 as int),
        dim_fits(res.2@.len() - res.1.1@.len()),
        valid_zero_list(res.1.1@, res.3 as int),
        forall|q: int| res.2@.len() - res.1.1@.len() <= q < res.2@.len() ==> rv(#[trigger] res.2@[q]) == 0.0real,
        forall|i: int| 0 <= i < res.5@.len() ==> #[trigger] res.5@[i] >= 1,
        forall|n: int| 0 <= n < res.4@.len() ==> #[trigger] res.4@[n] >= 1,
        res.4@.len() + res.5@.len() > 0 ==> std_dim(dims_of(res.5@), dims_of(res.4@)) == res.3,
        block_sum(dims_of(res.5@)) <= res.3,
{
    let d = res.3 as int;
    let cols = pk_len(d);
    let nz = res.1.1@.len() as int;
    let rows = res.2@.len() as int - nz;
    lemma_pk_col_start_zero();
    if sdp.lmi@.len() == 0 {
        let d0 = sdp.objective@[0]@.len() as int;
        let m = sdp.constraint.1@.len() as int;
        assert(dims_of(res.5@) =~= Seq::<int>::empty());
        if sdp.constraint.0@.len() == 0 {
            lemma_dim_fits_monotone(0, d);
        } else {
            let aug = choose|aug: Seq<real>| #[trigger] single_rows_ok(sdp.constraint.0@[0]@, sdp.constraint.1@, d0, aug)
                && (if check { reduced_from(aug, result_aug(res.1.0@, res.2@, rows), d0) } else { result_aug(res.1.0@, res.2@, rows) == aug });
            lemma_reduced_rows(aug, res.1.0@, res.2@, rows, m, d, check);
            lemma_dim_fits_monotone(rows, m);
        }
    } else {
        let nv = n_vars(sdp);
        let bd = bdims_r(sdp.lmi@);
        let vd = vdims_r(sdp.lmi@, nv);
        let sd = std_dim(bd, vd);
        let m = sdp.constraint.1@.len() as int;
        let lr = lmi_rows(bd);
        lemma_sum_upto_ge(bd, bd.len() as int);
        lemma_sum_upto_ge(vd, nv);
        lemma_lmi_rows_ge(bd, bd.len() as int);
        let aug = choose|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
            && (if check { reduced_from(aug, result_aug(res.1.0@, res.2@, rows), sd) } else { result_aug(res.1.0@, res.2@, rows) == aug });
        assert(aug.len() == (lr + m) * (cols + 1));
        lemma_reduced_rows(aug, res.1.0@, res.2@, rows, lr + m, d, check);
        lemma_dim_fits_monotone(rows, lr + m);
        assert forall|i: int| 0 <= i < res.5@.len() implies #[trigger] res.5@[i] >= 1 by {
            assert(dims_of(res.5@)[i] == bd[i]);
        }
        assert forall|n: int| 0 <= n < res.4@.len() implies #[trigger] res.4@[n] >= 1 by {
            assert(dims_of(res.4@)[n] == vd[n]);
        }
    }
}

/* ---------- the Frobenius-weighted objective ---------- */

/// cf is c with every off-diagonal packed entry halved.
pub open spec fn half_off_diag(cf: Seq<real>, c: Seq<real>, dim: int) -> bool {
    &&& cf.len() == pk_len(dim)
    &&& c.len() == pk_len(dim)
    &&& forall|r: int, k: int| 0 <= r <= k < dim
            ==> #[trigger] cf[pk_idx(r, k)] == (if r == k { c[pk_idx(r, k)] } else { 0.5real * c[pk_idx(r, k)] })
}

/// <cf, z>_F = c . z: the halved off-diagonal entries compensate the double count of the
/// Frobenius product, so the weighted objective acts as the packed objective.
pub proof fn lemma_frob_half_dot(cf: Seq<real>, c: Seq<real>, z: Seq<real>, dim: int)
    requires
        dim >= 0,
        half_off_diag(cf, c, dim),
        z.len() == pk_len(dim),
    ensures
        frob_r(cf, z, dim) == dot_r(c, z),
{
    lemma_frob_tri(cf, z, dim as nat);
    lemma_dot_sum_fn(c, z);
    let g = |t: int| c[t] * z[t];
    lemma_tri_sum_packed(g, dim as nat);
    let f = |r: int, k: int| pk_wt(r, k) * (cf[pk_idx(r, k)] * z[pk_idx(r, k)]);
    let h = |r: int, k: int| g(pk_idx(r, k));
    assert forall|r: int, k: int| 0 <= r <= k < dim implies #[trigger] f(r, k) == h(r, k) by {
        lemma_pk_idx_bound(r, k, dim);
        let (u, w) = (c[pk_idx(r, k)], z[pk_idx(r, k)]);
        if r == k {
            assert(1.0real * (u * w) == u * w) by (nonlinear_arith);
        } else {
            assert(2.0real * ((0.5real * u) * w) == u * w) by (nonlinear_arith);
        }
    }
    lemma_tri_sum_ext(f, h, dim as nat);
}

/* ---------- the PSD block partition ---------- */

/// The diagonal-block partition of the standard-form variable that the solver checks for
/// positive semidefiniteness: the LMI blocks and the two blocks of every user variable
/// (block_sizes, L16), or the whole variable for a single-variable problem.
pub open spec fn psd_partition(bd: Seq<int>, vd: Seq<int>, dim: int) -> Seq<int> {
    if bd.len() + vd.len() == 0 { seq![dim] } else { block_sizes(bd, vd) }
}

/// list is the (start, size) list of the consecutive diagonal blocks of sizes bs.
pub open spec fn block_list_ok(list: Seq<(usize, usize)>, bs: Seq<int>) -> bool {
    &&& list.len() == bs.len()
    &&& forall|j: int| 0 <= j < bs.len()
            ==> (#[trigger] list[j]).0 as int == sum_upto(bs, j) && list[j].1 as int == bs[j]
}

/// Every entry of block_sizes(bd, vd) is one of the given sizes (and nonnegative when they are).
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

/// A block list of a partition of the dim x dim variable fits the packed matrix of order dim.
pub proof fn lemma_block_list_fits(list: Seq<(usize, usize)>, bs: Seq<int>, dim: int)
    requires
        block_list_ok(list, bs),
        forall|j: int| 0 <= j < bs.len() ==> #[trigger] bs[j] >= 0,
        sum_upto(bs, bs.len() as int) <= dim,
        dim_fits(dim),
    ensures
        pk_len(dim) >= 0,
        blocks_fit(list, pk_len(dim) as nat),
{
    lemma_pk_col_start_zero();
    lemma_sum_upto_monotone(bs, 0, bs.len() as int);
    lemma_pk_col_start_monotone(0, dim);
    assert forall|b: int| 0 <= b < list.len()
        implies dim_fits((#[trigger] list[b]).0 + list[b].1) && pk_len(dim) as nat >= pk_len(list[b].0 + list[b].1) by {
        let end = sum_upto(bs, b + 1);
        assert(end == list[b].0 + list[b].1);
        lemma_sum_upto_monotone(bs, 0, b + 1);
        lemma_sum_upto_monotone(bs, b + 1, bs.len() as int);
        lemma_dim_fits_monotone(end, dim);
        lemma_pk_col_start_monotone(end, dim);
    }
}

/* ---------- the equality-system solve ---------- */

/// The diagonal shift of the Gram matrix: 0 with reduced constraints, otherwise
/// TOL * max(max_i |G_ii|, 1).
pub open spec fn reg_shift_r(g: Seq<real>, c_len: int, check: bool) -> real {
    if check { 0.0real } else { rv(TOL) * max_r(max_abs_diag_r(g, c_len), 1.0real) }
}

/// The residual test of the regularized solve: ||G y - b|| / (||G||_F ||y|| + ||b||) < 1e-9
/// (nothing is claimed when the denominator is zero).
pub open spec fn residual_ok(g: Seq<real>, c_len: int, yt: Seq<real>, bt: Seq<real>) -> bool {
    let top = dist_r(sym_mat_vec_r(g, c_len, yt), bt);
    let bottom = frob_norm_r(g, c_len) * norm_r(yt) + norm_r(bt);
    bottom != 0.0real ==> top / bottom < 0.000000001real
}

/// y solves the block system of one iteration: (G + s I) y_top = b_top with the shift s of
/// `reg_shift_r`, y_bottom = b_bottom (the multipliers of the zero constraints), and, when G
/// was regularized, the residual of the unregularized system passes the test.
pub open spec fn dual_solve_ok(g: Seq<real>, b: Seq<real>, c_len: int, check: bool, y: Seq<real>) -> bool {
    let n = b.len() as int;
    let yt = y.subrange(0, c_len);
    let bt = b.subrange(0, c_len);
    &&& y.len() == n
    &&& mat_vec(madd(sym_view(g), mscale(reg_shift_r(g, c_len, check), mident())), c_len, yt) == bt
    &&& forall|t: int| c_len <= t < n ==> #[trigger] y[t] == b[t]
    &&& !check && c_len > 0 ==> residual_ok(g, c_len, yt, bt)
}

/// A solution of the block system (A, I) x = b solves A x_top = b_top and has x_bottom = b_bottom.
pub proof fn lemma_block_ident_solve_inv(a: MatF, n: int, total: int, x: Seq<real>, b: Seq<real>)
    requires
        0 <= n <= total,
        x.len() == total,
        b.len() == total,
        mat_vec(block_ident_view(a, n), total, x) == b,
    ensures
        mat_vec(a, n, x.subrange(0, n)) == b.subrange(0, n),
        forall|t: int| n <= t < total ==> #[trigger] x[t] == b[t],
{
    let m = block_ident_view(a, n);
    let xs = x.subrange(0, n);
    assert forall|r: int| 0 <= r < n implies #[trigger] mat_vec(a, n, xs)[r] == b.subrange(0, n)[r] by {
        let f = |k: int| m(r, k) * x[k];
        let g = |k: int| a(r, k) * xs[k];
        assert forall|k: int| n <= k < total implies #[trigger] f(k) == 0.0real by {
            assert(m(r, k) == 0.0real);
        }
        lemma_sum_fn_tail_zero(f, n as nat, total as nat);
        assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {
            assert(m(r, k) == a(r, k));
            assert(xs[k] == x[k]);
        }
        lemma_sum_fn_ext(f, g, n as nat);
        assert(mat_vec(m, total, x)[r] == b[r]);
    }
    assert(mat_vec(a, n, xs) =~= b.subrange(0, n));
    assert forall|t: int| n <= t < total implies #[trigger] x[t] == b[t] by {
        let f = |k: int| m(t, k) * x[k];
        assert forall|k: int| 0 <= k < total && k != t implies #[trigger] f(k) == 0.0real by {
            assert(m(t, k) == 0.0real);
        }
        lemma_sum_fn_single(f, total as nat, t);
        assert(m(t, t) == 1.0real);
        assert(mat_vec(m, total, x)[t] == b[t]);
    }
}

/// A + 0 I = A on the block.
pub proof fn lemma_shift_zero_meq(a: MatF, dim: int)
    ensures
        meq(madd(a, mscale(0.0real, mident())), a, dim, dim),
{
    assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim
        implies #[trigger] madd(a, mscale(0.0real, mident()))(i, j) == a(i, j) by {
        if i == j {
            assert(madd(a, mscale(0.0real, mident()))(i, j) == a(i, j) + 0.0real * 1.0real);
        } else {
            assert(madd(a, mscale(0.0real, mident()))(i, j) == a(i, j) + 0.0real * 0.0real);
        }
    }
}

/// gr is g with s added to every diagonal entry of the leading dim x dim block.
pub open spec fn diag_shifted(gr: Seq<real>, g: Seq<real>, dim: int, s: real) -> bool {
    &&& gr.len() == g.len()
    &&& forall|r: int, k: int| 0 <= r <= k < dim
            ==> #[trigger] gr[pk_idx(r, k)] == g[pk_idx(r, k)] + (if r == k { s } else { 0.0real })
}

/// The symmetric view of a diagonally shifted matrix is G + s I on the block.
pub proof fn lemma_diag_shifted_view(gr: Seq<real>, g: Seq<real>, dim: int, s: real)
    requires
        diag_shifted(gr, g, dim, s),
    ensures
        meq(sym_view(gr), madd(sym_view(g), mscale(s, mident())), dim, dim),
{
    assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim
        implies #[trigger] sym_view(gr)(i, j) == madd(sym_view(g), mscale(s, mident()))(i, j) by {
        if i == j {
            assert(sym_view(gr)(i, j) == gr[pk_idx(i, i)]);
            assert(madd(sym_view(g), mscale(s, mident()))(i, j) == g[pk_idx(i, i)] + s * 1.0real);
        } else if i < j {
            assert(sym_view(gr)(i, j) == gr[pk_idx(i, j)]);
            assert(madd(sym_view(g), mscale(s, mident()))(i, j) == g[pk_idx(i, j)] + s * 0.0real);
        } else {
            assert(sym_view(gr)(i, j) == gr[pk_idx(j, i)]);
            assert(madd(sym_view(g), mscale(s, mident()))(i, j) == g[pk_idx(j, i)] + s * 0.0real);
        }
    }
}

/* ---------- the initial iterates ---------- */

/// Z = I and S = 0.
pub open spec fn initial_iterates_ok(z: Seq<real>, s: Seq<real>, dim: int) -> bool {
    &&& is_sym_ident_r(z, dim)
    &&& s.len() == pk_len(dim)
    &&& forall|t: int| 0 <= t < s.len() ==> #[trigger] s[t] == 0.0real
}

} // verus!
