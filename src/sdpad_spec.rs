// Solver specification: setup, iteration, KKT quantities and loop (not trusted).
//
// sdpad_setup (sdp_altdir.rs) prepares the standard-form problem (C, A, b) of
// sdp_to_standard for the Wen-Goldfarb-Yin iteration:
//
//   - the objective is reweighted so that it acts through the Frobenius
//     product, <C_F, Z>_F = C . Z, by halving the off-diagonal packed entries
//     (`half_off_diag`, `lemma_frob_half_dot`);
//   - the packed Gram matrix G of the dense constraint rows is formed
//     (constraint_gram, L04);
//   - the diagonal-block partition of Z that the termination test checks for
//     positive semidefiniteness is listed (`psd_partition`, `block_list_ok`);
//   - every iteration solves the block system (G, I/2) y = b, either directly
//     (reduced constraints) or after regularizing G by TOL * max(max_i |G_ii|, 1)
//     on the diagonal, with the residual of the unregularized system bounded
//     relative to the data (`dual_solve_ok`); (G, I/2) is the Gram matrix A A* of
//     the full constraint operator (`lemma_setup_full_gram`): the zero constraints
//     have squared Frobenius norm 1/2 and are orthogonal to the dense rows, which vanish on
//     the zero pattern (`lemma_setup_gram_hypotheses`);
//   - the initial iterates are Z = I and S = 0 (`initial_iterates_ok`).
//
// Each iteration then performs the Wen-Goldfarb-Yin update for the penalty
// mu and the step rho (sdpad.pdf, (7), (9), (10), (26)):
//
//   y   solves  (A A*) y = A(C - S) + mu (b - A(Z))      (`wgy_rhs_r`, `dual_solve_ok`, `lemma_wgy_dual_gram`)
//   V   = C - A*(y) - mu Z                               (`is_dual_slack`, `is_shifted_slack`)
//   S'  = V_+ = Q diag(lambda_+) Q^T                     (`is_pos_part`)
//   Z'  = (1 - rho) Z + (rho / mu) (S' - V)              (`is_primal_step`)
//
// where V = Q B Q^T is the eigensolver's approximate decomposition (Q
// orthogonal, lambda = diag(B), ||B - diag(B)||_F <= eps) and lambda_+ keeps
// the entries lambda_k > 0 of poseigendecomp; nothing relates S' to the exact
// PSD projection of V. The penalty is adapted from the ratio of the primal
// and dual infeasibilities (`penalty_update_ok`).
//
// After each update the residuals and KKT quantities of the iterate are
// computed (`kkt_ok`):
//
//   pinf  = ||A(Z) - b|| / (1 + ||b||)
//   dinf  = ||(C - A*(y)) - S||_F / (1 + ||C||_1)
//   prim  = <C, Z>_F  (= the user's objective, `objective_r`)
//   dual  = b . y
//   gap   = |dual - prim| / (1 + |dual| + |prim|)
//   delta = max(pinf, dinf, gap)
//
// The loop records its trajectory (`trajectory_ok`): the iterates from Z = I,
// S = 0, each the update of the previous one for the penalty in force, which
// the infeasibilities of the new iterate then update (`penalty_follows`). The
// loop stops with the current iterate when delta < TOL and its PSD blocks pass
// the check (`psd_blocks_ok`), with the best PSD-verified iterate when the
// stagnation rule fires (`stagnation_ok`, `stagnation_stop`), or with the last
// iterate after TOTAL_STEPS iterations; the returned iterate is an element of
// the trajectory (`outcome_ok`). The user's solution is read from its diagonal
// blocks (`solution_ok`). With check, the residual bound of
// resid_spec.rs applies to the dense rows (`lemma_reduced_from_residual_bound`).

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
use crate::lin_alg::ldlt_spec::{blocks_fit, block_psd_ok, lemma_block_half_solve};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::reduce_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::norm_spec::frob_rc;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::resid_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_matrix::*;
#[cfg(verus_keep_ghost)]
use crate::sdp_spec::*;
#[cfg(verus_keep_ghost)]
use crate::sdp::*;
#[cfg(verus_keep_ghost)]
use crate::sdp_altdir::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::eigen_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::spectral_spec::*;

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

/// The reduced constraint system has at most as many rows as the stacked system, and at least
/// one with reduction. Without reduction it has exactly as many.
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
        check ==> rows >= 1,
        !check ==> rows == total,
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
        assert(aug.len() as int == cd * total);
        lemma_div_multiples_vanish(total, cd);
    } else {
        assert(rows == total) by (nonlinear_arith)
            requires cd >= 1, rows * cd == total * cd;
    }
}

/// The facts of the standard form that the solver setup relies on: the order d >= 1 of the
/// variable, the fitting of d and of the number of dense constraint rows, the lengths of the
/// returned data, the zero pattern, the zero right-hand sides of the zero constraints, the
/// block sizes (whose partition has order d, and whose LMI part is nonempty exactly with
/// LMIs), and dense rows whenever there are zero constraints (with LMIs the dense rows are
/// never all eliminated; without LMIs there are no zero constraints).
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
        (block_sum(dims_of(res.5@)) == 0) == (sdp.lmi@.len() == 0),
        res.2@.len() == res.1.1@.len() ==> res.1.1@.len() == 0,
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
        assert(block_sum(dims_of(res.5@)) == 0);
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
        // the LMI blocks are nonempty, so their total size and the row count are positive
        assert(sum_upto(bd, 1) == sum_upto(bd, 0) + bd[0]);
        lemma_sum_upto_monotone(bd, 1, bd.len() as int);
        assert(block_sum(bd) >= 1);
        assert(rows >= 1);
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
            assert(1.0real * (u * w) == u * w);
        } else {
            assert(2.0real * ((0.5real * u) * w) == u * w) by (nonlinear_arith);
        }
    }
    lemma_tri_sum_ext(f, h, dim as nat);
}

/* ---------- the PSD block partition ---------- */

/// The diagonal-block partition of the standard-form variable that the solver checks for
/// positive semidefiniteness: the LMI blocks and the two blocks of every user variable
/// (block_sizes, L16), or the whole variable when there are no LMI blocks (a single-variable
/// problem; user-variable blocks only exist together with LMI blocks).
pub open spec fn psd_partition(bd: Seq<int>, vd: Seq<int>, dim: int) -> Seq<int> {
    if block_sum(bd) == 0 { seq![dim] } else { block_sizes(bd, vd) }
}

/// list is the (start, size) list of the consecutive diagonal blocks of sizes bs.
pub open spec fn block_list_ok(list: Seq<(usize, usize)>, bs: Seq<int>) -> bool {
    &&& list.len() == bs.len()
    &&& forall|j: int| 0 <= j < bs.len()
            ==> (#[trigger] list[j]).0 as int == sum_upto(bs, j) && list[j].1 as int == bs[j]
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
/// `reg_shift_r`, y_bottom = 2 b_bottom (the multipliers of the zero constraints, whose Gram
/// block is I/2), and, when G was regularized, the residual of the unregularized system
/// passes the test.
pub open spec fn dual_solve_ok(g: Seq<real>, b: Seq<real>, c_len: int, check: bool, y: Seq<real>) -> bool {
    let n = b.len() as int;
    let yt = y.subrange(0, c_len);
    let bt = b.subrange(0, c_len);
    &&& y.len() == n
    &&& mat_vec(madd(sym_view(g), mscale(reg_shift_r(g, c_len, check), mident())), c_len, yt) == bt
    &&& forall|t: int| c_len <= t < n ==> #[trigger] y[t] == 2.0real * b[t]
    &&& !check && c_len > 0 ==> residual_ok(g, c_len, yt, bt)
}

/// A solution of the block system (A, I/2) x = b solves A x_top = b_top and has x_bottom = 2 b_bottom.
pub proof fn lemma_block_half_solve_inv(a: MatF, n: int, total: int, x: Seq<real>, b: Seq<real>)
    requires
        0 <= n <= total,
        x.len() == total,
        b.len() == total,
        mat_vec(block_half_view(a, n), total, x) == b,
    ensures
        mat_vec(a, n, x.subrange(0, n)) == b.subrange(0, n),
        forall|t: int| n <= t < total ==> #[trigger] x[t] == 2.0real * b[t],
{
    let m = block_half_view(a, n);
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
    assert forall|t: int| n <= t < total implies #[trigger] x[t] == 2.0real * b[t] by {
        let f = |k: int| m(t, k) * x[k];
        assert forall|k: int| 0 <= k < total && k != t implies #[trigger] f(k) == 0.0real by {
            assert(m(t, k) == 0.0real);
        }
        lemma_sum_fn_single(f, total as nat, t);
        assert(m(t, t) == 0.5real);
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

/* ---------- the Wen-Goldfarb-Yin update ---------- */

/// The right-hand side of the dual system, mu (b - A(Z)) + A(C - S), with one entry per
/// entry of A(Z).
pub open spec fn wgy_rhs_r(mu: real, b: Seq<real>, az: Seq<real>, acs: Seq<real>) -> Seq<real> {
    Seq::new(az.len(), |i: int| mu * (b[i] - az[i]) + acs[i])
}

/// diff is the packed dual slack C - A*(y) of the constraint operator with c_len dense rows
/// (the multipliers of the zero-pattern rows follow them in y).
pub open spec fn is_dual_slack(diff: Seq<real>, cf: Seq<real>, m: Seq<real>, c_len: int, list: Seq<(usize, usize)>, y: Seq<real>, dim: int) -> bool {
    &&& diff.len() == pk_len(dim)
    &&& forall|r: int, c: int| 0 <= r <= c < dim
            ==> #[trigger] diff[pk_idx(r, c)] == cf[pk_idx(r, c)] - adjoint_entry_r(m, c_len, list, y, r, c)
}

/// v = diff - mu z entrywise on the packed matrices of order dim.
pub open spec fn is_shifted_slack(v: Seq<real>, diff: Seq<real>, mu: real, z: Seq<real>, dim: int) -> bool {
    &&& v.len() == pk_len(dim)
    &&& forall|t: int| 0 <= t < pk_len(dim) ==> #[trigger] v[t] == diff[t] - mu * z[t]
}

/// s is the positive spectral part of v computed from the eigensolver output (q, b):
/// v = Q B Q^T with Q orthogonal and ||B - diag(B)||_F <= eps (eigensolver_ok), and
/// s = Q diag(lambda_+) Q^T with lambda = diag(B) and lambda_+ keeping the entries > 0
/// (the selection of poseigendecomp).
pub open spec fn pos_part_of(v: Seq<real>, b: Seq<real>, q: Seq<real>, s: Seq<real>, dim: int) -> bool {
    let vals = sym_diag_r(b, dim);
    &&& eigensolver_ok(v, b, q, vals, dim)
    &&& s.len() == pk_len(dim)
    &&& meq(sym_view(s), mmul(mmul(cm_view(q, dim), diag_view(masked_r(vals, pos_sel(vals))), dim), mtr(cm_view(q, dim)), dim), dim, dim)
}

/// s is the positive spectral part of v for some eigensolver output.
pub open spec fn is_pos_part(v: Seq<real>, s: Seq<real>, dim: int) -> bool {
    exists|q: Seq<real>, b: Seq<real>| #[trigger] pos_part_of(v, b, q, s, dim)
}

/// z1 = (1 - rho) z + (rho / mu) (s - v) entrywise on the packed matrices of order dim.
pub open spec fn is_primal_step(z1: Seq<real>, rho: real, mu: real, z: Seq<real>, s: Seq<real>, v: Seq<real>, dim: int) -> bool {
    &&& z1.len() == pk_len(dim)
    &&& forall|t: int| 0 <= t < pk_len(dim)
            ==> #[trigger] z1[t] == (1.0real - rho) * z[t] + (rho / mu) * (s[t] - v[t])
}

/// Without dense rows and zero-pattern rows the adjoint is zero.
pub proof fn lemma_adjoint_entry_empty(m: Seq<real>, list: Seq<(usize, usize)>, y: Seq<real>, r: int, c: int)
    requires
        list.len() == 0,
    ensures
        adjoint_entry_r(m, 0, list, y, r, c) == 0.0real,
{
    assert(col_comb_r(m, 0, y, pk_idx(r, c)) == 0.0real);
    assert(zero_comb_r(list, 0, y, r, c) == 0.0real);
}

/// Q diag(0) Q^T = 0: when no value is selected, the zero matrix is the positive part.
pub proof fn lemma_zero_pos_part(v: Seq<real>, b: Seq<real>, q: Seq<real>, s: Seq<real>, dim: int)
    requires
        dim >= 0,
        eigensolver_ok(v, b, q, sym_diag_r(b, dim), dim),
        sel_indices(pos_sel(sym_diag_r(b, dim)), dim as nat).len() == 0,
        s.len() == pk_len(dim),
        forall|t: int| 0 <= t < s.len() ==> #[trigger] s[t] == 0.0real,
    ensures
        pos_part_of(v, b, q, s, dim),
{
    let vals = sym_diag_r(b, dim);
    let sel = pos_sel(vals);
    let md = masked_r(vals, sel);
    let qv = cm_view(q, dim);
    lemma_sel_indices_empty(sel, dim as nat);
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim
        implies #[trigger] sym_view(s)(r, c) == mmul(mmul(qv, diag_view(md), dim), mtr(qv), dim)(r, c) by {
        lemma_mul_diag_transpose(qv, md, dim, r, c);
        let f = |k: int| md[k] * qv(r, k) * qv(c, k);
        assert forall|k: int| 0 <= k < dim implies #[trigger] f(k) == 0.0real by {
            assert(!sel(k));
            assert(md[k] == 0.0real);
            let (u, w) = (qv(r, k), qv(c, k));
            assert(0.0real * u * w == 0.0real);
        }
        lemma_sum_fn_zero(dim as nat);
        lemma_sum_fn_ext(f, |k: int| 0.0real, dim as nat);
        if r <= c {
            lemma_pk_idx_bound(r, c, dim);
        } else {
            lemma_pk_idx_bound(c, r, dim);
        }
    }
}

/// The reconstruction diag_scale_gram computes from the selected pairs of the positive
/// part is the positive part.
pub proof fn lemma_selected_pos_part(v: Seq<real>, b: Seq<real>, q: Seq<real>, vals: Seq<real>, vects: Seq<real>, s: Seq<real>, dim: int)
    requires
        dim >= 0,
        spectral_part_ok(v, b, q, pos_sel(sym_diag_r(b, dim)), vals, vects, dim),
        s.len() == pk_len(dim),
        meq(sym_view(s), mmul(mmul(cm_view(vects, dim), diag_view(vals), vals.len() as int), mtr(cm_view(vects, dim)), vals.len() as int), dim, dim),
    ensures
        pos_part_of(v, b, q, s, dim),
{
    let lam = sym_diag_r(b, dim);
    let sel = pos_sel(lam);
    lemma_selected_reconstruction(q, lam, sel, vals, vects, dim);
    assert(sel_count(sel, dim) == vals.len());
    lemma_meq_trans(sym_view(s),
        mmul(mmul(cm_view(vects, dim), diag_view(vals), vals.len() as int), mtr(cm_view(vects, dim)), vals.len() as int),
        mmul(mmul(cm_view(q, dim), diag_view(masked_r(lam, sel)), dim), mtr(cm_view(q, dim)), dim), dim, dim);
}

/// The positive spectral part is a packed matrix of order dim, symmetric and positive
/// semidefinite.
pub proof fn lemma_pos_part_psd_sym(v: Seq<real>, s: Seq<real>, dim: int)
    requires
        dim >= 0,
        is_pos_part(v, s, dim),
    ensures
        s.len() == pk_len(dim),
        is_symmetric(sym_view(s), dim),
        is_psd(sym_view(s), dim),
{
    let (q, b) = choose|q: Seq<real>, b: Seq<real>| #[trigger] pos_part_of(v, b, q, s, dim);
    let vals = sym_diag_r(b, dim);
    lemma_sym_view_symmetric(s, dim);
    lemma_pos_part_psd(q, vals, dim);
    let rec = mmul(mmul(cm_view(q, dim), diag_view(masked_r(vals, pos_sel(vals))), dim), mtr(cm_view(q, dim)), dim);
    lemma_meq_sym(sym_view(s), rec, dim, dim);
    lemma_psd_shifted_meq(rec, sym_view(s), dim, 0.0real);
}

/* ---------- residuals and KKT quantities ---------- */

/// The primal infeasibility ||A(Z) - b|| / (1 + ||b||).
pub open spec fn pinf_r(s: &SdpadSetup, z: Seq<real>) -> real {
    let b = rvs(s.point@);
    dist_r(setup_action_r(s, z), b) / (1.0real + norm_r(b))
}

/// The dual infeasibility ||diff - S||_F / (1 + ||C||_1) for the dual slack diff = C - A*(y).
pub open spec fn dinf_r(s: &SdpadSetup, diff: Seq<real>, sm: Seq<real>) -> real {
    let d = s.var_dim as int;
    frob_norm_r(vsub_r(diff, sm), d) / (1.0real + sym_one_norm_r(rvs(s.obj_frob@), d))
}

/// The primal objective value <C, Z>_F.
pub open spec fn prim_val_r(s: &SdpadSetup, z: Seq<real>) -> real {
    frob_r(rvs(s.obj_frob@), z, s.var_dim as int)
}

/// The dual objective value b . y.
pub open spec fn dual_val_r(s: &SdpadSetup, y: Seq<real>) -> real {
    dot_r(rvs(s.point@), y)
}

/// The relative duality gap |dual - prim| / (1 + |dual| + |prim|).
pub open spec fn gap_r(prim: real, dual: real) -> real {
    abs_r(dual - prim) / (1.0real + abs_r(dual) + abs_r(prim))
}

/// The accuracy measure max(pinf, dinf, gap).
pub open spec fn delta_r(pinf: real, dinf: real, gap: real) -> real {
    max_r(max_r(pinf, dinf), gap)
}

/// The residuals, primal objective value, gap (against the dual objective value) and delta of
/// the iterate st.
pub open spec fn kkt_ok(s: &SdpadSetup, st: &WgyState, pinf: real, dinf: real, prim: real, gap: real, delta: real) -> bool {
    &&& pinf == pinf_r(s, rvs(st.prim_z@))
    &&& dinf == dinf_r(s, rvs(st.diff@), rvs(st.dual_s@))
    &&& prim == prim_val_r(s, rvs(st.prim_z@))
    &&& gap == gap_r(prim, dual_val_r(s, rvs(st.dual_y@)))
    &&& delta == delta_r(pinf, dinf, gap)
}

/// The objective of the user's problem at the standard-form variable z: C(Z) for the single
/// variable without LMIs, C_1(Z_1) + .. + C_n(Z_n) with LMIs (obj_r).
pub open spec fn objective_r(sdp: &SDP, z: Seq<real>) -> real {
    if sdp.lmi@.len() == 0 {
        dot_r(part_r(sdp.objective@[0]@, sdp.objective@[0]@.len() as int), z)
    } else {
        obj_r(sdp.objective@, block_sum(bdims_r(sdp.lmi@)), vdims_r(sdp.lmi@, n_vars(sdp)), z)
    }
}

/// The standard-form objective before Frobenius reweighting: the witness of the setup
/// contract (sdpad_setup_ok), whose halved off-diagonal entries give obj_frob.
pub open spec fn setup_obj0(sdp: &SDP, check: bool, s: &SdpadSetup) -> SymMatrix {
    choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), s.var_dim as int)
}

/// Under the setup contract, setup_obj0 is such a witness.
pub proof fn lemma_setup_obj0(sdp: &SDP, check: bool, s: &SdpadSetup)
    requires
        sdpad_setup_ok(sdp, check, *s),
    ensures
        std_form_ok(sdp, (setup_obj0(sdp, check, s), (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check),
        half_off_diag(rvs(s.obj_frob@), rvs(setup_obj0(sdp, check, s)@), s.var_dim as int),
{
}

/// The primal objective value is the user's objective (lemma_frob_half_dot and the
/// standard-form contract).
pub proof fn lemma_objective_value(sdp: &SDP, check: bool, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_setup_ok(sdp, check, *s),
        z.len() == pk_len(s.var_dim as int),
    ensures
        prim_val_r(s, z) == objective_r(sdp, z),
{
    let d = s.var_dim as int;
    let obj0 = setup_obj0(sdp, check, s);
    lemma_setup_obj0(sdp, check, s);
    lemma_frob_half_dot(rvs(s.obj_frob@), rvs(obj0@), z, d);
    if sdp.lmi@.len() == 0 {
        assert(rvs(obj0@) == part_r(sdp.objective@[0]@, sdp.objective@[0]@.len() as int));
    } else {
        assert(dot_r(rvs(obj0@), z) == obj_r(sdp.objective@, block_sum(bdims_r(sdp.lmi@)), vdims_r(sdp.lmi@, n_vars(sdp)), z));
    }
}

/* ---------- the solution ---------- */

/// result is the user's solution read from the standard-form variable z: z itself without
/// LMIs, otherwise Z_n = B_{s+2n} - B_{s+2n+1}, the difference of the two diagonal blocks of
/// variable n after the LMI blocks (zvar_r).
pub open spec fn solution_ok(s: &SdpadSetup, z: Seq<real>, result: Seq<SymMatrix>) -> bool {
    let vd = dims_of(s.symm_dims@);
    let bsum = s.block_size_sum as int;
    if s.block_size_sum == 0 {
        result.len() == 1 && rvs(result[0]@) == z
    } else {
        &&& result.len() == vd.len()
        &&& forall|n: int| 0 <= n < vd.len() ==> rvs(#[trigger] result[n]@) == zvar_r(z, var_off(bsum, vd, n), vd[n])
    }
}

/// The packed diagonal block that sym_matrix_diag_block reads from zf is blk_r of its view.
pub proof fn lemma_diag_block_view(b: Seq<f64>, zf: Seq<f64>, st: int, dn: int)
    requires
        st >= 0,
        dn >= 0,
        b.len() == pk_len(dn),
        zf.len() >= pk_len(st + dn),
        forall|r: int, c: int| 0 <= r <= c < dn ==> rv(#[trigger] b[pk_idx(r, c)]) == rv(zf[pk_idx(st + r, st + c)]),
    ensures
        rvs(b) == blk_r(rvs(zf), st, dn),
{
    assert forall|t: int| 0 <= t < pk_len(dn) implies rvs(b)[t] == #[trigger] blk_r(rvs(zf), st, dn)[t] by {
        lemma_pk_idx_surjective(t, dn);
        let (r, c) = choose|r: int, c: int| 0 <= r <= c < dn && #[trigger] pk_idx(r, c) == t;
        lemma_pk_unpack_idx(r, c);
        lemma_pk_idx_bound(st + r, st + c, st + dn);
        assert(rvs(zf)[pk_idx(st + r, st + c)] == rv(zf[pk_idx(st + r, st + c)]));
    }
    assert(rvs(b) =~= blk_r(rvs(zf), st, dn));
}

/* ---------- the termination of the loop ---------- */

/// Every block of the PSD partition of z passed the check: B >= -eps I (block_psd_ok, L07).
pub open spec fn psd_blocks_ok(s: &SdpadSetup, z: Seq<real>) -> bool {
    forall|b: int| 0 <= b < s.psd_blocks@.len()
        ==> #[trigger] block_psd_ok(z, s.psd_blocks@[b].0 as int, s.psd_blocks@[b].1 as int)
}

/// The penalty bookkeeping of an iteration: the penalty mu and the counters it_pinf, it_dinf
/// of penalty_update.
pub ghost struct Penalty {
    pub mu: real,
    pub ip: int,
    pub id: int,
}

/// The initial penalty PEN_INIT with zero counters.
pub open spec fn initial_penalty() -> Penalty {
    Penalty { mu: rv(PEN_INIT), ip: 0, id: 0 }
}

/// pen1 is the penalty update of pen0 from the infeasibilities of the iterate st.
pub open spec fn penalty_follows(s: &SdpadSetup, pen0: Penalty, st: &WgyState, pen1: Penalty) -> bool {
    penalty_update_ok(pen0.mu, pinf_r(s, rvs(st.prim_z@)), dinf_r(s, rvs(st.diff@), rvs(st.dual_s@)),
        pen0.ip, pen0.id, pen1.mu, pen1.ip, pen1.id)
}

/// The trajectory of the loop after pens.len() iterations: states[0] holds the initial
/// iterates Z = I, S = 0; states[k + 1] is the Wen-Goldfarb-Yin update of states[k] for the
/// penalty pens[k].mu and the step STEP; pens[0] is the initial penalty, and pens[k + 1] is
/// the penalty update of pens[k] from the infeasibilities of states[k + 1].
pub open spec fn trajectory_ok(s: &SdpadSetup, check: bool, states: Seq<WgyState>, pens: Seq<Penalty>) -> bool {
    let n = pens.len() as int;
    &&& states.len() == n + 1
    &&& initial_iterates_ok(rvs(states[0].prim_z@), rvs(states[0].dual_s@), s.var_dim as int)
    &&& n >= 1 ==> pens[0] == initial_penalty()
    &&& forall|k: int| 0 <= k < n ==> #[trigger] wgy_update_ok(s, check, pens[k].mu, rv(STEP), &states[k], &states[k + 1])
    &&& forall|k: int| 0 <= k < n - 1 ==> #[trigger] penalty_follows(s, pens[k], &states[k + 1], pens[k + 1])
}

/// The trajectory is extended by an iteration: the update st of the last state for the
/// penalty pen in force (the initial penalty, or the update of the last penalty from the
/// last state).
pub proof fn lemma_trajectory_step(s: &SdpadSetup, check: bool, states: Seq<WgyState>, pens: Seq<Penalty>, pen: Penalty, st: WgyState)
    requires
        trajectory_ok(s, check, states, pens),
        pens.len() == 0 ==> pen == initial_penalty(),
        pens.len() >= 1 ==> penalty_follows(s, pens.last(), &states.last(), pen),
        wgy_update_ok(s, check, pen.mu, rv(STEP), &states.last(), &st),
    ensures
        trajectory_ok(s, check, states.push(st), pens.push(pen)),
{
    let n = pens.len() as int;
    let (states2, pens2) = (states.push(st), pens.push(pen));
    assert forall|k: int| 0 <= k < n + 1
        implies #[trigger] wgy_update_ok(s, check, pens2[k].mu, rv(STEP), &states2[k], &states2[k + 1]) by {
        if k < n {
            assert(wgy_update_ok(s, check, pens[k].mu, rv(STEP), &states[k], &states[k + 1]));
        }
    }
    assert forall|k: int| 0 <= k < n
        implies #[trigger] penalty_follows(s, pens2[k], &states2[k + 1], pens2[k + 1]) by {
        if k < n - 1 {
            assert(penalty_follows(s, pens[k], &states[k + 1], pens[k + 1]));
        }
    }
}

/// delta of the iterate st: max(pinf, dinf, gap) of its KKT quantities (kkt_ok).
pub open spec fn iterate_delta_r(s: &SdpadSetup, st: &WgyState) -> real {
    let prim = prim_val_r(s, rvs(st.prim_z@));
    let dual = dual_val_r(s, rvs(st.dual_y@));
    delta_r(pinf_r(s, rvs(st.prim_z@)), dinf_r(s, rvs(st.diff@), rvs(st.dual_s@)), gap_r(prim, dual))
}

/// The stagnation bookkeeping over the deltas of the iterations so far: best bounds every
/// delta from below, the last it_stag deltas lie strictly above best, and the one before them
/// equals best (when there is one; initially best is the sentinel +infinity, which the model
/// leaves unconstrained).
pub open spec fn stagnation_ok(deltas: Seq<real>, best: real, it_stag: int) -> bool {
    let n = deltas.len() as int;
    &&& 0 <= it_stag <= n
    &&& forall|k: int| 0 <= k < n ==> #[trigger] deltas[k] >= best
    &&& forall|k: int| n - it_stag <= k < n ==> #[trigger] deltas[k] > best
    &&& it_stag < n ==> deltas[n - 1 - it_stag] == best
}

/// One step of the bookkeeping: a delta above best is a stagnating iteration, any other delta
/// becomes the best.
pub open spec fn stag_step(delta: real, best: real, it_stag: int) -> (real, int) {
    if delta > best { (best, it_stag + 1) } else { (delta, 0) }
}

/// The bookkeeping is preserved by a step.
pub proof fn lemma_stag_step(deltas: Seq<real>, best: real, it_stag: int, delta: real)
    requires
        stagnation_ok(deltas, best, it_stag),
    ensures
        stagnation_ok(deltas.push(delta), stag_step(delta, best, it_stag).0, stag_step(delta, best, it_stag).1),
{
}

/// best_feas bounds the deltas of the candidate iterations (feasible) from below and, once a
/// solution has been recorded, is attained by one of them.
pub open spec fn best_feasible_ok(deltas: Seq<real>, feasible: Seq<bool>, best_feas: real, recorded: bool) -> bool {
    let n = deltas.len() as int;
    &&& feasible.len() == n
    &&& forall|k: int| 0 <= k < n && feasible[k] ==> #[trigger] deltas[k] >= best_feas
    &&& recorded ==> exists|k: int| 0 <= k < n && feasible[k] && #[trigger] deltas[k] == best_feas
}

/// The candidate bookkeeping is preserved by an iteration: a recorded candidate (delta not
/// above the best) becomes the best.
pub proof fn lemma_best_feasible_step(deltas: Seq<real>, feasible: Seq<bool>, best_feas: real, recorded: bool, delta: real, record: bool)
    requires
        best_feasible_ok(deltas, feasible, best_feas, recorded),
        record ==> delta <= best_feas,
    ensures
        best_feasible_ok(deltas.push(delta), feasible.push(record), if record { delta } else { best_feas }, recorded || record),
{
    let n = deltas.len() as int;
    let (d2, f2) = (deltas.push(delta), feasible.push(record));
    let b2 = if record { delta } else { best_feas };
    assert forall|k: int| 0 <= k < n + 1 && f2[k] implies #[trigger] d2[k] >= b2 by {
        if k < n {
            assert(deltas[k] >= best_feas);
        }
    }
    if record {
        assert(f2[n] && d2[n] == b2);
    } else if recorded {
        let k = choose|k: int| 0 <= k < n && feasible[k] && #[trigger] deltas[k] == best_feas;
        assert(f2[k] && d2[k] == b2);
    }
}

/// The recorded best solution: the iterate states[k], recorded as a candidate (feasible[k - 1])
/// with its PSD block check succeeded, whose Z is stored in zb, with its KKT quantities.
pub open spec fn best_record_ok(s: &SdpadSetup, states: Seq<WgyState>, feasible: Seq<bool>, k: int, zb: Seq<f64>, pinf: real, dinf: real, gap: real, val: real, delta: real) -> bool {
    &&& 1 <= k < states.len()
    &&& k <= feasible.len()
    &&& feasible[k - 1]
    &&& states[k].prim_z@ == zb
    &&& psd_blocks_ok(s, rvs(zb))
    &&& kkt_ok(s, &states[k], pinf, dinf, val, gap, delta)
}

/// The stagnation stopping rule: more than STAG1 (STAG2, STAG3) iterations without improving
/// the best delta, and the current delta below 1e-5 (1e-4, 1e-3).
pub open spec fn stagnation_stop(delta: real, it_stag: int) -> bool {
    ||| it_stag > STAG1 as int && delta < 0.00001real
    ||| it_stag > STAG2 as int && delta < 0.0001real
    ||| it_stag > STAG3 as int && delta < 0.001real
}

/// The outcome of the loop: the trajectory of the n iterations run (states, pens), whose
/// deltas are recorded in order; which iterations were recorded as candidates; the stagnation
/// bookkeeping after the last iteration; and the solution read from the iterate
/// states[iterate], with that iterate's KKT quantities and objective value. The termination
/// reason fixes the iterate: Accuracy (the last iterate, with delta < TOL and PSD-verified
/// blocks), Stagnation (a recorded PSD-verified candidate attaining the least recorded delta,
/// after the stopping rule fired on the last iterate), or MaxIter (the last of TOTAL_STEPS
/// iterates).
pub open spec fn outcome_ok(sdp: &SDP, s: &SdpadSetup, check: bool, o: SdpadOutcome) -> bool {
    let states = o.states@;
    let deltas = rvs(o.deltas@);
    let n = deltas.len() as int;
    let k = o.iterate@;
    let st = states[k];
    let delta = delta_r(rv(o.pinf), rv(o.dinf), rv(o.gap));
    &&& 1 <= n <= TOTAL_STEPS
    &&& trajectory_ok(s, check, states, o.pens@)
    &&& o.pens@.len() == n
    &&& forall|j: int| 0 <= j < n ==> #[trigger] deltas[j] == iterate_delta_r(s, &states[j + 1])
    &&& o.feasible@.len() == n
    &&& stagnation_ok(deltas, rv(o.best_delta@), o.it_stag@ as int)
    &&& 1 <= k <= n
    &&& solution_ok(s, rvs(st.prim_z@), o.result@)
    &&& kkt_ok(s, &st, rv(o.pinf), rv(o.dinf), rv(o.value), rv(o.gap), delta)
    &&& rv(o.value) == objective_r(sdp, rvs(st.prim_z@))
    &&& match o.termination {
            Termination::Accuracy => k == n && psd_blocks_ok(s, rvs(st.prim_z@)) && delta < rv(TOL),
            Termination::Stagnation => psd_blocks_ok(s, rvs(st.prim_z@))
                && o.feasible@[k - 1]
                && best_feasible_ok(deltas, o.feasible@, delta, true)
                && stagnation_stop(deltas[n - 1], o.it_stag@ as int),
            Termination::MaxIter => k == n && n == TOTAL_STEPS,
        }
}

/* ---------- the penalty update ---------- */

/// The penalty step for the infeasibility ratio pinf / dinf: a ratio <= PEN_RATIO1 counts a
/// primal-dominated iteration (and resets the dual count), a ratio > PEN_RATIO2 a
/// dual-dominated one; after PEN_COUNT consecutive iterations of one kind the penalty is
/// scaled by PEN_FACT (resp. 1 / PEN_FACT), clamped to [PEN_MIN, PEN_MAX], and the count
/// restarts.
pub open spec fn penalty_step_ok(mu0: real, ratio: real, ip0: int, id0: int, mu1: real, ip1: int, id1: int) -> bool {
    if !(ratio > rv(PEN_RATIO1)) {
        &&& id1 == 0
        &&& if ip0 + 1 >= PEN_COUNT as int {
                mu1 == max_r(rv(PEN_FACT) * mu0, rv(PEN_MIN)) && ip1 == 0
            } else {
                mu1 == mu0 && ip1 == ip0 + 1
            }
    } else if ratio > rv(PEN_RATIO2) {
        &&& ip1 == 0
        &&& if id0 + 1 >= PEN_COUNT as int {
                mu1 == min_r(mu0 / rv(PEN_FACT), rv(PEN_MAX)) && id1 == 0
            } else {
                mu1 == mu0 && id1 == id0 + 1
            }
    } else {
        mu1 == mu0 && ip1 == ip0 && id1 == id0
    }
}

/// The penalty update from the infeasibilities: the step for the ratio pinf / dinf (an
/// unspecified value when dinf = 0, where the executable quotient is not modeled).
pub open spec fn penalty_update_ok(mu0: real, pinf: real, dinf: real, ip0: int, id0: int, mu1: real, ip1: int, id1: int) -> bool {
    exists|ratio: real| (dinf != 0.0real ==> ratio == pinf / dinf)
        && #[trigger] penalty_step_ok(mu0, ratio, ip0, id0, mu1, ip1, id1)
}


/* ---------- the Gram system of the constraint operator ---------- */

/// The packed part of the returned augmented system is the constraint matrix.
pub proof fn lemma_result_aug_entry(mat: Seq<f64>, point: Seq<f64>, rows: int, cols: int, a: int, t: int)
    requires
        mat.len() == rows * cols,
        0 <= a < rows,
        0 <= t < cols,
    ensures
        cm_entry(result_aug(mat, point, rows), rows, a, t) == constr_entry(rvs(mat), rows, a, t),
{
    lemma_cm_idx_bound(rows, a, t, cols);
    assert(cols * rows == rows * cols);
    assert((rvs(mat) + rvs(point.subrange(0, rows)))[cm_idx(rows, a, t)] == rvs(mat)[cm_idx(rows, a, t)]);
}

/// A zero column of the stacked system is a zero column of its reduction: the elimination
/// keeps zero columns and the kept rows are only rescaled.
pub proof fn lemma_reduced_zero_col(a0: Seq<real>, red: Seq<real>, dim: int, total: int, rows: int, t: int)
    requires
        dim >= 0,
        total >= 0,
        rows >= 0,
        a0.len() == total * red_col_dim(dim),
        red.len() == rows * red_col_dim(dim),
        reduced_from(a0, red, dim),
        0 <= t < red_col_dim(dim),
        col_zero(a0, total, t),
    ensures
        col_zero(red, rows, t),
{
    let cd = red_col_dim(dim);
    lemma_pk_col_start_monotone(0, dim);
    let e = choose|e: Seq<real>| reduced_system(a0, e, red, dim);
    let pivots = choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(a0, e, red, dim, pivots);
    let rd = a0.len() as int / cd;
    assert(a0.len() as int == cd * total) by (nonlinear_arith)
        requires a0.len() as int == total * cd;
    lemma_div_multiples_vanish(total, cd);
    assert(rd == total);
    let rank = pivots.len() as int;
    assert(rank == rows) by (nonlinear_arith)
        requires cd >= 1, cd * rank == rows * cd;
    assert(col_zero(e, rd, t));
    assert forall|i: int| 0 <= i < rows implies #[trigger] cm_entry(red, rows, i, t) == 0.0real by {
        let sc = row_scale_r(e, rd, dim, i);
        assert(cm_entry(red, rank, i, t) == cm_entry(e, rd, i, t) / sc);
        assert(cm_entry(e, rd, i, t) == 0.0real);
        assert(0.0real / sc == 0.0real);
    }
}

/// The echelon form behind a reduction (a witness of `reduced_from`).
pub open spec fn red_echelon(a0: Seq<real>, red: Seq<real>, dim: int) -> Seq<real> {
    choose|e: Seq<real>| reduced_system(a0, e, red, dim)
}

/// The residual bound for the dense rows of a setup with `check`: U and C are built from the
/// witnesses `red_echelon`, `red_pivots` and `red_transform`, all independent of z. With
/// d(z) = ||A||_F ||z|| + ||b||, ||A z - b|| <= C TOL d(z) + ||U||_F ||A_K z - b_K||, in relative
/// form when d(z) > 0; d(z) = 0 forces A z - b = 0.
pub proof fn lemma_reduced_from_residual_bound(a0: Seq<real>, red: Seq<real>, dim: int, z: Seq<real>)
    requires
        dim >= 1,
        reduced_from(a0, red, dim),
        z.len() == red_col_dim(dim) - 1,
    ensures
        ({
            let e = red_echelon(a0, red, dim);
            let t = red_transform(a0, e, dim);
            let rank = red_pivots(a0, e, red, dim).len() as int;
            let nr = norm_r(resid_vec_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z));
            let d = resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z);
            let ct = red_const_r(a0, dim, t, rank) * rv(TOL);
            let w = frob_rc(recovery_r(a0, e, dim, t), red_row_dim(a0, dim), rank)
                * norm_r(resid_vec_r(red, rank, red_col_dim(dim), z));
            &&& nr <= ct * d + w
            &&& d > 0.0real ==> nr / d <= ct + w / d
            &&& d == 0.0real ==> nr == 0.0real
        }),
{
    let e = red_echelon(a0, red, dim);
    assert(reduced_system(a0, e, red, dim));
    lemma_reduction_residual_bound(a0, e, red, dim, z);
}

/// The solver's constraint operator satisfies the hypotheses of the block Gram form: the
/// zero-pattern positions are distinct (zeros_ok) and the dense rows vanish on them (the
/// stacked rows read the variable through its diagonal blocks, and the reduction keeps zero
/// columns); without LMIs there are no zero constraints.
pub proof fn lemma_setup_gram_hypotheses(sdp: &SDP, check: bool, s: &SdpadSetup)
    requires
        sdpad_setup_ok(sdp, check, *s),
    ensures
        distinct_zero_list(s.zeros@),
        rows_vanish_on_zeros(rvs(s.mat_dense@), s.c_len as int, s.zeros@),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let nz = s.zeros@.len() as int;
    let rows = s.c_len as int;
    let list = s.zeros@;
    let obj0 = setup_obj0(sdp, check, s);
    lemma_setup_obj0(sdp, check, s);
    if sdp.lmi@.len() == 0 {
        assert(nz == 0);
    } else {
        let nv = n_vars(sdp);
        let bd = bdims_r(sdp.lmi@);
        let vd = vdims_r(sdp.lmi@, nv);
        let sd = std_dim(bd, vd);
        let bs = block_sizes(bd, vd);
        let m = sdp.constraint.1@.len() as int;
        let total = lmi_rows(bd) + m;
        assert(d == sd);
        assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 0 by {
            assert(bd[i] == dims_of(s.block_dims@)[i]);
        }
        assert forall|n: int| 0 <= n < vd.len() implies #[trigger] vd[n] >= 0 by {
            assert(vd[n] == dims_of(s.symm_dims@)[n]);
        }
        assert(zeros_ok(list, bs, sd));
        let aug = choose|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
            && (if check { reduced_from(aug, result_aug(s.mat_dense@, s.point@, rows), sd) } else { result_aug(s.mat_dense@, s.point@, rows) == aug });
        lemma_pk_col_start_monotone(0, d);
        lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
        lemma_reduced_rows(aug, s.mat_dense@, s.point@, rows, total, d, check);
        assert forall|a: int, q: int| 0 <= a < rows && 0 <= q < nz
            implies #[trigger] constr_entry(rvs(s.mat_dense@), rows, a, zero_idx(list, q)) == 0.0real by {
            let (r0, c0) = (list[q].0 as int, list[q].1 as int);
            let t0 = pk_idx(r0, c0);
            assert(in_zero_region(bs, r0, c0));
            lemma_pk_idx_bound(r0, c0, d);
            lemma_std_rows_vanish(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug, r0, c0);
            assert(col_zero(aug, total, t0));
            let red = result_aug(s.mat_dense@, s.point@, rows);
            assert(red.len() == rows * cols + rows);
            assert(rows * cols + rows == rows * (cols + 1)) by (nonlinear_arith);
            if check {
                lemma_reduced_zero_col(aug, red, d, total, rows, t0);
            } else {
                assert(rows == total);
            }
            assert(cm_entry(red, rows, a, t0) == 0.0real);
            lemma_result_aug_entry(s.mat_dense@, s.point@, rows, cols, a, t0);
        }
    }
}

/// The dense Gram block of the setup satisfies is_constraint_gram also without dense rows.
pub proof fn lemma_setup_gram(sdp: &SDP, check: bool, s: &SdpadSetup)
    requires
        sdpad_setup_ok(sdp, check, *s),
    ensures
        is_constraint_gram(rvs(s.gram_small@), rvs(s.mat_dense@), s.c_len as int, s.var_dim as int),
{
    if s.c_len == 0 {
        lemma_pk_col_start_zero();
    }
}

/// The Gram matrix of the setup's constraint operator (dense rows, then zero rows) is the
/// block matrix (G, I/2) of the dense Gram matrix and half the identity.
pub proof fn lemma_setup_full_gram(sdp: &SDP, check: bool, s: &SdpadSetup)
    requires
        sdpad_setup_ok(sdp, check, *s),
    ensures
        meq(full_gram_view(rvs(s.mat_dense@), s.c_len as int, s.zeros@, s.var_dim as int),
            block_half_view(sym_view(rvs(s.gram_small@)), s.c_len as int), s.point@.len() as int, s.point@.len() as int),
{
    lemma_setup_gram_hypotheses(sdp, check, s);
    lemma_setup_gram(sdp, check, s);
    lemma_full_gram_block(rvs(s.gram_small@), rvs(s.mat_dense@), s.c_len as int, s.zeros@, s.var_dim as int);
}

/// (A + s I) x = A x + s x on the leading dim rows.
pub proof fn lemma_mat_vec_shift(a: MatF, s: real, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        forall|i: int| 0 <= i < dim
            ==> #[trigger] mat_vec(madd(a, mscale(s, mident())), dim, x)[i] == mat_vec(a, dim, x)[i] + s * x[i],
{
    let ms = madd(a, mscale(s, mident()));
    assert forall|i: int| 0 <= i < dim
        implies #[trigger] mat_vec(ms, dim, x)[i] == mat_vec(a, dim, x)[i] + s * x[i] by {
        let f = |k: int| a(i, k) * x[k];
        let g = |k: int| (s * mident()(i, k)) * x[k];
        let h = |k: int| ms(i, k) * x[k];
        assert forall|k: int| 0 <= k < dim implies #[trigger] h(k) == f(k) + g(k) by {
            let (u, v, w) = (a(i, k), s * mident()(i, k), x[k]);
            assert(h(k) == (u + v) * w);
            assert((u + v) * w == u * w + v * w) by (nonlinear_arith);
        }
        lemma_sum_fn_add(f, g, h, dim as nat);
        assert forall|k: int| 0 <= k < dim && k != i implies #[trigger] g(k) == 0.0real by {
            assert(mident()(i, k) == 0.0real);
        }
        lemma_sum_fn_single(g, dim as nat, i);
        assert(g(i) == (s * 1.0real) * x[i]);
    }
}

/// The right-hand side of the Gram system actually satisfied by the regularized solve: b less
/// the shift s y on the dense rows (b itself when there is no shift).
pub open spec fn reg_rhs_r(g: Seq<real>, b: Seq<real>, c_len: int, check: bool, y: Seq<real>) -> Seq<real> {
    Seq::new(b.len(), |t: int| b[t] - (if t < c_len { reg_shift_r(g, c_len, check) } else { 0.0real }) * y[t])
}

/// A solution of the block system of dual_solve solves the Gram system of the full constraint
/// operator, up to the regularization shift on the dense rows: A A* y = b - s (y_top, 0).
pub proof fn lemma_dual_solve_gram(g: Seq<real>, m: Seq<real>, list: Seq<(usize, usize)>, dim: int, c_len: int, check: bool, b: Seq<real>, y: Seq<real>)
    requires
        c_len >= 0,
        dim >= 0,
        m.len() == c_len * pk_len(dim),
        is_constraint_gram(g, m, c_len, dim),
        valid_zero_list(list, dim),
        distinct_zero_list(list),
        rows_vanish_on_zeros(m, c_len, list),
        b.len() == c_len + list.len(),
        dual_solve_ok(g, b, c_len, check, y),
    ensures
        mat_vec(full_gram_view(m, c_len, list, dim), b.len() as int, y) == reg_rhs_r(g, b, c_len, check, y),
{
    let n = b.len() as int;
    let sh = reg_shift_r(g, c_len, check);
    let bs = reg_rhs_r(g, b, c_len, check, y);
    let yt = y.subrange(0, c_len);
    let bt = b.subrange(0, c_len);
    let gv = sym_view(g);
    lemma_mat_vec_shift(gv, sh, c_len, yt);
    assert forall|i: int| 0 <= i < c_len implies #[trigger] mat_vec(gv, c_len, yt)[i] == bs.subrange(0, c_len)[i] by {
        assert(mat_vec(madd(gv, mscale(sh, mident())), c_len, yt)[i] == bt[i]);
    }
    assert(mat_vec(gv, c_len, yt) =~= bs.subrange(0, c_len));
    assert forall|t: int| c_len <= t < n implies #[trigger] y[t] == 2.0real * bs[t] by {
        assert(bs[t] == b[t] - 0.0real * y[t]);
    }
    lemma_block_half_solve(gv, c_len, n, y, bs);
    lemma_full_gram_block(g, m, c_len, list, dim);
    lemma_mat_vec_meq(full_gram_view(m, c_len, list, dim), block_half_view(gv, c_len), n, y);
}

/// The multipliers of every iteration solve the Gram system of the constraint operator:
/// (A A*) y = mu (b - A(Z)) + A(C - S), less the regularization shift on the dense rows.
pub proof fn lemma_wgy_dual_gram(sdp: &SDP, check: bool, s: &SdpadSetup, mu: real, rho: real, st0: &WgyState, st1: &WgyState)
    requires
        sdpad_setup_ok(sdp, check, *s),
        wgy_update_ok(s, check, mu, rho, st0, st1),
    ensures
        ({
            let rhs = wgy_rhs_r(mu, rvs(s.point@), setup_action_r(s, rvs(st0.prim_z@)),
                setup_action_r(s, vsub_r(rvs(s.obj_frob@), rvs(st0.dual_s@))));
            mat_vec(full_gram_view(rvs(s.mat_dense@), s.c_len as int, s.zeros@, s.var_dim as int), s.point@.len() as int, rvs(st1.dual_y@))
                == reg_rhs_r(rvs(s.gram_small@), rhs, s.c_len as int, check, rvs(st1.dual_y@))
        }),
{
    let rhs = wgy_rhs_r(mu, rvs(s.point@), setup_action_r(s, rvs(st0.prim_z@)),
        setup_action_r(s, vsub_r(rvs(s.obj_frob@), rvs(st0.dual_s@))));
    lemma_setup_gram_hypotheses(sdp, check, s);
    lemma_setup_gram(sdp, check, s);
    lemma_dual_solve_gram(rvs(s.gram_small@), rvs(s.mat_dense@), s.zeros@, s.var_dim as int, s.c_len as int, check, rhs, rvs(st1.dual_y@));
}

/* ---------- the duality gap ---------- */

/// The dual slack C_F - A*(y) of the setup's constraint operator, as a packed matrix of order d.
pub open spec fn setup_slack_r(s: &SdpadSetup, y: Seq<real>) -> Seq<real> {
    vsub_r(rvs(s.obj_frob@), adjoint_r(rvs(s.mat_dense@), s.c_len as int, s.zeros@, y, s.var_dim as int))
}

/// The gap identity: for packed Z, S of order d and multipliers y,
///
///     <C_F, Z>_F - b . y = <Z, S>_F + <Z, C_F - A*(y) - S>_F + y . (A(Z) - b),
///
/// by the adjoint identity <A(Z), y> = <Z, A*(y)>_F and linearity.
pub proof fn lemma_gap_identity(sdp: &SDP, check: bool, s: &SdpadSetup, z: Seq<real>, sm: Seq<real>, y: Seq<real>)
    requires
        sdpad_setup_ok(sdp, check, *s),
        z.len() == pk_len(s.var_dim as int),
        sm.len() == pk_len(s.var_dim as int),
        y.len() == s.point@.len(),
    ensures
        prim_val_r(s, z) - dual_val_r(s, y)
            == frob_r(z, sm, s.var_dim as int)
                + frob_r(z, vsub_r(setup_slack_r(s, y), sm), s.var_dim as int)
                + dot_r(y, vsub_r(setup_action_r(s, z), rvs(s.point@))),
{
    let d = s.var_dim as int;
    let cf = rvs(s.obj_frob@);
    let m = rvs(s.mat_dense@);
    let (rows, list) = (s.c_len as int, s.zeros@);
    let b = rvs(s.point@);
    let a = adjoint_r(m, rows, list, y, d);
    let az = setup_action_r(s, z);
    lemma_pk_col_start_monotone(0, d);
    // <Z, S> + <Z, (C_F - A*(y)) - S> = <Z, C_F> - <Z, A*(y)>
    lemma_frob_sub_right(z, setup_slack_r(s, y), sm, d);
    lemma_frob_sub_right(z, cf, a, d);
    lemma_frob_comm(z, cf, d);
    // <Z, A*(y)> = <A(Z), y>
    lemma_adjoint_r_ok(m, rows, list, y, d);
    lemma_constraint_adjoint(m, rows, d, list, z, y, a);
    lemma_dot_comm(az, y);
    // y . (A(Z) - b) = y . A(Z) - b . y
    lemma_dot_sub_right(y, az, b);
    lemma_dot_comm(y, b);
}

/// Complementarity: with A(Z) = b and C_F - A*(y) = S the objective values differ by <Z, S>_F.
pub proof fn lemma_gap_complementarity(sdp: &SDP, check: bool, s: &SdpadSetup, z: Seq<real>, sm: Seq<real>, y: Seq<real>)
    requires
        sdpad_setup_ok(sdp, check, *s),
        z.len() == pk_len(s.var_dim as int),
        y.len() == s.point@.len(),
        setup_action_r(s, z) == rvs(s.point@),
        setup_slack_r(s, y) == sm,
    ensures
        prim_val_r(s, z) - dual_val_r(s, y) == frob_r(z, sm, s.var_dim as int),
{
    let d = s.var_dim as int;
    let b = rvs(s.point@);
    lemma_pk_col_start_monotone(0, d);
    lemma_gap_identity(sdp, check, s, z, sm, y);
    // both residual terms vanish
    lemma_frob_sub_right(z, sm, sm, d);
    lemma_dot_sub_right(y, b, b);
}

} // verus!
