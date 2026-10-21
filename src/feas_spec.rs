// Feasibility correspondence between the user's problem and its standard form (not trusted).
//
// The user's problem with LMIs (sdp.rs) asks for packed symmetric X_1, .., X_n with
//
//     sum_n A_n(X_n) = b,      H_i(X) - M_i  symmetric positive semidefinite  (i = 1, .., s),
//
// where A_n, H_i are the maps stored as MatrixSym rows (`user_row_r`, `user_lmi_view`). The
// standard form (sdp_spec.rs) carries one packed variable Z of order d whose diagonal blocks are
// the LMI slacks B_i and the pairs P_n, N_n with X_n = P_n - N_n (`vars_of`).
//
// Soundness (`lemma_standard_feasible_sound`): a Z that solves every row of the unreduced
// system, vanishes on the zero pattern and is positive semidefinite is block diagonal, its
// LMI blocks are the PSD slacks H_i(X) - M_i, the variables X_n read from it satisfy the
// equality constraints, and <C_F, Z>_F is their objective value. Without reduction this is the
// case A(Z) = b for the setup's operator (`lemma_action_feasible_sound`); with reduction a
// solution of the reduced system satisfies the dropped rows only approximately
// (lemma_reduced_solution_echelon), so the theorem is stated for the unreduced system.
//
// Converse (`lemma_user_feasible_lifts`): feasible X_1, .., X_n lift to the block-diagonal
// Z = lift_z with blocks B_i = H_i(X) - M_i, P_n = X_n + t_n I and N_n = t_n I, where
// t_n = sum |X_n[r, c]| makes P_n positive semidefinite (lemma_psd_shifted_abs_sum). Z reads
// back the X_n, is positive semidefinite (the quadratic form splits by block), solves the
// unreduced and hence (lemma_reduced_of_input_solution) the reduced system, so A(Z) = b, and
// has the user's objective value.
//
// Without LMIs the variable is Z = X itself (`user_feasible_single`, `lemma_action_feasible_sound_single`,
// `lemma_user_feasible_lifts_single`).
//
// Approximate soundness, for the iterates the solver actually returns: the user-level
// residuals at vars_of(Z), namely sum_n A_n(X_n) - b and (H_i(X) - M_i) - B_i, are the
// residuals of the unreduced rows at Z (`lemma_user_residuals`). Without reduction these are
// entries of A(Z) - b (`lemma_approx_sound_unreduced`); with reduction they are bounded by the
// dense residual of A(Z) - b weighted by the Frobenius norms of the echelon rows, plus the
// reduction residual of the dropped rows (`lemma_approx_sound_reduced`, `reduction_bound_r`).
// The blocks B_i are the LMI blocks of the PSD partition, so a successful PSD check gives
// B_i >= -eps_i I (`lemma_lmi_blocks_checked`).

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_fundamental_div_mod_converse, lemma_div_multiples_vanish};
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::reduce_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::ldlt_spec::block_psd_ok;
#[cfg(verus_keep_ghost)]
use crate::real_spec::abs_r;
#[cfg(verus_keep_ghost)]
use crate::sdp_spec::*;
#[cfg(verus_keep_ghost)]
use crate::sdpad_spec::*;
#[cfg(verus_keep_ghost)]
use crate::sdp::*;
#[cfg(verus_keep_ghost)]
use crate::sdp_altdir::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- the user's problem ---------- */

/// Row e of the n-th map applied to the user variable xs[n]: A_n[e](X_n).
pub open spec fn user_term_r(maps: Seq<Vec<Vec<Vec<f64>>>>, vd: Seq<int>, e: int, xs: Seq<Seq<real>>, n: int) -> real {
    dot_r(symrow_r(maps[n]@, e, vd[n]), xs[n])
}

/// Row e of the sum of the maps (one per user variable): sum_n A_n[e](X_n).
pub open spec fn user_row_r(maps: Seq<Vec<Vec<Vec<f64>>>>, vd: Seq<int>, e: int, xs: Seq<Seq<real>>) -> real {
    sum_fn_r(|n: int| user_term_r(maps, vd, e, xs, n), vd.len())
}

/// Objective term n: C_n(X_n).
pub open spec fn user_obj_term_r(objective: Seq<Vec<Vec<f64>>>, vd: Seq<int>, xs: Seq<Seq<real>>, n: int) -> real {
    dot_r(part_r(objective[n]@, vd[n]), xs[n])
}

/// The objective C_1(X_1) + .. + C_n(X_n).
pub open spec fn user_obj_r(objective: Seq<Vec<Vec<f64>>>, vd: Seq<int>, xs: Seq<Seq<real>>) -> real {
    sum_fn_r(|n: int| user_obj_term_r(objective, vd, xs, n), vd.len())
}

/// The k x k slack matrix H_i(X) - M_i of an LMI with maps `maps` and constant matrix `m`
/// (column-major, k x k): entry (r, c) reads row c k + r of the maps and entry c k + r of m.
pub open spec fn user_lmi_view(maps: Seq<Vec<Vec<Vec<f64>>>>, m: Seq<f64>, vd: Seq<int>, k: int, xs: Seq<Seq<real>>) -> MatF {
    |r: int, c: int| user_row_r(maps, vd, c * k + r, xs) - rv(m[c * k + r])
}

/// X = (X_1, .., X_n) is feasible for the user's problem with LMIs: the variables are packed
/// matrices of the orders vd, the equality constraints hold, and every LMI slack matrix
/// H_i(X) - M_i is symmetric and positive semidefinite.
pub open spec fn user_feasible(sdp: &SDP, xs: Seq<Seq<real>>) -> bool {
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    &&& xs.len() == nv
    &&& forall|n: int| 0 <= n < nv ==> (#[trigger] xs[n]).len() == pk_len(vd[n])
    &&& forall|e: int| 0 <= e < sdp.constraint.1@.len()
            ==> #[trigger] user_row_r(sdp.constraint.0@, vd, e, xs) == rv(sdp.constraint.1@[e])
    &&& forall|i: int| 0 <= i < sdp.lmi@.len()
            ==> is_symmetric(#[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs), bd[i])
                && is_psd(user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs), bd[i])
}

/// X is feasible for the user's problem without LMIs: A(X) = b when a constraint map is given,
/// and X is positive semidefinite.
pub open spec fn user_feasible_single(sdp: &SDP, x: Seq<real>) -> bool {
    let d = sdp.objective@[0]@.len() as int;
    &&& x.len() == pk_len(d)
    &&& sdp.constraint.0@.len() == 1
            ==> forall|e: int| 0 <= e < sdp.constraint.1@.len()
                ==> #[trigger] dot_r(symrow_r(sdp.constraint.0@[0]@, e, d), x) == rv(sdp.constraint.1@[e])
    &&& is_psd(sym_view(x), d)
}

/// The user variables read from the standard-form variable z (read_solution, solution_ok):
/// X_n = zvar_r(z, var_off(bsum, vd, n), vd[n]).
pub open spec fn vars_of(z: Seq<real>, bsum: int, vd: Seq<int>) -> Seq<Seq<real>> {
    Seq::new(vd.len(), |n: int| zvar_r(z, var_off(bsum, vd, n), vd[n]))
}

/* ---------- the user maps at the variables read from z ---------- */

/// The rows of the maps at the variables read from z are the rows maps_row_r of the standard form.
pub proof fn lemma_user_row_of_vars(maps: Seq<Vec<Vec<Vec<f64>>>>, bsum: int, vd: Seq<int>, e: int, z: Seq<real>)
    ensures
        user_row_r(maps, vd, e, vars_of(z, bsum, vd)) == maps_row_r(maps, bsum, vd, e, z),
{
    let xs = vars_of(z, bsum, vd);
    let f = |n: int| user_term_r(maps, vd, e, xs, n);
    let g = |n: int| map_term_r(maps, bsum, vd, e, z, n);
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] f(n) == g(n) by {
        assert(xs[n] == zvar_r(z, var_off(bsum, vd, n), vd[n]));
    }
    lemma_sum_fn_ext(f, g, vd.len());
}

/// The objective at the variables read from z is the objective obj_r of the standard form.
pub proof fn lemma_user_obj_of_vars(objective: Seq<Vec<Vec<f64>>>, bsum: int, vd: Seq<int>, z: Seq<real>)
    ensures
        user_obj_r(objective, vd, vars_of(z, bsum, vd)) == obj_r(objective, bsum, vd, z),
{
    let xs = vars_of(z, bsum, vd);
    let f = |n: int| user_obj_term_r(objective, vd, xs, n);
    let g = |n: int| obj_term_r(objective, bsum, vd, z, n);
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] f(n) == g(n) by {
        assert(xs[n] == zvar_r(z, var_off(bsum, vd, n), vd[n]));
    }
    lemma_sum_fn_ext(f, g, vd.len());
}

/// The solution read by read_solution (solution_ok, with LMI blocks) is vars_of.
pub proof fn lemma_solution_vars(s: &SdpadSetup, z: Seq<real>, result: Seq<SymMatrix>)
    requires
        s.block_size_sum > 0,
        solution_ok(s, z, result),
    ensures
        Seq::new(result.len(), |n: int| rvs(result[n]@)) == vars_of(z, s.block_size_sum as int, dims_of(s.symm_dims@)),
{
    let vd = dims_of(s.symm_dims@);
    let bsum = s.block_size_sum as int;
    let got = Seq::new(result.len(), |n: int| rvs(result[n]@));
    assert forall|n: int| 0 <= n < vd.len() implies got[n] == #[trigger] vars_of(z, bsum, vd)[n] by {
        assert(rvs(result[n]@) == zvar_r(z, var_off(bsum, vd, n), vd[n]));
    }
    assert(got =~= vars_of(z, bsum, vd));
}

/* ---------- rows of the augmented systems ---------- */

/// Row i of an augmented system [M | rhs] (rows rows, cols + 1 columns) is satisfied by z
/// exactly when the row dot product equals the right-hand side.
pub proof fn lemma_aug_row_dot(aug: Seq<real>, rows: int, cols: int, i: int, z: Seq<real>)
    requires
        cols >= 0,
        z.len() == cols,
    ensures
        aug_row_ok(aug, rows, cols + 1, i, z) <==> row_dot_r(aug, rows, i, z) == cm_entry(aug, rows, i, cols),
{
    let f = |c: int| cm_entry(aug, rows, i, c) * z[c];
    let g = |n: int| constr_entry(aug, rows, i, n) * z[n];
    assert forall|c: int| 0 <= c < cols implies #[trigger] f(c) == g(c) by {}
    lemma_sum_fn_ext(f, g, cols as nat);
}

/// z solves the returned augmented system [mat | point[..rows]] exactly when every dense row's
/// dot product equals its right-hand side.
pub proof fn lemma_solves_result_aug(mat: Seq<f64>, point: Seq<f64>, rows: int, cols: int, z: Seq<real>)
    requires
        rows >= 0,
        cols >= 0,
        mat.len() == rows * cols,
        point.len() >= rows,
        z.len() == cols,
    ensures
        solves(result_aug(mat, point, rows), rows, cols + 1, z)
            <==> forall|i: int| 0 <= i < rows ==> #[trigger] row_dot_r(rvs(mat), rows, i, z) == rv(point[i]),
{
    let res = result_aug(mat, point, rows);
    assert(cols * rows == rows * cols);
    assert forall|i: int| 0 <= i < rows
        implies (#[trigger] aug_row_ok(res, rows, cols + 1, i, z) <==> row_dot_r(rvs(mat), rows, i, z) == rv(point[i])) by {
        lemma_aug_row_dot(res, rows, cols, i, z);
        let f = |n: int| constr_entry(res, rows, i, n) * z[n];
        let g = |n: int| constr_entry(rvs(mat), rows, i, n) * z[n];
        assert forall|n: int| 0 <= n < cols implies #[trigger] f(n) == g(n) by {
            lemma_result_aug_entry(mat, point, rows, cols, i, n);
        }
        lemma_sum_fn_ext(f, g, cols as nat);
        assert(cm_idx(rows, i, cols) == cols * rows + i);
        assert(res[cols * rows + i] == rvs(point.subrange(0, rows))[i]);
    }
    if solves(res, rows, cols + 1, z) {
        assert forall|i: int| 0 <= i < rows implies #[trigger] row_dot_r(rvs(mat), rows, i, z) == rv(point[i]) by {
            assert(aug_row_ok(res, rows, cols + 1, i, z));
        }
    }
    if forall|i: int| 0 <= i < rows ==> #[trigger] row_dot_r(rvs(mat), rows, i, z) == rv(point[i]) {
        assert forall|i: int| 0 <= i < rows implies #[trigger] aug_row_ok(res, rows, cols + 1, i, z) by {
            assert(row_dot_r(rvs(mat), rows, i, z) == rv(point[i]));
        }
    }
}

/// A(z) = b for the setup's operator exactly when every dense row's dot product equals its
/// right-hand side and z vanishes at every zero-pattern position (whose right-hand sides are 0).
pub proof fn lemma_setup_action_point(sdp: &SDP, check: bool, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_setup_ok(sdp, check, *s),
        z.len() == pk_len(s.var_dim as int),
    ensures
        setup_action_r(s, z) == rvs(s.point@)
            <==> (forall|i: int| 0 <= i < s.c_len ==> #[trigger] row_dot_r(rvs(s.mat_dense@), s.c_len as int, i, z) == rv(s.point@[i]))
                && (forall|q: int| 0 <= q < s.zeros@.len() ==> #[trigger] z[zero_idx(s.zeros@, q)] == 0.0real),
{
    let rows = s.c_len as int;
    let nz = s.zeros@.len() as int;
    let az = setup_action_r(s, z);
    let b = rvs(s.point@);
    if az == b {
        assert forall|i: int| 0 <= i < rows
            implies #[trigger] row_dot_r(rvs(s.mat_dense@), rows, i, z) == rv(s.point@[i]) by {
            assert(az[i] == b[i]);
        }
        assert forall|q: int| 0 <= q < nz implies #[trigger] z[zero_idx(s.zeros@, q)] == 0.0real by {
            assert(az[rows + q] == b[rows + q]);
            assert(rv(s.point@[rows + q]) == 0.0real);
        }
    }
    if (forall|i: int| 0 <= i < rows ==> #[trigger] row_dot_r(rvs(s.mat_dense@), rows, i, z) == rv(s.point@[i]))
        && (forall|q: int| 0 <= q < nz ==> #[trigger] z[zero_idx(s.zeros@, q)] == 0.0real) {
        assert forall|i: int| 0 <= i < rows + nz implies az[i] == b[i] by {
            if i < rows {
                assert(row_dot_r(rvs(s.mat_dense@), rows, i, z) == rv(s.point@[i]));
            } else {
                assert(z[zero_idx(s.zeros@, i - rows)] == 0.0real);
                assert(rv(s.point@[i]) == 0.0real);
            }
        }
        assert(az =~= b);
    }
}

/* ---------- shapes of the standard form with LMIs ---------- */

/// The block partition of the standard form with LMIs: nonnegative sizes adding up to the
/// order of the variable, the LMI blocks first, then the two blocks of every user variable.
pub proof fn lemma_block_sizes_facts(bd: Seq<int>, vd: Seq<int>)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
    ensures
        block_sizes(bd, vd).len() == bd.len() + 2 * vd.len(),
        forall|j: int| 0 <= j < block_sizes(bd, vd).len() ==> #[trigger] block_sizes(bd, vd)[j] >= 0,
        sum_upto(block_sizes(bd, vd), block_sizes(bd, vd).len() as int) == std_dim(bd, vd),
        forall|i: int| 0 <= i < bd.len()
            ==> #[trigger] sum_upto(block_sizes(bd, vd), i) == block_off(bd, i) && block_sizes(bd, vd)[i] == bd[i],
        forall|n: int| 0 <= n < vd.len()
            ==> #[trigger] sum_upto(block_sizes(bd, vd), bd.len() + 2 * n) == var_off(block_sum(bd), vd, n)
                && block_sizes(bd, vd)[bd.len() + 2 * n] == vd[n]
                && block_sizes(bd, vd)[bd.len() + 2 * n + 1] == vd[n]
                && sum_upto(block_sizes(bd, vd), bd.len() + 2 * n + 1) == var_off(block_sum(bd), vd, n) + vd[n],
        0 <= block_sum(bd) <= std_dim(bd, vd),
{
    let bs = block_sizes(bd, vd);
    let s = bd.len() as int;
    let tail = Seq::new(2 * vd.len(), |j: int| var_block_size(vd, j));
    lemma_sum_upto_monotone(bd, 0, bd.len() as int);
    assert(bs == bd + tail);
    assert forall|j: int| 0 <= j < bs.len() implies #[trigger] bs[j] >= 0 by {
        lemma_block_sizes_entry(bd, vd, j);
    }
    lemma_block_sizes_sum(bd, vd, vd.len() as int);
    assert forall|i: int| 0 <= i < bd.len()
        implies #[trigger] sum_upto(bs, i) == block_off(bd, i) && bs[i] == bd[i] by {
        lemma_sum_upto_prefix(bd, tail, i);
    }
    assert forall|n: int| 0 <= n < vd.len()
        implies #[trigger] sum_upto(bs, s + 2 * n) == var_off(block_sum(bd), vd, n)
            && bs[s + 2 * n] == vd[n]
            && bs[s + 2 * n + 1] == vd[n]
            && sum_upto(bs, s + 2 * n + 1) == var_off(block_sum(bd), vd, n) + vd[n] by {
        lemma_block_sizes_sum(bd, vd, n);
        lemma_block_sizes_entry(bd, vd, s + 2 * n);
        lemma_block_sizes_entry(bd, vd, s + 2 * n + 1);
        lemma_fundamental_div_mod_converse(2 * n, 2, n, 0);
        lemma_fundamental_div_mod_converse(2 * n + 1, 2, n, 1);
        assert(sum_upto(bs, s + 2 * n + 1) == sum_upto(bs, s + 2 * n) + bs[s + 2 * n]);
    }
    lemma_sum_upto_monotone(vd, 0, vd.len() as int);
}

/// The LMI rows of LMI i lie below lmi_rows(bd), and the entry c k + r of its k x k block is a
/// row index of the LMI addressing the entries (r, c).
pub proof fn lemma_lmi_row_index(bd: Seq<int>, i: int, r: int, c: int)
    requires
        forall|j: int| 0 <= j < bd.len() ==> #[trigger] bd[j] >= 1,
        0 <= i < bd.len(),
        0 <= r < bd[i],
        0 <= c < bd[i],
    ensures
        0 <= c * bd[i] + r < bd[i] * bd[i],
        0 <= row_off(bd, i),
        row_off(bd, i) + bd[i] * bd[i] <= lmi_rows(bd),
        lmi_row_lo(bd[i], c * bd[i] + r) == (if r <= c { r } else { c }),
        lmi_row_hi(bd[i], c * bd[i] + r) == (if r <= c { c } else { r }),
{
    let k = bd[i];
    assert(0 <= c * k + r < k * k) by (nonlinear_arith)
        requires 0 <= r < k, 0 <= c < k;
    lemma_sq_sum_upto_monotone(bd, 0, i);
    lemma_sq_sum_upto_monotone(bd, i + 1, bd.len() as int);
    lemma_lmi_row_of(k, c, r);
}

/* ---------- soundness: a feasible standard-form point gives a feasible user point ---------- */

/// What a feasible standard-form point z establishes for the user's problem: z is block
/// diagonal, the variables vars_of(z) are feasible, every LMI block of z is the slack matrix
/// H_i(X) - M_i (and positive semidefinite), and <C_F, z>_F is the user's objective value.
pub open spec fn standard_point_feasible(sdp: &SDP, s: &SdpadSetup, z: Seq<real>) -> bool {
    let d = s.var_dim as int;
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, n_vars(sdp));
    let bsum = block_sum(bd);
    let xs = vars_of(z, bsum, vd);
    &&& forall|r: int, c: int| 0 <= r < c < d && in_zero_region(block_sizes(bd, vd), r, c) ==> z[#[trigger] pk_idx(r, c)] == 0.0real
    &&& user_feasible(sdp, xs)
    &&& forall|i: int| 0 <= i < bd.len()
            ==> meq(#[trigger] block_view(z, block_off(bd, i)), user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs), bd[i], bd[i])
                && is_psd(block_view(z, block_off(bd, i)), bd[i])
    &&& prim_val_r(s, z) == user_obj_r(sdp.objective@, vd, xs)
}

/// Soundness with LMIs: a packed z of order d that solves every row of the unreduced system aug
/// of std_rows_ok, vanishes on the zero pattern and is positive semidefinite is a feasible
/// standard-form point (standard_point_feasible).
pub proof fn lemma_standard_feasible_sound(sdp: &SDP, check: bool, s: &SdpadSetup, aug: Seq<real>, z: Seq<real>)
    requires
        sdpad_input_ok(sdp, check),
        sdpad_setup_ok(sdp, check, *s),
        sdp.lmi@.len() >= 1,
        std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)), aug),
        z.len() == pk_len(s.var_dim as int),
        solves(aug, lmi_rows(bdims_r(sdp.lmi@)) + sdp.constraint.1@.len() as int, pk_len(s.var_dim as int) + 1, z),
        forall|q: int| 0 <= q < s.zeros@.len() ==> #[trigger] z[zero_idx(s.zeros@, q)] == 0.0real,
        is_psd(sym_view(z), s.var_dim as int),
    ensures
        standard_point_feasible(sdp, s, z),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let bs = block_sizes(bd, vd);
    let m = sdp.constraint.1@.len() as int;
    let rows = lmi_rows(bd) + m;
    let xs = vars_of(z, bsum, vd);
    let list = s.zeros@;
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    assert(d == std_dim(bd, vd));
    assert(zeros_ok(list, bs, d));
    assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 1 by {}
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] vd[n] >= 1 by {}
    lemma_block_sizes_facts(bd, vd);
    lemma_pk_col_start_monotone(0, d);

    // z vanishes off the diagonal blocks: every off-block position is listed
    assert forall|r: int, c: int| 0 <= r < c < d && in_zero_region(bs, r, c)
        implies z[#[trigger] pk_idx(r, c)] == 0.0real by {
        assert(listed(list, r, c));
        let q = choose|q: int| 0 <= q < list.len() && (#[trigger] list[q]).0 == r && list[q].1 == c;
        assert(z[zero_idx(list, q)] == 0.0real);
    }

    // the variables are packed matrices of the orders vd
    assert forall|n: int| 0 <= n < nv implies (#[trigger] xs[n]).len() == pk_len(vd[n]) by {
        lemma_pk_col_start_monotone(0, vd[n]);
    }

    // the equality rows
    assert forall|e: int| 0 <= e < m
        implies #[trigger] user_row_r(sdp.constraint.0@, vd, e, xs) == rv(sdp.constraint.1@[e]) by {
        lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
        assert(aug_row_ok(aug, rows, cols + 1, lmi_rows(bd) + e, z));
        lemma_aug_row_dot(aug, rows, cols, lmi_rows(bd) + e, z);
        assert(row_dot_r(aug, rows, lmi_rows(bd) + e, z) == maps_row_r(sdp.constraint.0@, bsum, vd, e, z));
        lemma_user_row_of_vars(sdp.constraint.0@, bsum, vd, e, z);
    }

    // the LMI blocks: block i of z is the slack matrix H_i(X) - M_i, and it is PSD
    assert forall|i: int| 0 <= i < bd.len()
        implies meq(#[trigger] block_view(z, block_off(bd, i)), user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs), bd[i], bd[i])
            && is_psd(block_view(z, block_off(bd, i)), bd[i]) by {
        let k = bd[i];
        let off = block_off(bd, i);
        let w = user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, k, xs);
        let bv = block_view(z, off);
        assert forall|r: int, c: int| #![trigger bv(r, c)] #![trigger w(r, c)]
            0 <= r < k && 0 <= c < k implies bv(r, c) == w(r, c) by {
            lemma_lmi_row_index(bd, i, r, c);
            let e = c * k + r;
            assert(aug_row_ok(aug, rows, cols + 1, row_off(bd, i) + e, z));
            lemma_aug_row_dot(aug, rows, cols, row_off(bd, i) + e, z);
            assert(row_dot_r(aug, rows, row_off(bd, i) + e, z)
                == maps_row_r(sdp.lmi@[i].0@, bsum, vd, e, z) - sym_entry(z, off + lmi_row_lo(k, e), off + lmi_row_hi(k, e)));
            assert(cm_entry(aug, rows, row_off(bd, i) + e, cols) == rv(sdp.lmi@[i].1@[e]));
            lemma_user_row_of_vars(sdp.lmi@[i].0@, bsum, vd, e, z);
        }
        lemma_sum_upto_monotone(bd, 0, i);
        lemma_sum_upto_monotone(bd, i + 1, bd.len() as int);
        lemma_psd_shifted_block(z, d, off, k, 0.0real);
    }

    // the slack matrices are symmetric and PSD
    assert forall|i: int| 0 <= i < sdp.lmi@.len()
        implies is_symmetric(#[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs), bd[i])
            && is_psd(user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs), bd[i]) by {
        let k = bd[i];
        let w = user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, k, xs);
        let bv = block_view(z, block_off(bd, i));
        assert(meq(bv, w, k, k));
        assert forall|r: int, c: int| 0 <= r < k && 0 <= c < k implies #[trigger] w(r, c) == w(c, r) by {
            assert(w(r, c) == bv(r, c));
            assert(w(c, r) == bv(c, r));
        }
        lemma_psd_shifted_meq(bv, w, k, 0.0real);
    }

    // the objective
    lemma_objective_value(sdp, check, s, z);
    lemma_user_obj_of_vars(sdp.objective@, bsum, vd, z);
}

/// Without constraint reduction, A(Z) = b and Z >= 0 give a feasible standard-form point: Z
/// solves the unreduced system, which the setup carries unchanged.
pub proof fn lemma_action_feasible_sound(sdp: &SDP, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_input_ok(sdp, false),
        sdpad_setup_ok(sdp, false, *s),
        sdp.lmi@.len() >= 1,
        z.len() == pk_len(s.var_dim as int),
        setup_action_r(s, z) == rvs(s.point@),
        is_psd(sym_view(z), s.var_dim as int),
    ensures
        standard_point_feasible(sdp, s, z),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let m = sdp.constraint.1@.len() as int;
    let total = lmi_rows(bd) + m;
    let rows = s.c_len as int;
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), false)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    let aug = choose|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
        && result_aug(s.mat_dense@, s.point@, rows) == aug;
    lemma_pk_col_start_monotone(0, d);
    lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
    lemma_reduced_rows(aug, s.mat_dense@, s.point@, rows, total, d, false);
    lemma_setup_action_point(sdp, false, s, z);
    lemma_solves_result_aug(s.mat_dense@, s.point@, rows, cols, z);
    lemma_standard_feasible_sound(sdp, false, s, aug, z);
}

/// Without LMIs and without constraint reduction, A(Z) = b and Z >= 0 make Z itself feasible
/// for the user's problem, with <C_F, Z>_F its objective value.
pub proof fn lemma_action_feasible_sound_single(sdp: &SDP, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_input_ok(sdp, false),
        sdpad_setup_ok(sdp, false, *s),
        sdp.lmi@.len() == 0,
        z.len() == pk_len(s.var_dim as int),
        setup_action_r(s, z) == rvs(s.point@),
        is_psd(sym_view(z), s.var_dim as int),
    ensures
        user_feasible_single(sdp, z),
        prim_val_r(s, z) == dot_r(part_r(sdp.objective@[0]@, sdp.objective@[0]@.len() as int), z),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let m = sdp.constraint.1@.len() as int;
    let rows = s.c_len as int;
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), false)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    assert(d == sdp.objective@[0]@.len() as int);
    lemma_pk_col_start_monotone(0, d);
    lemma_setup_action_point(sdp, false, s, z);
    if sdp.constraint.0@.len() == 1 {
        let aug = choose|aug: Seq<real>| #[trigger] single_rows_ok(sdp.constraint.0@[0]@, sdp.constraint.1@, d, aug)
            && result_aug(s.mat_dense@, s.point@, rows) == aug;
        lemma_reduced_rows(aug, s.mat_dense@, s.point@, rows, m, d, false);
        lemma_solves_result_aug(s.mat_dense@, s.point@, rows, cols, z);
        assert forall|e: int| 0 <= e < m
            implies #[trigger] dot_r(symrow_r(sdp.constraint.0@[0]@, e, d), z) == rv(sdp.constraint.1@[e]) by {
            assert(aug_row_ok(aug, m, cols + 1, e, z));
            lemma_aug_row_dot(aug, m, cols, e, z);
            assert(row_dot_r(aug, m, e, z) == dot_r(symrow_r(sdp.constraint.0@[0]@, e, d), z));
        }
    }
    lemma_objective_value(sdp, false, s, z);
}

/* ---------- the block partition ---------- */

/// The block of row r in the partition bs: the j with sum_upto(bs, j) <= r < sum_upto(bs, j + 1).
pub open spec fn block_of(bs: Seq<int>, r: int) -> int {
    choose|j: int| 0 <= j < bs.len() && #[trigger] sum_upto(bs, j) <= r < sum_upto(bs, j + 1)
}

/// Every row below sum_upto(bs, j) lies in one of the first j blocks.
pub proof fn lemma_block_of_exists(bs: Seq<int>, j: int, r: int)
    requires
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] bs[i] >= 0,
        0 <= j <= bs.len(),
        0 <= r < sum_upto(bs, j),
    ensures
        exists|i: int| 0 <= i < j && #[trigger] sum_upto(bs, i) <= r < sum_upto(bs, i + 1),
    decreases j,
{
    if r < sum_upto(bs, j - 1) {
        lemma_block_of_exists(bs, j - 1, r);
    } else {
        assert(sum_upto(bs, j - 1) <= r < sum_upto(bs, j));
    }
}

/// The block of a row is unique.
pub proof fn lemma_block_of_unique(bs: Seq<int>, j: int, r: int)
    requires
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] bs[i] >= 0,
        0 <= j < bs.len(),
        sum_upto(bs, j) <= r < sum_upto(bs, j + 1),
    ensures
        block_of(bs, r) == j,
{
    let i = block_of(bs, r);
    assert(0 <= i < bs.len() && sum_upto(bs, i) <= r < sum_upto(bs, i + 1));
    if i < j {
        lemma_sum_upto_monotone(bs, i + 1, j);
    } else if i > j {
        lemma_sum_upto_monotone(bs, j + 1, i);
    }
}

/// A packed matrix that vanishes at the off-block-diagonal positions of the partition bs and
/// whose diagonal blocks satisfy B_i >= -eps I satisfies Z >= -eps I on the leading
/// sum_upto(bs, j) rows: the quadratic form splits block by block (lemma_psd_shifted_two_blocks).
pub proof fn lemma_block_diag_psd_shifted(z: Seq<real>, bs: Seq<int>, j: int, eps: real)
    requires
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] bs[i] >= 0,
        0 <= j <= bs.len(),
        forall|r: int, c: int| 0 <= r < c < sum_upto(bs, bs.len() as int) && in_zero_region(bs, r, c)
            ==> #[trigger] sym_entry(z, r, c) == 0.0real,
        forall|i: int| 0 <= i < bs.len() ==> #[trigger] psd_shifted(block_view(z, sum_upto(bs, i)), bs[i], eps),
    ensures
        psd_shifted(sym_view(z), sum_upto(bs, j), eps),
    decreases j,
{
    let a = sym_view(z);
    if j == 0 {
        lemma_psd_shifted_zero_dim(a, eps);
    } else {
        lemma_block_diag_psd_shifted(z, bs, j - 1, eps);
        let k = sum_upto(bs, j - 1);
        let m = bs[j - 1];
        lemma_sum_upto_monotone(bs, 0, j - 1);
        lemma_sum_upto_monotone(bs, j, bs.len() as int);
        // the rows of the first j - 1 blocks are not coupled to block j - 1
        assert forall|r: int, c: int| #![trigger a(r, c)] #![trigger a(c, r)]
            0 <= r < k && k <= c < k + m implies a(r, c) == 0.0real && a(c, r) == 0.0real by {
            lemma_block_of_exists(bs, j - 1, r);
            let i = choose|i: int| 0 <= i < j - 1 && #[trigger] sum_upto(bs, i) <= r < sum_upto(bs, i + 1);
            lemma_sum_upto_monotone(bs, i + 1, j - 1);
            assert(in_zero_region(bs, r, c));
            assert(sym_entry(z, r, c) == 0.0real);
            assert(sym_entry(z, c, r) == z[pk_idx(r, c)]);
        }
        // block j - 1 is the shifted view of a at k
        let sh = shift_view(a, k);
        let bv = block_view(z, k);
        assert forall|r: int, c: int| 0 <= r < m && 0 <= c < m implies #[trigger] bv(r, c) == #[trigger] sh(r, c) by {}
        assert(psd_shifted(bv, m, eps));
        lemma_psd_shifted_meq(bv, sh, m, eps);
        lemma_psd_shifted_two_blocks(a, k, m, eps);
        assert(sum_upto(bs, j) == k + m);
    }
}

/* ---------- the lifted standard-form variable ---------- */

/// The shift t_n = sum_{r, c} |X_n[r, c]| that makes X_n + t_n I positive semidefinite.
pub open spec fn lift_shift_r(xs: Seq<Seq<real>>, vd: Seq<int>, n: int) -> real {
    abs_sum_r(xs[n], vd[n])
}

/// Entry (a, b) of block j of the lifted variable: the slack matrix H_j(X) - M_j for j < s, the
/// shifted variable X_n + t_n I for j = s + 2 n, and the shift t_n I for j = s + 2 n + 1.
pub open spec fn lift_block_entry(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>, j: int, a: int, b: int) -> real {
    let s = bd.len() as int;
    if j < s {
        user_lmi_view(lmis[j].0@, lmis[j].1@, vd, bd[j], xs)(a, b)
    } else {
        let n = (j - s) / 2;
        let t = if a == b { lift_shift_r(xs, vd, n) } else { 0.0real };
        if (j - s) % 2 == 0 { sym_entry(xs[n], a, b) + t } else { t }
    }
}

/// Entry (r, c), r <= c, of the lifted variable: zero at the off-block-diagonal positions,
/// otherwise the entry of the block containing r. Opaque: unfolding in_zero_region at every
/// entry would flood the prover; lemma_lift_z_entry exposes the two cases.
#[verifier::opaque]
pub open spec fn lift_entry(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>, r: int, c: int) -> real {
    let bs = block_sizes(bd, vd);
    if in_zero_region(bs, r, c) {
        0.0real
    } else {
        let j = block_of(bs, r);
        let st = sum_upto(bs, j);
        lift_block_entry(lmis, xs, bd, vd, j, r - st, c - st)
    }
}

/// The lifted variable: the packed matrix of order std_dim(bd, vd) with entries lift_entry.
pub open spec fn lift_z(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>) -> Seq<real> {
    Seq::new(pk_len(std_dim(bd, vd)) as nat, |t: int| lift_entry(lmis, xs, bd, vd, pk_unpack(t).0, pk_unpack(t).1))
}

/// The slack matrices of all LMIs are symmetric (the hypothesis the block lemmas need).
pub open spec fn slacks_symmetric(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>) -> bool {
    forall|i: int| 0 <= i < bd.len() ==> is_symmetric(#[trigger] user_lmi_view(lmis[i].0@, lmis[i].1@, vd, bd[i], xs), bd[i])
}

/// The entry (r, c), r <= c, of the lifted variable is lift_entry(r, c).
pub proof fn lemma_lift_z_entry(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>, r: int, c: int)
    requires
        0 <= r <= c < std_dim(bd, vd),
    ensures
        lift_z(lmis, xs, bd, vd)[pk_idx(r, c)] == lift_entry(lmis, xs, bd, vd, r, c),
        in_zero_region(block_sizes(bd, vd), r, c) ==> lift_z(lmis, xs, bd, vd)[pk_idx(r, c)] == 0.0real,
        !in_zero_region(block_sizes(bd, vd), r, c)
            ==> lift_z(lmis, xs, bd, vd)[pk_idx(r, c)]
                == lift_block_entry(lmis, xs, bd, vd, block_of(block_sizes(bd, vd), r),
                    r - sum_upto(block_sizes(bd, vd), block_of(block_sizes(bd, vd), r)),
                    c - sum_upto(block_sizes(bd, vd), block_of(block_sizes(bd, vd), r))),
{
    reveal(lift_entry);
    lemma_pk_idx_bound(r, c, std_dim(bd, vd));
    lemma_pk_unpack_idx(r, c);
}

/// Every block of the lifted variable is symmetric when the slack matrices are.
pub proof fn lemma_lift_block_sym(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>, j: int, a: int, b: int)
    requires
        slacks_symmetric(lmis, xs, bd, vd),
        0 <= j < bd.len() + 2 * vd.len(),
        0 <= a < block_sizes(bd, vd)[j],
        0 <= b < block_sizes(bd, vd)[j],
    ensures
        lift_block_entry(lmis, xs, bd, vd, j, a, b) == lift_block_entry(lmis, xs, bd, vd, j, b, a),
{
    lemma_block_sizes_entry(bd, vd, j);
    if j < bd.len() {
        let w = user_lmi_view(lmis[j].0@, lmis[j].1@, vd, bd[j], xs);
        assert(w(a, b) == w(b, a));
    }
}

/// On block j (offset st = sum_upto(bs, j)) the lifted variable reads its block entries:
/// sym_entry(Z, st + a, st + b) = lift_block_entry(j, a, b).
pub proof fn lemma_lift_z_block(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>, j: int, a: int, b: int)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        slacks_symmetric(lmis, xs, bd, vd),
        0 <= j < bd.len() + 2 * vd.len(),
        0 <= a < block_sizes(bd, vd)[j],
        0 <= b < block_sizes(bd, vd)[j],
    ensures
        sym_entry(lift_z(lmis, xs, bd, vd), sum_upto(block_sizes(bd, vd), j) + a, sum_upto(block_sizes(bd, vd), j) + b)
            == lift_block_entry(lmis, xs, bd, vd, j, a, b),
{
    let bs = block_sizes(bd, vd);
    let st = sum_upto(bs, j);
    lemma_block_sizes_facts(bd, vd);
    lemma_sum_upto_monotone(bs, 0, j);
    lemma_sum_upto_monotone(bs, j + 1, bs.len() as int);
    assert(sum_upto(bs, j + 1) == st + bs[j]);
    lemma_block_of_unique(bs, j, st + a);
    lemma_block_of_unique(bs, j, st + b);
    if a <= b {
        lemma_lift_z_entry(lmis, xs, bd, vd, st + a, st + b);
        lemma_block_not_zero_region(bs, j, st + a, st + b);
    } else {
        lemma_lift_z_entry(lmis, xs, bd, vd, st + b, st + a);
        lemma_block_not_zero_region(bs, j, st + b, st + a);
        lemma_lift_block_sym(lmis, xs, bd, vd, j, a, b);
    }
}

/// The lifted variable reads back the user variables: zvar_r(Z, var_off(bsum, vd, n), vd[n]) = X_n.
pub proof fn lemma_lift_z_vars(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>, n: int)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        slacks_symmetric(lmis, xs, bd, vd),
        0 <= n < vd.len(),
        xs[n].len() == pk_len(vd[n]),
    ensures
        zvar_r(lift_z(lmis, xs, bd, vd), var_off(block_sum(bd), vd, n), vd[n]) == xs[n],
{
    let bs = block_sizes(bd, vd);
    let z = lift_z(lmis, xs, bd, vd);
    let s = bd.len() as int;
    let d = vd[n];
    let st = var_off(block_sum(bd), vd, n);
    let (j1, j2) = (s + 2 * n, s + 2 * n + 1);
    lemma_block_sizes_facts(bd, vd);
    lemma_fundamental_div_mod_converse(2 * n, 2, n, 0);
    lemma_fundamental_div_mod_converse(2 * n + 1, 2, n, 1);
    lemma_pk_col_start_monotone(0, d);
    let zv = zvar_r(z, st, d);
    assert forall|t: int| 0 <= t < pk_len(d) implies #[trigger] zv[t] == xs[n][t] by {
        lemma_pk_unpack(t);
        let (a, b) = (pk_unpack(t).0, pk_unpack(t).1);
        lemma_pk_idx_lt_len(a, b, d);
        lemma_lift_z_block(lmis, xs, bd, vd, j1, a, b);
        lemma_lift_z_block(lmis, xs, bd, vd, j2, a, b);
        assert(blk_r(z, st, d)[t] == z[pk_idx(st + a, st + b)]);
        assert(blk_r(z, st + d, d)[t] == z[pk_idx(st + d + a, st + d + b)]);
        assert(sym_entry(z, st + a, st + b) == z[pk_idx(st + a, st + b)]);
        assert(sym_entry(z, st + d + a, st + d + b) == z[pk_idx(st + d + a, st + d + b)]);
        assert(sym_entry(xs[n], a, b) == xs[n][t]);
    }
    assert(zv =~= xs[n]);
}

/// The lifted variable is positive semidefinite: the slack blocks are PSD by hypothesis, the
/// blocks X_n + t_n I by the shift bound, the blocks t_n I as nonnegative multiples of I, and
/// the quadratic form splits by block.
pub proof fn lemma_lift_z_psd(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        slacks_symmetric(lmis, xs, bd, vd),
        forall|i: int| 0 <= i < bd.len() ==> is_psd(#[trigger] user_lmi_view(lmis[i].0@, lmis[i].1@, vd, bd[i], xs), bd[i]),
    ensures
        is_psd(sym_view(lift_z(lmis, xs, bd, vd)), std_dim(bd, vd)),
{
    let bs = block_sizes(bd, vd);
    let z = lift_z(lmis, xs, bd, vd);
    let s = bd.len() as int;
    lemma_block_sizes_facts(bd, vd);
    // the off-block entries vanish
    assert forall|r: int, c: int| 0 <= r < c < sum_upto(bs, bs.len() as int) && in_zero_region(bs, r, c)
        implies #[trigger] sym_entry(z, r, c) == 0.0real by {
        lemma_lift_z_entry(lmis, xs, bd, vd, r, c);
    }
    // every block is PSD
    assert forall|j: int| 0 <= j < bs.len()
        implies #[trigger] psd_shifted(block_view(z, sum_upto(bs, j)), bs[j], 0.0real) by {
        let st = sum_upto(bs, j);
        let k = bs[j];
        let bv = block_view(z, st);
        assert forall|a: int, b: int| 0 <= a < k && 0 <= b < k
            implies #[trigger] bv(a, b) == lift_block_entry(lmis, xs, bd, vd, j, a, b) by {
            lemma_lift_z_block(lmis, xs, bd, vd, j, a, b);
        }
        if j < s {
            let w = user_lmi_view(lmis[j].0@, lmis[j].1@, vd, bd[j], xs);
            assert forall|a: int, b: int| 0 <= a < k && 0 <= b < k implies #[trigger] w(a, b) == #[trigger] bv(a, b) by {}
            lemma_psd_shifted_meq(w, bv, k, 0.0real);
        } else {
            let n = (j - s) / 2;
            lemma_block_sizes_entry(bd, vd, j);
            let t = lift_shift_r(xs, vd, n);
            lemma_abs_sum_nonneg(xs[n], vd[n]);
            if (j - s) % 2 == 0 {
                let sh = madd(sym_view(xs[n]), mscale(t, mident()));
                lemma_psd_shifted_abs_sum(xs[n], vd[n]);
                lemma_psd_shifted_iff(sym_view(xs[n]), vd[n], t);
                assert forall|a: int, b: int| 0 <= a < k && 0 <= b < k implies #[trigger] sh(a, b) == #[trigger] bv(a, b) by {
                    if a == b {
                        assert(sh(a, b) == sym_entry(xs[n], a, b) + t * 1.0real);
                    } else {
                        assert(sh(a, b) == sym_entry(xs[n], a, b) + t * 0.0real);
                    }
                }
                lemma_psd_shifted_meq(sh, bv, k, 0.0real);
            } else {
                let dv = const_r(k as nat, t);
                let dg = diag_view(dv);
                lemma_diag_psd(dv, k);
                assert forall|a: int, b: int| 0 <= a < k && 0 <= b < k implies #[trigger] dg(a, b) == #[trigger] bv(a, b) by {}
                lemma_psd_shifted_meq(dg, bv, k, 0.0real);
            }
        }
    }
    lemma_block_diag_psd_shifted(z, bs, bs.len() as int, 0.0real);
}

/// The lifted variable reads back all user variables.
pub proof fn lemma_lift_z_vars_all(lmis: Seq<(Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>)>, xs: Seq<Seq<real>>, bd: Seq<int>, vd: Seq<int>)
    requires
        forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
        forall|n: int| 0 <= n < vd.len() ==> #[trigger] vd[n] >= 0,
        slacks_symmetric(lmis, xs, bd, vd),
        xs.len() == vd.len(),
        forall|n: int| 0 <= n < vd.len() ==> (#[trigger] xs[n]).len() == pk_len(vd[n]),
    ensures
        vars_of(lift_z(lmis, xs, bd, vd), block_sum(bd), vd) == xs,
{
    let z = lift_z(lmis, xs, bd, vd);
    let bsum = block_sum(bd);
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] vars_of(z, bsum, vd)[n] == xs[n] by {
        lemma_lift_z_vars(lmis, xs, bd, vd, n);
    }
    assert(vars_of(z, bsum, vd) =~= xs);
}

/// The shape facts of the LMI input used by the row lemmas: positive orders, and the lifted
/// variable of order std_dim reading back the variables.
pub proof fn lemma_lift_z_shapes(sdp: &SDP, xs: Seq<Seq<real>>)
    requires
        sdp_lmi_input_ok(sdp),
        user_feasible(sdp, xs),
    ensures
        forall|i: int| 0 <= i < bdims_r(sdp.lmi@).len() ==> #[trigger] bdims_r(sdp.lmi@)[i] >= 1,
        forall|n: int| 0 <= n < vdims_r(sdp.lmi@, n_vars(sdp)).len() ==> #[trigger] vdims_r(sdp.lmi@, n_vars(sdp))[n] >= 1,
        std_dim(bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp))) >= 0,
        lift_z(sdp.lmi@, xs, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp))).len()
            == pk_len(std_dim(bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))),
        vars_of(lift_z(sdp.lmi@, xs, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp))), block_sum(bdims_r(sdp.lmi@)), vdims_r(sdp.lmi@, n_vars(sdp))) == xs,
        lmi_rows(bdims_r(sdp.lmi@)) >= 0,
{
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 1 by {}
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] vd[n] >= 1 by {}
    lemma_block_sizes_facts(bd, vd);
    lemma_pk_col_start_monotone(0, std_dim(bd, vd));
    lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
    lemma_lift_z_vars_all(sdp.lmi@, xs, bd, vd);
}

/// The lifted variable satisfies LMI row e of LMI i0 of the unreduced system: the row reads
/// H_i0(X)[e] - B_i0[e] with B_i0 the slack block, so it equals M_i0[e].
pub proof fn lemma_lift_z_lmi_row(sdp: &SDP, xs: Seq<Seq<real>>, aug: Seq<real>, i0: int, e: int)
    requires
        sdp_lmi_input_ok(sdp),
        user_feasible(sdp, xs),
        std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)), aug),
        0 <= i0 < bdims_r(sdp.lmi@).len(),
        0 <= e < bdims_r(sdp.lmi@)[i0] * bdims_r(sdp.lmi@)[i0],
    ensures
        aug_row_ok(aug, lmi_rows(bdims_r(sdp.lmi@)) + sdp.constraint.1@.len() as int,
            pk_len(std_dim(bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))) + 1, row_off(bdims_r(sdp.lmi@), i0) + e,
            lift_z(sdp.lmi@, xs, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))),
{
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let sd = std_dim(bd, vd);
    let cols = pk_len(sd);
    let m = sdp.constraint.1@.len() as int;
    let rows = lmi_rows(bd) + m;
    let lmis = sdp.lmi@;
    let z = lift_z(lmis, xs, bd, vd);
    lemma_lift_z_shapes(sdp, xs);
    lemma_block_sizes_facts(bd, vd);
    let k = bd[i0];
    let off = block_off(bd, i0);
    lemma_lmi_row_bounds(k, e);
    lemma_fundamental_div_mod(e, k);
    let (r, c) = (e % k, e / k);
    lemma_lmi_row_index(bd, i0, r, c);
    assert(c * k + r == e);
    let (lo, hi) = (lmi_row_lo(k, e), lmi_row_hi(k, e));
    lemma_lift_z_block(lmis, xs, bd, vd, i0, lo, hi);
    lemma_aug_row_dot(aug, rows, cols, row_off(bd, i0) + e, z);
    assert(row_dot_r(aug, rows, row_off(bd, i0) + e, z)
        == maps_row_r(lmis[i0].0@, bsum, vd, e, z) - sym_entry(z, off + lo, off + hi));
    assert(cm_entry(aug, rows, row_off(bd, i0) + e, cols) == rv(lmis[i0].1@[e]));
    lemma_user_row_of_vars(lmis[i0].0@, bsum, vd, e, z);
    let w = user_lmi_view(lmis[i0].0@, lmis[i0].1@, vd, k, xs);
    assert(w(lo, hi) == w(r, c));
    assert(w(r, c) == user_row_r(lmis[i0].0@, vd, e, xs) - rv(lmis[i0].1@[e]));
}

/// The lifted variable satisfies equality row e of the unreduced system: the row reads
/// sum_n A_n(X_n), which equals b[e].
pub proof fn lemma_lift_z_eq_row(sdp: &SDP, xs: Seq<Seq<real>>, aug: Seq<real>, e: int)
    requires
        sdp_lmi_input_ok(sdp),
        user_feasible(sdp, xs),
        std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)), aug),
        0 <= e < sdp.constraint.1@.len(),
    ensures
        aug_row_ok(aug, lmi_rows(bdims_r(sdp.lmi@)) + sdp.constraint.1@.len() as int,
            pk_len(std_dim(bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))) + 1, lmi_rows(bdims_r(sdp.lmi@)) + e,
            lift_z(sdp.lmi@, xs, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))),
{
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let cols = pk_len(std_dim(bd, vd));
    let m = sdp.constraint.1@.len() as int;
    let rows = lmi_rows(bd) + m;
    let z = lift_z(sdp.lmi@, xs, bd, vd);
    lemma_lift_z_shapes(sdp, xs);
    lemma_aug_row_dot(aug, rows, cols, lmi_rows(bd) + e, z);
    assert(row_dot_r(aug, rows, lmi_rows(bd) + e, z) == maps_row_r(sdp.constraint.0@, bsum, vd, e, z));
    assert(cm_entry(aug, rows, lmi_rows(bd) + e, cols) == rv(sdp.constraint.1@[e]));
    lemma_user_row_of_vars(sdp.constraint.0@, bsum, vd, e, z);
}

/// The lifted variable solves every row of the unreduced system.
pub proof fn lemma_lift_z_solves(sdp: &SDP, xs: Seq<Seq<real>>, aug: Seq<real>)
    requires
        sdp_lmi_input_ok(sdp),
        user_feasible(sdp, xs),
        std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)), aug),
    ensures
        solves(aug, lmi_rows(bdims_r(sdp.lmi@)) + sdp.constraint.1@.len() as int, pk_len(std_dim(bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))) + 1,
            lift_z(sdp.lmi@, xs, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))),
{
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let cols = pk_len(std_dim(bd, vd));
    let m = sdp.constraint.1@.len() as int;
    let rows = lmi_rows(bd) + m;
    let z = lift_z(sdp.lmi@, xs, bd, vd);
    assert forall|i: int| 0 <= i < rows implies #[trigger] aug_row_ok(aug, rows, cols + 1, i, z) by {
        if i < lmi_rows(bd) {
            lemma_lmi_row_decomp(bd, bd.len() as int, i);
            let i0 = choose|i0: int| 0 <= i0 < bd.len() && #[trigger] row_off(bd, i0) <= i < row_off(bd, i0) + bd[i0] * bd[i0];
            lemma_lift_z_lmi_row(sdp, xs, aug, i0, i - row_off(bd, i0));
        } else {
            lemma_lift_z_eq_row(sdp, xs, aug, i - lmi_rows(bd));
        }
    }
}

/// A solution of the stacked system solves the returned (possibly reduced) system of the
/// conversion: every dense row's dot product equals its right-hand side
/// (lemma_reduced_of_input_solution for the reduced case).
pub proof fn lemma_solution_reduces(aug: Seq<real>, mat: Seq<f64>, point: Seq<f64>, rows: int, total: int, dim: int, check: bool, z: Seq<real>)
    requires
        rows >= 0,
        total >= 0,
        dim >= 0,
        mat.len() == rows * pk_len(dim),
        point.len() >= rows,
        aug.len() == total * (pk_len(dim) + 1),
        if check { reduced_from(aug, result_aug(mat, point, rows), dim) } else { result_aug(mat, point, rows) == aug },
        z.len() == pk_len(dim),
        solves(aug, total, pk_len(dim) + 1, z),
    ensures
        forall|i: int| 0 <= i < rows ==> #[trigger] row_dot_r(rvs(mat), rows, i, z) == rv(point[i]),
{
    let cols = pk_len(dim);
    let cd = red_col_dim(dim);
    lemma_pk_col_start_monotone(0, dim);
    lemma_reduced_rows(aug, mat, point, rows, total, dim, check);
    let res = result_aug(mat, point, rows);
    if check {
        let e = choose|e: Seq<real>| reduced_system(aug, e, res, dim);
        let pivots = choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(aug, e, res, dim, pivots);
        assert(aug.len() as int == cd * total) by (nonlinear_arith)
            requires aug.len() as int == total * cd;
        lemma_div_multiples_vanish(total, cd);
        lemma_reduced_of_input_solution(aug, e, res, dim, pivots, z);
        let rank = pivots.len() as int;
        assert(res.len() == rows * cols + rows);
        assert(rows * cols + rows == rows * (cols + 1)) by (nonlinear_arith);
        assert(rank == rows) by (nonlinear_arith)
            requires cd >= 1, cd * rank == rows * cd;
    }
    lemma_solves_result_aug(mat, point, rows, cols, z);
}

/// The lifted variable satisfies A(Z) = b for the setup's operator: it solves the unreduced
/// system, hence the returned one, and vanishes on the zero pattern.
pub proof fn lemma_lift_z_action(sdp: &SDP, check: bool, s: &SdpadSetup, xs: Seq<Seq<real>>)
    requires
        sdpad_input_ok(sdp, check),
        sdpad_setup_ok(sdp, check, *s),
        sdp.lmi@.len() >= 1,
        user_feasible(sdp, xs),
    ensures
        setup_action_r(s, lift_z(sdp.lmi@, xs, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))) == rvs(s.point@),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bs = block_sizes(bd, vd);
    let m = sdp.constraint.1@.len() as int;
    let total = lmi_rows(bd) + m;
    let rows = s.c_len as int;
    let list = s.zeros@;
    let z = lift_z(sdp.lmi@, xs, bd, vd);
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    let aug = choose|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
        && (if check { reduced_from(aug, result_aug(s.mat_dense@, s.point@, rows), d) } else { result_aug(s.mat_dense@, s.point@, rows) == aug });
    assert(d == std_dim(bd, vd));
    assert(zeros_ok(list, bs, d));
    lemma_pk_col_start_monotone(0, d);
    lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
    lemma_lift_z_solves(sdp, xs, aug);
    lemma_solution_reduces(aug, s.mat_dense@, s.point@, rows, total, d, check, z);
    // the zero pattern
    assert forall|q: int| 0 <= q < list.len() implies #[trigger] z[zero_idx(list, q)] == 0.0real by {
        let (r0, c0) = (list[q].0 as int, list[q].1 as int);
        assert(in_zero_region(bs, r0, c0));
        lemma_lift_z_entry(sdp.lmi@, xs, bd, vd, r0, c0);
    }
    lemma_setup_action_point(sdp, check, s, z);
}

/// Converse with LMIs: feasible user variables lift to the standard-form variable lift_z, a
/// packed matrix of order d that reads back the variables, is positive semidefinite and block
/// diagonal, solves every row of the unreduced system, satisfies A(Z) = b for the setup's
/// operator (with and without reduction), and has the user's objective value.
pub proof fn lemma_user_feasible_lifts(sdp: &SDP, check: bool, s: &SdpadSetup, xs: Seq<Seq<real>>)
    requires
        sdpad_input_ok(sdp, check),
        sdpad_setup_ok(sdp, check, *s),
        sdp.lmi@.len() >= 1,
        user_feasible(sdp, xs),
    ensures
        ({
            let bd = bdims_r(sdp.lmi@);
            let vd = vdims_r(sdp.lmi@, n_vars(sdp));
            let z = lift_z(sdp.lmi@, xs, bd, vd);
            &&& z.len() == pk_len(s.var_dim as int)
            &&& vars_of(z, block_sum(bd), vd) == xs
            &&& is_psd(sym_view(z), s.var_dim as int)
            &&& forall|r: int, c: int| 0 <= r < c < s.var_dim && in_zero_region(block_sizes(bd, vd), r, c)
                    ==> z[#[trigger] pk_idx(r, c)] == 0.0real
            &&& forall|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
                    ==> solves(aug, lmi_rows(bd) + sdp.constraint.1@.len() as int, pk_len(s.var_dim as int) + 1, z)
            &&& setup_action_r(s, z) == rvs(s.point@)
            &&& prim_val_r(s, z) == user_obj_r(sdp.objective@, vd, xs)
        }),
{
    let d = s.var_dim as int;
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let bs = block_sizes(bd, vd);
    let lmis = sdp.lmi@;
    let z = lift_z(lmis, xs, bd, vd);
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    assert(d == std_dim(bd, vd));
    assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 1 by {}
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] vd[n] >= 1 by {}
    lemma_block_sizes_facts(bd, vd);
    lemma_pk_col_start_monotone(0, d);
    // the variables are read back
    lemma_lift_z_vars_all(lmis, xs, bd, vd);
    // positive semidefinite and block diagonal
    lemma_lift_z_psd(lmis, xs, bd, vd);
    assert forall|r: int, c: int| 0 <= r < c < d && in_zero_region(bs, r, c)
        implies z[#[trigger] pk_idx(r, c)] == 0.0real by {
        lemma_lift_z_entry(lmis, xs, bd, vd, r, c);
    }
    // the rows and the constraint operator
    assert forall|aug: Seq<real>| #[trigger] std_rows_ok(lmis, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
        implies solves(aug, lmi_rows(bd) + sdp.constraint.1@.len() as int, pk_len(d) + 1, z) by {
        lemma_lift_z_solves(sdp, xs, aug);
    }
    lemma_lift_z_action(sdp, check, s, xs);
    // the objective
    lemma_objective_value(sdp, check, s, z);
    lemma_user_obj_of_vars(sdp.objective@, bsum, vd, z);
}

/// Without LMIs a feasible X is itself the standard-form variable: A(X) = b for the setup's
/// operator (with and without reduction), X >= 0, and <C_F, X>_F is the objective value.
pub proof fn lemma_user_feasible_lifts_single(sdp: &SDP, check: bool, s: &SdpadSetup, x: Seq<real>)
    requires
        sdpad_input_ok(sdp, check),
        sdpad_setup_ok(sdp, check, *s),
        sdp.lmi@.len() == 0,
        user_feasible_single(sdp, x),
    ensures
        x.len() == pk_len(s.var_dim as int),
        setup_action_r(s, x) == rvs(s.point@),
        is_psd(sym_view(x), s.var_dim as int),
        prim_val_r(s, x) == dot_r(part_r(sdp.objective@[0]@, sdp.objective@[0]@.len() as int), x),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let m = sdp.constraint.1@.len() as int;
    let rows = s.c_len as int;
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    assert(d == sdp.objective@[0]@.len() as int);
    lemma_pk_col_start_monotone(0, d);
    if sdp.constraint.0@.len() == 1 {
        let aug = choose|aug: Seq<real>| #[trigger] single_rows_ok(sdp.constraint.0@[0]@, sdp.constraint.1@, d, aug)
            && (if check { reduced_from(aug, result_aug(s.mat_dense@, s.point@, rows), d) } else { result_aug(s.mat_dense@, s.point@, rows) == aug });
        assert forall|e: int| 0 <= e < m implies #[trigger] aug_row_ok(aug, m, cols + 1, e, x) by {
            lemma_aug_row_dot(aug, m, cols, e, x);
            assert(row_dot_r(aug, m, e, x) == dot_r(symrow_r(sdp.constraint.0@[0]@, e, d), x));
            assert(cm_entry(aug, m, e, cols) == rv(sdp.constraint.1@[e]));
        }
        lemma_solution_reduces(aug, s.mat_dense@, s.point@, rows, m, d, check, x);
    } else {
        assert(rows == 0);
    }
    lemma_setup_action_point(sdp, check, s, x);
    lemma_objective_value(sdp, check, s, x);
}

/* ---------- approximate soundness: residuals ---------- */

/// The residual A(z) - b of the setup's constraint operator at z.
pub open spec fn action_residual_r(s: &SdpadSetup, z: Seq<real>) -> Seq<real> {
    vsub_r(setup_action_r(s, z), rvs(s.point@))
}

/// The residual of row i of the unreduced standard-form system aug at z.
pub open spec fn std_residual_r(sdp: &SDP, aug: Seq<real>, i: int, z: Seq<real>) -> real {
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, n_vars(sdp));
    row_residual_r(aug, lmi_rows(bd) + sdp.constraint.1@.len() as int, pk_len(std_dim(bd, vd)) + 1, i, z)
}

/// The user-level residuals at the variables read from z are bounded by bnd: those of the
/// equality constraints, and those of the LMI entries against the LMI blocks of z.
pub open spec fn user_residuals_bounded(sdp: &SDP, s: &SdpadSetup, z: Seq<real>, bnd: real) -> bool {
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, n_vars(sdp));
    let xs = vars_of(z, block_sum(bd), vd);
    &&& forall|e: int| 0 <= e < sdp.constraint.1@.len()
            ==> abs_r(#[trigger] user_row_r(sdp.constraint.0@, vd, e, xs) - rv(sdp.constraint.1@[e])) <= bnd
    &&& forall|i: int, r: int, c: int| 0 <= i < bd.len() && 0 <= r < bd[i] && 0 <= c < bd[i]
            ==> abs_r(#[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs)(r, c)
                - block_view(z, block_off(bd, i))(r, c)) <= bnd
}

/// The bound of the reduced case for the echelon form e of the unreduced system aug:
/// rows * sum_{l < rank} ||E_l||_F |(A(z) - b)_l| + (rows - rank) * 2^rank * eps_G * (1 + ||vec z||_1),
/// with rows the number of unreduced rows, rank = c_len the number of kept rows, and
/// eps_G = TOL * max |entry of aug| the elimination tolerance.
pub open spec fn reduction_bound_r(sdp: &SDP, s: &SdpadSetup, e: Seq<real>, aug: Seq<real>, z: Seq<real>) -> real {
    let bd = bdims_r(sdp.lmi@);
    let d = s.var_dim as int;
    let rows = lmi_rows(bd) + sdp.constraint.1@.len() as int;
    let rank = s.c_len as nat;
    let res = action_residual_r(s, z);
    (rows as real) * sum_fn_r(|l: int| row_scale_r(e, rows, d, l) * abs_r(res[l]), rank)
        + ((rows - rank) as real) * (pow2_r(rank) * gauss_eps_r(aug) * (1.0real + one_norm_r(z)))
}

/// The residual of a row is its dot product less its right-hand side.
pub proof fn lemma_residual_row_dot(aug: Seq<real>, rows: int, cols: int, i: int, z: Seq<real>)
    requires
        cols >= 0,
        z.len() == cols,
    ensures
        row_residual_r(aug, rows, cols + 1, i, z) == row_dot_r(aug, rows, i, z) - cm_entry(aug, rows, i, cols),
{
    let f = |c: int| cm_entry(aug, rows, i, c) * z[c];
    let g = |n: int| constr_entry(aug, rows, i, n) * z[n];
    assert forall|c: int| 0 <= c < cols implies #[trigger] f(c) == g(c) by {}
    lemma_sum_fn_ext(f, g, cols as nat);
}

/// The entries of A(z) - b: the dense rows' residuals, then the zero-pattern entries of z.
pub proof fn lemma_action_residual_entries(sdp: &SDP, check: bool, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_setup_ok(sdp, check, *s),
        z.len() == pk_len(s.var_dim as int),
    ensures
        action_residual_r(s, z).len() == s.point@.len(),
        forall|i: int| 0 <= i < s.c_len
            ==> #[trigger] action_residual_r(s, z)[i] == row_dot_r(rvs(s.mat_dense@), s.c_len as int, i, z) - rv(s.point@[i]),
        forall|q: int| 0 <= q < s.zeros@.len()
            ==> z[#[trigger] zero_idx(s.zeros@, q)] == action_residual_r(s, z)[s.c_len + q],
{
    let rows = s.c_len as int;
    let res = action_residual_r(s, z);
    assert forall|q: int| 0 <= q < s.zeros@.len()
        implies z[#[trigger] zero_idx(s.zeros@, q)] == res[rows + q] by {
        assert(rv(s.point@[rows + q]) == 0.0real);
    }
}

/// The residual of a row of the returned system [mat | point] is the dense row's dot product
/// less its right-hand side.
pub proof fn lemma_result_aug_residual(mat: Seq<f64>, point: Seq<f64>, rows: int, cols: int, z: Seq<real>, i: int)
    requires
        rows >= 0,
        cols >= 0,
        mat.len() == rows * cols,
        point.len() >= rows,
        z.len() == cols,
        0 <= i < rows,
    ensures
        row_residual_r(result_aug(mat, point, rows), rows, cols + 1, i, z) == row_dot_r(rvs(mat), rows, i, z) - rv(point[i]),
{
    let res = result_aug(mat, point, rows);
    assert(cols * rows == rows * cols);
    lemma_residual_row_dot(res, rows, cols, i, z);
    let f = |n: int| constr_entry(res, rows, i, n) * z[n];
    let g = |n: int| constr_entry(rvs(mat), rows, i, n) * z[n];
    assert forall|n: int| 0 <= n < cols implies #[trigger] f(n) == g(n) by {
        lemma_result_aug_entry(mat, point, rows, cols, i, n);
    }
    lemma_sum_fn_ext(f, g, cols as nat);
    assert(cm_idx(rows, i, cols) == cols * rows + i);
    assert(res[cols * rows + i] == rvs(point.subrange(0, rows))[i]);
}

/// The user-level residuals at the variables read from z are the residuals of the unreduced
/// rows: sum_n A_n(X_n)_e - b_e is that of equality row e, and (H_i(X) - M_i)_rc - (B_i)_rc that
/// of LMI row c k_i + r, for every z.
pub proof fn lemma_user_residuals(sdp: &SDP, aug: Seq<real>, z: Seq<real>)
    requires
        sdp_lmi_input_ok(sdp),
        std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)), aug),
        z.len() == pk_len(std_dim(bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)))),
    ensures
        ({
            let bd = bdims_r(sdp.lmi@);
            let vd = vdims_r(sdp.lmi@, n_vars(sdp));
            let xs = vars_of(z, block_sum(bd), vd);
            &&& forall|e: int| 0 <= e < sdp.constraint.1@.len()
                    ==> #[trigger] user_row_r(sdp.constraint.0@, vd, e, xs) - rv(sdp.constraint.1@[e])
                        == std_residual_r(sdp, aug, lmi_rows(bd) + e, z)
            &&& forall|i: int, r: int, c: int| 0 <= i < bd.len() && 0 <= r < bd[i] && 0 <= c < bd[i]
                    ==> #[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs)(r, c)
                        == block_view(z, block_off(bd, i))(r, c) + std_residual_r(sdp, aug, row_off(bd, i) + c * bd[i] + r, z)
        }),
{
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let cols = pk_len(std_dim(bd, vd));
    let m = sdp.constraint.1@.len() as int;
    let rows = lmi_rows(bd) + m;
    let xs = vars_of(z, bsum, vd);
    assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 1 by {}
    assert forall|e: int| 0 <= e < m
        implies #[trigger] user_row_r(sdp.constraint.0@, vd, e, xs) - rv(sdp.constraint.1@[e])
            == std_residual_r(sdp, aug, lmi_rows(bd) + e, z) by {
        lemma_residual_row_dot(aug, rows, cols, lmi_rows(bd) + e, z);
        assert(row_dot_r(aug, rows, lmi_rows(bd) + e, z) == maps_row_r(sdp.constraint.0@, bsum, vd, e, z));
        assert(cm_entry(aug, rows, lmi_rows(bd) + e, cols) == rv(sdp.constraint.1@[e]));
        lemma_user_row_of_vars(sdp.constraint.0@, bsum, vd, e, z);
    }
    assert forall|i: int, r: int, c: int| 0 <= i < bd.len() && 0 <= r < bd[i] && 0 <= c < bd[i]
        implies #[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs)(r, c)
            == block_view(z, block_off(bd, i))(r, c) + std_residual_r(sdp, aug, row_off(bd, i) + c * bd[i] + r, z) by {
        let k = bd[i];
        let off = block_off(bd, i);
        lemma_lmi_row_index(bd, i, r, c);
        let e = c * k + r;
        lemma_residual_row_dot(aug, rows, cols, row_off(bd, i) + e, z);
        assert(row_dot_r(aug, rows, row_off(bd, i) + e, z)
            == maps_row_r(sdp.lmi@[i].0@, bsum, vd, e, z) - sym_entry(z, off + lmi_row_lo(k, e), off + lmi_row_hi(k, e)));
        assert(cm_entry(aug, rows, row_off(bd, i) + e, cols) == rv(sdp.lmi@[i].1@[e]));
        lemma_user_row_of_vars(sdp.lmi@[i].0@, bsum, vd, e, z);
    }
}

/// The LMI blocks of z are blocks of the PSD partition, so a successful PSD check establishes
/// B_i >= -eps_i I with the implementation's tolerance eps_i (block_psd_ok).
pub proof fn lemma_lmi_blocks_checked(sdp: &SDP, check: bool, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_input_ok(sdp, check),
        sdpad_setup_ok(sdp, check, *s),
        sdp.lmi@.len() >= 1,
        psd_blocks_ok(s, z),
    ensures
        forall|i: int| 0 <= i < bdims_r(sdp.lmi@).len() ==> #[trigger] block_psd_ok(z, block_off(bdims_r(sdp.lmi@), i), bdims_r(sdp.lmi@)[i]),
{
    let d = s.var_dim as int;
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bs = block_sizes(bd, vd);
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    assert(dims_of(s.block_dims@) == bd);
    assert(dims_of(s.symm_dims@) == vd);
    assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 1 by {}
    assert forall|n: int| 0 <= n < vd.len() implies #[trigger] vd[n] >= 1 by {}
    lemma_block_sizes_facts(bd, vd);
    lemma_sum_upto_ge(bd, bd.len() as int);
    assert(block_sum(bd) >= 1);
    assert(psd_partition(bd, vd, d) == bs);
    assert forall|i: int| 0 <= i < bd.len() implies #[trigger] block_psd_ok(z, block_off(bd, i), bd[i]) by {
        assert(s.psd_blocks@[i].0 as int == sum_upto(bs, i));
        assert(s.psd_blocks@[i].1 as int == bs[i]);
        assert(block_psd_ok(z, s.psd_blocks@[i].0 as int, s.psd_blocks@[i].1 as int));
    }
}

/// Approximate soundness without reduction: the user-level residuals at the variables read
/// from z are entries of A(z) - b (the dense rows carry the unreduced rows), the zero-pattern
/// entries of z are the remaining entries, and <C_F, z>_F is the user's objective.
pub proof fn lemma_approx_sound_unreduced(sdp: &SDP, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_input_ok(sdp, false),
        sdpad_setup_ok(sdp, false, *s),
        sdp.lmi@.len() >= 1,
        z.len() == pk_len(s.var_dim as int),
    ensures
        ({
            let bd = bdims_r(sdp.lmi@);
            let vd = vdims_r(sdp.lmi@, n_vars(sdp));
            let xs = vars_of(z, block_sum(bd), vd);
            let res = action_residual_r(s, z);
            &&& res.len() == s.point@.len()
            &&& forall|q: int| 0 <= q < s.zeros@.len() ==> z[#[trigger] zero_idx(s.zeros@, q)] == res[s.c_len + q]
            &&& forall|e: int| 0 <= e < sdp.constraint.1@.len()
                    ==> #[trigger] user_row_r(sdp.constraint.0@, vd, e, xs) - rv(sdp.constraint.1@[e]) == res[lmi_rows(bd) + e]
            &&& forall|i: int, r: int, c: int| 0 <= i < bd.len() && 0 <= r < bd[i] && 0 <= c < bd[i]
                    ==> #[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs)(r, c)
                        == block_view(z, block_off(bd, i))(r, c) + res[row_off(bd, i) + c * bd[i] + r]
            &&& prim_val_r(s, z) == user_obj_r(sdp.objective@, vd, xs)
        }),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let m = sdp.constraint.1@.len() as int;
    let total = lmi_rows(bd) + m;
    let rows = s.c_len as int;
    let xs = vars_of(z, bsum, vd);
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), false)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    let aug = choose|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
        && result_aug(s.mat_dense@, s.point@, rows) == aug;
    assert(d == std_dim(bd, vd));
    lemma_pk_col_start_monotone(0, d);
    lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
    lemma_reduced_rows(aug, s.mat_dense@, s.point@, rows, total, d, false);
    lemma_user_residuals(sdp, aug, z);
    lemma_action_residual_entries(sdp, false, s, z);
    // the unreduced rows are the dense rows: their residuals are the entries of A(z) - b
    assert forall|i: int| 0 <= i < rows implies #[trigger] std_residual_r(sdp, aug, i, z) == action_residual_r(s, z)[i] by {
        lemma_result_aug_residual(s.mat_dense@, s.point@, rows, cols, z, i);
    }
    assert forall|i: int, r: int, c: int| 0 <= i < bd.len() && 0 <= r < bd[i] && 0 <= c < bd[i]
        implies #[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs)(r, c)
            == block_view(z, block_off(bd, i))(r, c) + action_residual_r(s, z)[row_off(bd, i) + c * bd[i] + r] by {
        lemma_lmi_row_index(bd, i, r, c);
        assert(std_residual_r(sdp, aug, row_off(bd, i) + c * bd[i] + r, z) == action_residual_r(s, z)[row_off(bd, i) + c * bd[i] + r]);
    }
    lemma_objective_value(sdp, false, s, z);
    lemma_user_obj_of_vars(sdp.objective@, bsum, vd, z);
}

/// Approximate soundness with reduction: for the echelon form e of the unreduced system aug
/// (reduced_with), the user-level residuals at the variables read from z are bounded by
/// reduction_bound_r (the dense residual of A(z) - b, weighted by the Frobenius norms of the
/// echelon rows, plus the reduction residual of the dropped rows); the zero-pattern entries of z
/// are the remaining entries of A(z) - b, and <C_F, z>_F is the user's objective.
pub proof fn lemma_approx_sound_reduced(sdp: &SDP, s: &SdpadSetup, z: Seq<real>)
    requires
        sdpad_input_ok(sdp, true),
        sdpad_setup_ok(sdp, true, *s),
        sdp.lmi@.len() >= 1,
        z.len() == pk_len(s.var_dim as int),
    ensures
        action_residual_r(s, z).len() == s.point@.len(),
        forall|q: int| 0 <= q < s.zeros@.len() ==> z[#[trigger] zero_idx(s.zeros@, q)] == action_residual_r(s, z)[s.c_len + q],
        prim_val_r(s, z) == user_obj_r(sdp.objective@, vdims_r(sdp.lmi@, n_vars(sdp)), vars_of(z, block_sum(bdims_r(sdp.lmi@)), vdims_r(sdp.lmi@, n_vars(sdp)))),
        exists|e: Seq<real>, aug: Seq<real>, pivots: Seq<int>|
            #[trigger] reduced_with(aug, e, result_aug(s.mat_dense@, s.point@, s.c_len as int), s.var_dim as int, pivots)
            && std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bdims_r(sdp.lmi@), vdims_r(sdp.lmi@, n_vars(sdp)), aug)
            && user_residuals_bounded(sdp, s, z, reduction_bound_r(sdp, s, e, aug, z)),
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let cd = red_col_dim(d);
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let xs = vars_of(z, bsum, vd);
    let m = sdp.constraint.1@.len() as int;
    let total = lmi_rows(bd) + m;
    let rows = s.c_len as int;
    let obj0 = choose|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), true)
        && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d);
    let red = result_aug(s.mat_dense@, s.point@, rows);
    let aug = choose|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
        && reduced_from(aug, red, d);
    let e = choose|e: Seq<real>| reduced_system(aug, e, red, d);
    let pivots = choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(aug, e, red, d, pivots);
    assert(d == std_dim(bd, vd));
    lemma_pk_col_start_monotone(0, d);
    lemma_sq_sum_upto_monotone(bd, 0, bd.len() as int);
    lemma_reduced_rows(aug, s.mat_dense@, s.point@, rows, total, d, true);
    // the row counts of the echelon form and of the reduced system
    assert(aug.len() as int == cd * total);
    lemma_div_multiples_vanish(total, cd);
    assert(red_row_dim(aug, d) == total);
    let rank = pivots.len() as int;
    assert(red.len() == rows * cols + rows);
    assert(rows * cols + rows == rows * (cols + 1)) by (nonlinear_arith);
    assert(rank == rows) by (nonlinear_arith)
        requires cd >= 1, cd * rank == rows * cd;
    lemma_user_residuals(sdp, aug, z);
    lemma_action_residual_entries(sdp, true, s, z);
    let res = action_residual_r(s, z);
    let bnd = reduction_bound_r(sdp, s, e, aug, z);
    // the residuals of the reduced rows are the dense entries of A(z) - b
    let f = |l: int| row_scale_r(e, total, d, l) * abs_r(row_residual_r(red, rank, cd, l, z));
    let g = |l: int| row_scale_r(e, total, d, l) * abs_r(res[l]);
    assert forall|l: int| 0 <= l < rank implies #[trigger] f(l) == g(l) by {
        lemma_result_aug_residual(s.mat_dense@, s.point@, rows, cols, z, l);
    }
    lemma_sum_fn_ext(f, g, pivots.len());
    // every unreduced row is bounded
    assert forall|i: int| 0 <= i < total implies abs_r(#[trigger] std_residual_r(sdp, aug, i, z)) <= bnd by {
        lemma_reduced_residual_bound(aug, e, red, d, pivots, z, i);
    }
    assert forall|ee: int| 0 <= ee < m
        implies abs_r(#[trigger] user_row_r(sdp.constraint.0@, vd, ee, xs) - rv(sdp.constraint.1@[ee])) <= bnd by {
        assert(abs_r(std_residual_r(sdp, aug, lmi_rows(bd) + ee, z)) <= bnd);
    }
    assert forall|i: int, r: int, c: int| 0 <= i < bd.len() && 0 <= r < bd[i] && 0 <= c < bd[i]
        implies abs_r(#[trigger] user_lmi_view(sdp.lmi@[i].0@, sdp.lmi@[i].1@, vd, bd[i], xs)(r, c)
            - block_view(z, block_off(bd, i))(r, c)) <= bnd by {
        lemma_lmi_row_index(bd, i, r, c);
        assert(abs_r(std_residual_r(sdp, aug, row_off(bd, i) + c * bd[i] + r, z)) <= bnd);
    }
    assert(user_residuals_bounded(sdp, s, z, bnd));
    lemma_objective_value(sdp, true, s, z);
    lemma_user_obj_of_vars(sdp.objective@, bsum, vd, z);
}

/// Each entry of A(z) - b is at most pinf * (1 + ||b||), the reported primal infeasibility
/// scaled back to an absolute residual.
pub proof fn lemma_residual_entry_pinf(s: &SdpadSetup, z: Seq<real>, l: int)
    requires
        setup_action_r(s, z).len() == s.point@.len(),
        0 <= l < s.point@.len(),
    ensures
        abs_r(action_residual_r(s, z)[l]) <= pinf_r(s, z) * (1.0real + norm_r(rvs(s.point@))),
{
    let res = action_residual_r(s, z);
    let b = rvs(s.point@);
    lemma_entry_le_norm(res, l);
    lemma_dot_self_nonneg(b);
    let den = 1.0real + norm_r(b);
    assert(pinf_r(s, z) == norm_r(res) / den);
    assert((norm_r(res) / den) * den == norm_r(res)) by (nonlinear_arith)
        requires den != 0.0real;
}

} // verus!
