// Spectral-part selection (not trusted).
//
// The spectral operations of jacobi.rs run one of the two eigensolvers on A,
// obtaining (vals, Q) with A = Q B Q^T, Q orthogonal, vals = diag(B) and
// ||B - diag(B)||_F <= eps (eigen_spec.rs), and keep the pairs
// (vals[k], column k of Q) whose value passes a selection test:
//
//     positive part:  vals[k] > 0,
//     negative part:  vals[k] < -1e-8 * scale,  scale = max(max_t |A_t|, 1).
//
// This module defines the list of selected indices, the predicate relating
// the returned (values, columns) to (vals, Q), and proves what the
// reconstruction Q_sel D_sel Q_sel^T (computed by diag_scale_gram) is:
//
//     Q_sel D_sel Q_sel^T = Q diag(masked vals) Q^T,
//
// with the unselected values replaced by 0. For the positive part this
// matrix is positive semidefinite. An empty negative part means that every
// returned value is >= -1e-8 * scale, hence Q diag(vals) Q^T >= -1e-8 * scale I.
// Nothing here claims that B is diagonal or that the selected pairs are
// exact eigenpairs of A.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_mod_multiples_basic, lemma_div_multiples_vanish};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::rows_of;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::eigen_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- selected indices ---------- */

/// The indices k < n with sel(k), in increasing order.
pub open spec fn sel_indices(sel: spec_fn(int) -> bool, n: nat) -> Seq<int>
    decreases n,
{
    if n == 0 {
        Seq::empty()
    } else {
        let prev = sel_indices(sel, (n - 1) as nat);
        if sel(n - 1) { prev.push(n - 1) } else { prev }
    }
}

/// The selection test of the positive part: vals[k] > 0.
pub open spec fn pos_sel(vals: Seq<real>) -> spec_fn(int) -> bool {
    |k: int| vals[k] > 0.0real
}

/// The tolerance of the negative part: 1e-8 * scale.
pub open spec fn neg_tol_r(scale: real) -> real {
    0.00000001real * scale
}

/// The selection test of the negative part: vals[k] < -1e-8 * scale.
pub open spec fn neg_sel(vals: Seq<real>, scale: real) -> spec_fn(int) -> bool {
    |k: int| vals[k] < -neg_tol_r(scale)
}

/// vals with every unselected entry replaced by 0.
pub open spec fn masked_r(vals: Seq<real>, sel: spec_fn(int) -> bool) -> Seq<real> {
    Seq::new(vals.len(), |k: int| if sel(k) { vals[k] } else { 0.0real })
}

/// The selected indices are exactly the k < n with sel(k), listed in increasing order.
pub proof fn lemma_sel_indices(sel: spec_fn(int) -> bool, n: nat)
    ensures
        sel_indices(sel, n).len() <= n,
        forall|t: int| 0 <= t < sel_indices(sel, n).len()
            ==> 0 <= #[trigger] sel_indices(sel, n)[t] < n && sel(sel_indices(sel, n)[t]),
        forall|t: int, u: int| 0 <= t < u < sel_indices(sel, n).len()
            ==> #[trigger] sel_indices(sel, n)[t] < #[trigger] sel_indices(sel, n)[u],
        forall|k: int| 0 <= k < n && #[trigger] sel(k)
            ==> exists|t: int| 0 <= t < sel_indices(sel, n).len() && #[trigger] sel_indices(sel, n)[t] == k,
    decreases n,
{
    if n > 0 {
        let m = (n - 1) as nat;
        let prev = sel_indices(sel, m);
        let idx = sel_indices(sel, n);
        lemma_sel_indices(sel, m);
        if sel(n - 1) {
            assert(idx == prev.push(n - 1));
            assert forall|t: int| 0 <= t < idx.len()
                implies 0 <= #[trigger] idx[t] < n && sel(idx[t]) by {
                if t < prev.len() {
                    assert(idx[t] == prev[t]);
                } else {
                    assert(idx[t] == n - 1);
                }
            }
            assert forall|t: int, u: int| 0 <= t < u < idx.len()
                implies #[trigger] idx[t] < #[trigger] idx[u] by {
                if u < prev.len() {
                    assert(idx[t] == prev[t] && idx[u] == prev[u]);
                } else {
                    assert(idx[t] == prev[t] && prev[t] < m);
                    assert(idx[u] == n - 1);
                }
            }
            assert forall|k: int| 0 <= k < n && #[trigger] sel(k)
                implies exists|t: int| 0 <= t < idx.len() && #[trigger] idx[t] == k by {
                if k < n - 1 {
                    let t = choose|t: int| 0 <= t < prev.len() && #[trigger] prev[t] == k;
                    assert(idx[t] == k);
                } else {
                    assert(idx[prev.len() as int] == k);
                }
            }
        } else {
            assert(idx == prev);
            assert forall|k: int| 0 <= k < n && #[trigger] sel(k)
                implies exists|t: int| 0 <= t < idx.len() && #[trigger] idx[t] == k by {
                assert(k < n - 1);
            }
        }
    }
}

/// The selection is empty if and only if no index below n is selected.
pub proof fn lemma_sel_indices_empty(sel: spec_fn(int) -> bool, n: nat)
    ensures
        sel_indices(sel, n).len() == 0 <==> forall|k: int| 0 <= k < n ==> !(#[trigger] sel(k)),
    decreases n,
{
    if n > 0 {
        lemma_sel_indices_empty(sel, (n - 1) as nat);
        if sel(n - 1) {
            assert(sel_indices(sel, n).len() > 0);
        } else {
            assert(sel_indices(sel, n) == sel_indices(sel, (n - 1) as nat));
            if forall|k: int| 0 <= k < n - 1 ==> !(#[trigger] sel(k)) {
                assert forall|k: int| 0 <= k < n implies !(#[trigger] sel(k)) by {}
            }
        }
    }
}

/* ---------- sums over the selected indices ---------- */

/// f evaluated along an index list: t |-> f(idx[t]).
pub open spec fn sel_terms(f: spec_fn(int) -> real, idx: Seq<int>) -> spec_fn(int) -> real {
    |t: int| f(idx[t])
}

/// f on the selected indices and 0 elsewhere.
pub open spec fn masked_terms(f: spec_fn(int) -> real, sel: spec_fn(int) -> bool) -> spec_fn(int) -> real {
    |k: int| if sel(k) { f(k) } else { 0.0real }
}

/// The sum of f over the selected indices below n is the sum over all indices below n
/// with the unselected terms masked to 0.
pub proof fn lemma_sum_selected(f: spec_fn(int) -> real, sel: spec_fn(int) -> bool, n: nat)
    ensures
        sum_fn_r(sel_terms(f, sel_indices(sel, n)), sel_indices(sel, n).len())
            == sum_fn_r(masked_terms(f, sel), n),
    decreases n,
{
    if n > 0 {
        let m = (n - 1) as nat;
        lemma_sum_selected(f, sel, m);
        let prev = sel_indices(sel, m);
        let idx = sel_indices(sel, n);
        if sel(n - 1) {
            assert(idx == prev.push(n - 1));
            let g = sel_terms(f, idx);
            let h = sel_terms(f, prev);
            assert forall|t: int| 0 <= t < prev.len() implies #[trigger] g(t) == h(t) by {
                assert(idx[t] == prev[t]);
            }
            lemma_sum_fn_ext(g, h, prev.len());
            assert(sum_fn_r(g, idx.len()) == sum_fn_r(g, prev.len()) + g(prev.len() as int));
            assert(g(prev.len() as int) == f(n - 1));
            assert(masked_terms(f, sel)(n - 1) == f(n - 1));
        } else {
            assert(idx == prev);
            assert(masked_terms(f, sel)(n - 1) == 0.0real);
        }
    }
}

/* ---------- the selected eigenpairs ---------- */

/// (sel_vals, sel_vects) lists the pairs (vals[k], column k of Q) for the selected indices
/// k < n in increasing order: sel_vals[t] = vals[idx[t]] and column t of sel_vects (dim rows,
/// column-major) is column idx[t] of Q, for idx = sel_indices(sel, n).
pub open spec fn selected_pairs(q: Seq<real>, vals: Seq<real>, sel: spec_fn(int) -> bool, n: int, sel_vals: Seq<real>, sel_vects: Seq<real>, dim: int) -> bool {
    let idx = sel_indices(sel, n as nat);
    &&& sel_vals.len() == idx.len()
    &&& sel_vects.len() == idx.len() * dim
    &&& forall|t: int| 0 <= t < idx.len() ==> #[trigger] sel_vals[t] == vals[idx[t]]
    &&& forall|t: int, r: int| 0 <= t < idx.len() && 0 <= r < dim
        ==> #[trigger] cm_entry(sel_vects, dim, r, t) == cm_entry(q, dim, r, idx[t])
}

/// The contract of a spectral part of A: the eigensolver contract for (B, Q) and the
/// selection of the pairs (diag(B)[k], column k of Q) with sel(k), k < dim.
pub open spec fn spectral_part_ok(a: Seq<real>, b: Seq<real>, q: Seq<real>, sel: spec_fn(int) -> bool, sel_vals: Seq<real>, sel_vects: Seq<real>, dim: int) -> bool {
    &&& eigensolver_ok(a, b, q, sym_diag_r(b, dim), dim)
    &&& selected_pairs(q, sym_diag_r(b, dim), sel, dim, sel_vals, sel_vects, dim)
}

/// The number of selected pairs, m = sel_indices(sel, dim).len().
pub open spec fn sel_count(sel: spec_fn(int) -> bool, dim: int) -> int {
    sel_indices(sel, dim as nat).len() as int
}

/// Q_sel D_sel Q_sel^T (inner dimension m) is Q diag(masked vals) Q^T (inner dimension dim):
/// the reconstruction from the selected pairs is the spectral reconstruction with the
/// unselected values replaced by 0.
pub proof fn lemma_selected_reconstruction(q: Seq<real>, vals: Seq<real>, sel: spec_fn(int) -> bool, sel_vals: Seq<real>, sel_vects: Seq<real>, dim: int)
    requires
        dim >= 0,
        vals.len() == dim,
        selected_pairs(q, vals, sel, dim, sel_vals, sel_vects, dim),
    ensures
        meq(mmul(mmul(cm_view(sel_vects, dim), diag_view(sel_vals), sel_count(sel, dim)), mtr(cm_view(sel_vects, dim)), sel_count(sel, dim)),
            mmul(mmul(cm_view(q, dim), diag_view(masked_r(vals, sel)), dim), mtr(cm_view(q, dim)), dim), dim, dim),
{
    let idx = sel_indices(sel, dim as nat);
    let m = sel_count(sel, dim);
    let qs = cm_view(sel_vects, dim);
    let qv = cm_view(q, dim);
    let md = masked_r(vals, sel);
    let lhs = mmul(mmul(qs, diag_view(sel_vals), m), mtr(qs), m);
    let rhs = mmul(mmul(qv, diag_view(md), dim), mtr(qv), dim);
    lemma_sel_indices(sel, dim as nat);
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim implies #[trigger] lhs(r, c) == rhs(r, c) by {
        lemma_mul_diag_transpose(qs, sel_vals, m, r, c);
        lemma_mul_diag_transpose(qv, md, dim, r, c);
        let f = |k: int| vals[k] * qv(r, k) * qv(c, k);
        let g = |k: int| sel_vals[k] * qs(r, k) * qs(c, k);
        let h = |k: int| md[k] * qv(r, k) * qv(c, k);
        let gs = sel_terms(f, idx);
        let hm = masked_terms(f, sel);
        assert forall|t: int| 0 <= t < m implies #[trigger] g(t) == gs(t) by {
            assert(sel_vals[t] == vals[idx[t]]);
            assert(qs(r, t) == cm_entry(q, dim, r, idx[t]));
            assert(qs(c, t) == cm_entry(q, dim, c, idx[t]));
        }
        lemma_sum_fn_ext(g, gs, m as nat);
        assert forall|k: int| 0 <= k < dim implies #[trigger] h(k) == hm(k) by {
            if sel(k) {
                assert(md[k] == vals[k]);
            } else {
                assert(md[k] == 0.0real);
                let (u, w) = (qv(r, k), qv(c, k));
                assert(0.0real * u * w == 0.0real) by (nonlinear_arith);
            }
        }
        lemma_sum_fn_ext(h, hm, dim as nat);
        lemma_sum_selected(f, sel, dim as nat);
    }
}

/// The reconstruction of the positive part, Q diag(masked vals) Q^T with the values
/// vals[k] <= 0 masked, is positive semidefinite (for any Q).
pub proof fn lemma_pos_part_psd(q: Seq<real>, vals: Seq<real>, dim: int)
    requires
        dim >= 0,
        vals.len() == dim,
    ensures
        is_psd(mmul(mmul(cm_view(q, dim), diag_view(masked_r(vals, pos_sel(vals))), dim), mtr(cm_view(q, dim)), dim), dim),
{
    let md = masked_r(vals, pos_sel(vals));
    assert forall|k: int| 0 <= k < dim implies #[trigger] md[k] >= 0.0real by {
        if pos_sel(vals)(k) {
            assert(md[k] == vals[k]);
        } else {
            assert(md[k] == 0.0real);
        }
    }
    lemma_diag_congruence_psd(cm_view(q, dim), md, dim);
}

/// An empty negative part certifies that every value is >= -1e-8 * scale, so that
/// Q diag(vals) Q^T >= -1e-8 * scale I for orthogonal Q.
pub proof fn lemma_no_negative_part(q: Seq<real>, vals: Seq<real>, scale: real, dim: int)
    requires
        dim >= 0,
        vals.len() == dim,
        is_orthogonal(cm_view(q, dim), dim),
        sel_indices(neg_sel(vals, scale), dim as nat).len() == 0,
    ensures
        forall|k: int| 0 <= k < dim ==> #[trigger] vals[k] >= -neg_tol_r(scale),
        psd_shifted(mmul(mmul(cm_view(q, dim), diag_view(vals), dim), mtr(cm_view(q, dim)), dim), dim, neg_tol_r(scale)),
{
    lemma_sel_indices_empty(neg_sel(vals, scale), dim as nat);
    assert forall|k: int| 0 <= k < dim implies #[trigger] vals[k] >= -neg_tol_r(scale) by {
        assert(!neg_sel(vals, scale)(k));
    }
    lemma_diag_congruence_psd_shifted(cm_view(q, dim), vals, dim, neg_tol_r(scale));
}

/// A column-major matrix of m > 0 columns with dim rows has the length diag_scale_gram
/// expects, and that function reads dim rows from it.
pub proof fn lemma_rows_of_product(m: int, dim: int)
    requires
        m > 0,
        dim >= 0,
    ensures
        (m * dim) % m == 0,
        rows_of(m * dim, m) == dim,
{
    lemma_mod_multiples_basic(dim, m);
    lemma_div_multiples_vanish(dim, m);
    assert(m * dim == dim * m) by (nonlinear_arith);
}

} // verus!
