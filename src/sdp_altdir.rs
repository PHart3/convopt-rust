pub use crate::lin_alg::jacobi::*;
use crate::sdp::*;
use crate::lin_alg::comparison::{relative_residual_is_small, residual_is_small};
use crate::trusted::{abort, pos_infinity};
use vstd::prelude::*;
use vstd::slice::slice_to_vec;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_mod_multiples_basic, lemma_div_multiples_vanish, lemma_div_multiples_vanish_fancy};
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
use crate::lin_alg::solve_spec::abs_max_r;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::spectral_spec::*;
#[cfg(verus_keep_ghost)]
use crate::sdp_spec::*;
use crate::sdpad_spec::Penalty;
#[cfg(verus_keep_ghost)]
use crate::sdpad_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

// maximum number of iterations to run
pub const TOTAL_STEPS: usize = 5000;

// tuning parameters: the penalty interval and scaling factor, the step of the primal update,
// the stagnation thresholds, and the consecutive-iteration count and ratio thresholds of the
// penalty update
pub const PEN_INIT: f64 = 5.0;
pub const PEN_MIN: f64 = 1e-4;
pub const PEN_MAX: f64 = 1e4;
pub const PEN_FACT: f64 = 0.5;
pub const STEP: f64 = 1.6;
pub const STAG1: i32 = 20;
pub const STAG2: i32 = 150;
pub const STAG3: i32 = 300;
pub const PEN_COUNT: i32 = 50;
pub const PEN_RATIO1: f64 = 1.0;
pub const PEN_RATIO2: f64 = 1.0;

/// The standard-form problem prepared for the iteration (see sdpad_spec.rs).
pub struct SdpadSetup {
    pub obj_frob: SymMatrix,
    pub mat_dense: Matrix,
    pub zeros: Vec<(usize, usize)>,
    pub point: Vector,
    pub var_dim: usize,
    pub symm_dims: Vec<usize>,
    pub block_dims: Vec<usize>,
    pub c_len: usize,
    pub block_size_sum: usize,
    pub gram_small: SymMatrix,
    pub psd_blocks: Vec<(usize, usize)>,
}

/// The contract of the setup: the data (with the objective obj0 before reweighting) is the
/// standard form of sdp, obj_frob is obj0 with halved off-diagonal entries, c_len is the number
/// of dense constraint rows (followed by the zero constraints in point, whose right-hand sides
/// are zero; there are dense rows whenever there are zero constraints), gram_small is the packed
/// Gram matrix of the dense rows, block_size_sum is the total LMI block size (zero exactly
/// without LMIs), and psd_blocks lists the diagonal-block partition of the variable.
pub open spec fn sdpad_setup_ok(sdp: &SDP, check: bool, s: SdpadSetup) -> bool {
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let nz = s.zeros@.len() as int;
    let rows = s.c_len as int;
    let bd = dims_of(s.block_dims@);
    let vd = dims_of(s.symm_dims@);
    let bs = psd_partition(bd, vd, d);
    &&& exists|obj0: SymMatrix| #[trigger] std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check)
            && half_off_diag(rvs(s.obj_frob@), rvs(obj0@), d)
    &&& d >= 1
    &&& dim_fits(d)
    &&& dim_fits(rows)
    &&& s.obj_frob@.len() == cols
    &&& s.point@.len() == rows + nz
    &&& s.mat_dense@.len() == rows * cols
    &&& valid_zero_list(s.zeros@, d)
    &&& forall|q: int| rows <= q < s.point@.len() ==> rv(#[trigger] s.point@[q]) == 0.0real
    &&& rows == 0 ==> nz == 0
    &&& s.block_size_sum as int == block_sum(bd)
    &&& (s.block_size_sum == 0) == (sdp.lmi@.len() == 0)
    &&& rows > 0 ==> is_constraint_gram(rvs(s.gram_small@), rvs(s.mat_dense@), rows, d)
    &&& rows == 0 ==> s.gram_small@.len() == 0
    &&& block_list_ok(s.psd_blocks@, bs)
    &&& sum_upto(bs, bs.len() as int) == d
    &&& blocks_fit(s.psd_blocks@, pk_len(d) as nat)
}

// the (start, size) list of the diagonal blocks of the standard-form variable (psd_partition):
// the LMI blocks, then the two blocks of each user variable; without LMIs, the whole variable
fn psd_block_list(block_dims: &Vec<usize>, symm_dims: &Vec<usize>, block_size_sum: usize, var_dim: usize) -> (list: Vec<(usize, usize)>)
    requires
        block_size_sum as int == block_sum(dims_of(block_dims@)),
        block_size_sum > 0 ==> std_dim(dims_of(block_dims@), dims_of(symm_dims@)) == var_dim,
    ensures
        block_list_ok(list@, psd_partition(dims_of(block_dims@), dims_of(symm_dims@), var_dim as int)),
{
    let ghost bd = dims_of(block_dims@);
    let ghost vd = dims_of(symm_dims@);
    let ghost bs = block_sizes(bd, vd);
    let nb = block_dims.len();
    let nv = symm_dims.len();
    let mut list: Vec<(usize, usize)> = Vec::new();
    if block_size_sum == 0 {
        list.push((0, var_dim));
        return list;
    }
    proof {
        lemma_block_sizes_sum(bd, vd, nv as int);
        assert forall|j: int| 0 <= j < bs.len() implies #[trigger] bs[j] >= 0 by {
            lemma_block_sizes_entry(bd, vd, j);
        }
    }
    let mut start: usize = 0;
    for i in 0..nb
        invariant
            nb == block_dims@.len(),
            nv == symm_dims@.len(),
            bd == dims_of(block_dims@),
            vd == dims_of(symm_dims@),
            bs == block_sizes(bd, vd),
            bs.len() == nb + 2 * nv,
            forall|j: int| 0 <= j < bs.len() ==> #[trigger] bs[j] >= 0,
            sum_upto(bs, bs.len() as int) == var_dim,
            list@.len() == i,
            start as int == sum_upto(bs, i as int),
            forall|j: int| 0 <= j < i ==> (#[trigger] list@[j]).0 as int == sum_upto(bs, j) && list@[j].1 as int == bs[j],
    {
        let dim = block_dims[i];
        proof {
            lemma_block_sizes_entry(bd, vd, i as int);
            lemma_sum_upto_monotone(bs, i as int + 1, bs.len() as int);
        }
        list.push((start, dim));
        start = start + dim;
    }
    for n in 0..nv
        invariant
            nb == block_dims@.len(),
            nv == symm_dims@.len(),
            bd == dims_of(block_dims@),
            vd == dims_of(symm_dims@),
            bs == block_sizes(bd, vd),
            bs.len() == nb + 2 * nv,
            forall|j: int| 0 <= j < bs.len() ==> #[trigger] bs[j] >= 0,
            sum_upto(bs, bs.len() as int) == var_dim,
            list@.len() == nb + 2 * n,
            start as int == sum_upto(bs, nb + 2 * n),
            forall|j: int| 0 <= j < nb + 2 * n ==> (#[trigger] list@[j]).0 as int == sum_upto(bs, j) && list@[j].1 as int == bs[j],
    {
        let d = symm_dims[n];
        proof {
            let j = nb + 2 * n;
            lemma_block_sizes_entry(bd, vd, j);
            lemma_block_sizes_entry(bd, vd, j + 1);
            lemma_div_multiples_vanish(n as int, 2);
            lemma_div_multiples_vanish_fancy(n as int, 1, 2);
            assert(bs[j] == d && bs[j + 1] == d);
            assert(sum_upto(bs, j + 1) == sum_upto(bs, j) + bs[j]);
            assert(sum_upto(bs, j + 2) == sum_upto(bs, j + 1) + bs[j + 1]);
            lemma_sum_upto_monotone(bs, j + 2, bs.len() as int);
            assert(start + 2 * d <= var_dim);
        }
        list.push((start, d));
        list.push((start + d, d));
        start = start + 2 * d;
    }
    list
}

// the equality-system solve of one iteration (dual_solve_ok): with reduced constraints the
// block system (G, I/2) y = rhs is solved directly; otherwise G is regularized by
// TOL * max(max_i |G_ii|, 1) on its diagonal, and the solution must have a small residual
// relative to the original system (aborts otherwise)
fn dual_solve(gram: &SymMatrix, rhs: &mut Vector, c_len: usize, check: bool) -> (y: Vector)
    requires
        c_len <= old(rhs)@.len(),
        dim_fits(c_len as int),
        gram@.len() >= pk_len(c_len as int),
    ensures
        dual_solve_ok(rvs(gram@), rvs(old(rhs)@), c_len as int, check, rvs(y@)),
{
    let ghost g = rvs(gram@);
    let ghost b = rvs(old(rhs)@);
    let ghost cl = c_len as int;
    let n = rhs.len();
    let nz = n - c_len;
    if n == 0 {
        let y: Vector = vec![];
        proof {
            assert(mat_vec(madd(sym_view(g), mscale(reg_shift_r(g, cl, check), mident())), cl, rvs(y@).subrange(0, cl))
                =~= b.subrange(0, cl));
        }
        return y;
    }
    if check || c_len == 0 {
        let y = pos_def_solver_upperleft(gram, rhs, nz);
        proof {
            lemma_block_half_solve_inv(sym_view(g), cl, n as int, rvs(y@), b);
            let yt = rvs(y@).subrange(0, cl);
            if check {
                lemma_shift_zero_meq(sym_view(g), cl);
                lemma_mat_vec_meq(madd(sym_view(g), mscale(0.0real, mident())), sym_view(g), cl, yt);
            } else {
                assert(mat_vec(madd(sym_view(g), mscale(reg_shift_r(g, cl, check), mident())), cl, yt) =~= b.subrange(0, cl));
            }
        }
        return y;
    }
    // the right-hand side of the dense rows, kept for the residual (the solver overwrites rhs)
    let rhs_top = slice_to_vec(&rhs[..c_len]);
    let ghost bt = b.subrange(0, cl);
    proof {
        assert(rvs(rhs_top@) =~= bt);
    }
    // regularize the Gram matrix so that it can be fed to the solver
    let reg = max_diag_sym(gram, c_len);
    let shift = TOL * reg.max(1.0);
    let ghost sh = reg_shift_r(g, cl, check);
    let mut gram_reg = gram.clone();
    proof {
        lemma_dim_fits_idx(cl, cl);
    }
    for i in 0..c_len
        invariant
            cl == c_len,
            dim_fits(cl),
            gram_reg@.len() == gram@.len(),
            gram@.len() >= pk_len(cl),
            g == rvs(gram@),
            rv(shift) == sh,
            forall|r: int, k: int| 0 <= r <= k < cl
                ==> #[trigger] rvs(gram_reg@)[pk_idx(r, k)] == g[pk_idx(r, k)] + (if r == k && r < i { sh } else { 0.0real }),
    {
        proof {
            lemma_dim_fits_idx(cl, i as int);
            lemma_pk_idx_bound(i as int, i as int, cl);
            assert((i * i + 3 * i) / 2 == pk_idx(i as int, i as int)) by (nonlinear_arith);
        }
        let k = (i * i + 3 * i) / 2;
        let ghost ki = pk_idx(i as int, i as int);
        let ghost prev = rvs(gram_reg@);
        let old_entry = gram_reg[k];
        gram_reg[k] = old_entry + shift;
        proof {
            assert(k == ki);
            assert(rv(old_entry) == prev[ki]);
            assert(prev[ki] == g[ki] + 0.0real);
            assert(rvs(gram_reg@)[ki] == rv(old_entry) + sh);
            assert forall|r: int, c: int| 0 <= r <= c < cl
                implies #[trigger] rvs(gram_reg@)[pk_idx(r, c)] == g[pk_idx(r, c)] + (if r == c && r < i + 1 { sh } else { 0.0real }) by {
                lemma_pk_idx_bound(r, c, cl);
                lemma_pk_idx_injective(r, c, i as int, i as int);
                if pk_idx(r, c) == ki {
                    assert(r == i && c == i);
                } else {
                    assert(rvs(gram_reg@)[pk_idx(r, c)] == rv(gram_reg@[pk_idx(r, c)]));
                    assert(rvs(gram_reg@)[pk_idx(r, c)] == prev[pk_idx(r, c)]);
                }
            }
        }
    }
    let y = pos_def_solver_upperleft(&gram_reg, rhs, nz);
    let ghost gr = rvs(gram_reg@);
    proof {
        lemma_block_half_solve_inv(sym_view(gr), cl, n as int, rvs(y@), b);
        assert(diag_shifted(gr, g, cl, sh));
        lemma_diag_shifted_view(gr, g, cl, sh);
        lemma_mat_vec_meq(sym_view(gr), madd(sym_view(g), mscale(sh, mident())), cl, rvs(y@).subrange(0, cl));
    }
    // the residual of the regularized solution with respect to the original system
    let y_top = &y[..c_len];
    let ghost yt = rvs(y@).subrange(0, cl);
    proof {
        assert(rvs(y_top@) =~= yt);
    }
    let resid_top = euclid_distance(&sym_mat_act(gram, y_top), &rhs_top);
    let resid_bottom = frob_norm_sym(gram, c_len) * euclid_norm(y_top) + euclid_norm(&rhs_top);
    if !relative_residual_is_small(resid_top, resid_bottom, 1e-9) {
        abort("regularized Gram solve is not an accurate solution to original system");
    }
    y
}

// the initial iterates Z = I, S = 0
fn initial_iterates(var_dim: usize) -> (res: (SymMatrix, SymMatrix))
    requires
        dim_fits(var_dim as int),
    ensures
        res.0@.len() == pk_len(var_dim as int),
        res.1@.len() == pk_len(var_dim as int),
        initial_iterates_ok(rvs(res.0@), rvs(res.1@), var_dim as int),
{
    proof {
        lemma_dim_fits_idx(var_dim as int, var_dim as int);
    }
    let var_dim_tot = (var_dim * (var_dim + 1)) / 2;
    let prim_z = ident_sym_mat(var_dim);
    let dual_s: SymMatrix = vec![0.0; var_dim_tot];
    proof {
        assert forall|t: int| 0 <= t < dual_s@.len() implies #[trigger] rvs(dual_s@)[t] == 0.0real by {
            assert(rv(dual_s@[t]) == 0.0real);
        }
    }
    (prim_z, dual_s)
}

// conversion to standard form, reweighting of the objective for the Frobenius product, Gram
// matrix of the dense constraint rows, and the PSD block partition (contract: sdpad_setup_ok)
pub fn sdpad_setup(sdp: &SDP, check_constraints: bool) -> (s: SdpadSetup)
    requires
        sdpad_input_ok(sdp, check_constraints),
    ensures
        sdpad_setup_ok(sdp, check_constraints, s),
{
    let (mut obj, (mat_dense, zeros), point, var_dim, symm_dims, block_dims) = sdp_to_standard(sdp, check_constraints);
    let ghost obj0: SymMatrix = obj;
    let ghost res0 = (obj0, (mat_dense, zeros), point, var_dim, symm_dims, block_dims);
    let ghost d = var_dim as int;
    let ghost cols = pk_len(d);
    let ghost nz = zeros@.len() as int;
    let ghost bd = dims_of(block_dims@);
    let ghost vd = dims_of(symm_dims@);
    proof {
        assert(std_form_ok(sdp, res0, check_constraints));
        lemma_std_form_facts(sdp, res0, check_constraints);
        lemma_pk_col_start_zero();
        lemma_pk_col_start_step(0);
        lemma_pk_col_start_monotone(1, d);
    }
    let c_len = point.len() - zeros.len();
    let ghost rows = c_len as int;
    // the total size of the LMI blocks
    let nb = block_dims.len();
    let mut block_size_sum: usize = 0;
    proof {
        assert forall|i: int| 0 <= i < bd.len() implies #[trigger] bd[i] >= 0 by {}
        lemma_sum_upto_monotone(bd, 0, nb as int);
    }
    for i in 0..nb
        invariant
            nb == block_dims@.len(),
            bd == dims_of(block_dims@),
            forall|i: int| 0 <= i < bd.len() ==> #[trigger] bd[i] >= 0,
            block_sum(bd) <= d,
            d == var_dim,
            block_size_sum as int == sum_upto(bd, i as int),
    {
        proof {
            lemma_sum_upto_monotone(bd, i as int + 1, nb as int);
        }
        block_size_sum = block_size_sum + block_dims[i];
    }
    // adjust objective function so that it acts via the Frobenius product instead of packed dot product
    let _obj_ref = scale_off_diag(0.5, var_dim, &mut obj);
    let ghost cf = rvs(_obj_ref@);
    // form the Gram matrix for the linear constraints (excluding the zero constraints)
    proof {
        if rows > 0 {
            lemma_mod_multiples_basic(rows, cols);
            lemma_div_multiples_vanish(rows, cols);
            assert(rows * cols == cols * rows) by (nonlinear_arith);
        } else {
            assert(0 * cols == 0);
        }
    }
    let gram_small = constraint_gram(var_dim, &mat_dense);
    let psd_blocks = psd_block_list(&block_dims, &symm_dims, block_size_sum, var_dim);
    let s = SdpadSetup { obj_frob: obj, mat_dense, zeros, point, var_dim, symm_dims, block_dims, c_len, block_size_sum, gram_small, psd_blocks };
    proof {
        let bs = psd_partition(bd, vd, d);
        assert(rvs(s.obj_frob@) == cf);
        assert forall|r: int, k: int| 0 <= r <= k < d
            implies #[trigger] cf[pk_idx(r, k)] == (if r == k { rvs(obj0@)[pk_idx(r, k)] } else { 0.5real * rvs(obj0@)[pk_idx(r, k)] }) by {
            lemma_pk_idx_bound(r, k, d);
            assert(rv(_obj_ref@[pk_idx(r, k)]) == (if r == k { rv(obj0@[pk_idx(r, k)]) } else { rv(0.5) * rv(obj0@[pk_idx(r, k)]) }));
        }
        assert(half_off_diag(cf, rvs(obj0@), d));
        assert(std_form_ok(sdp, (obj0, (s.mat_dense, s.zeros), s.point, s.var_dim, s.symm_dims, s.block_dims), check_constraints));
        if block_sum(bd) == 0 {
            assert(bs =~= seq![d]);
            assert(sum_upto(bs, 0) == 0);
            assert(sum_upto(bs, 1) == sum_upto(bs, 0) + bs[0]);
        } else {
            assert(nb >= 1);
            assert(bs == block_sizes(bd, vd));
            assert(bs.len() == bd.len() + 2 * vd.len());
            lemma_block_sizes_sum(bd, vd, vd.len() as int);
            assert(sum_upto(bs, bs.len() as int) == std_dim(bd, vd));
        }
        assert(sum_upto(bs, bs.len() as int) == d);
        assert forall|j: int| 0 <= j < bs.len() implies #[trigger] bs[j] >= 0 by {
            if block_sum(bd) != 0 {
                lemma_block_sizes_entry(bd, vd, j);
            }
        }
        lemma_block_list_fits(s.psd_blocks@, bs, d);
    }
    s
}

/* ---------- the Wen-Goldfarb-Yin update ---------- */

/// The iterates of the Wen-Goldfarb-Yin method: Z (prim_z), S (dual_s), y (dual_y), the dual
/// slack C - A*(y) (diff) and V = C - A*(y) - mu Z (dual_sfull), together with the buffers
/// every update reuses (the right-hand side of the dual system, overwritten by the solve, and
/// the eigensolver's working copy of V).
pub struct WgyState {
    pub prim_z: SymMatrix,
    pub dual_s: SymMatrix,
    pub dual_y: Vector,
    pub diff: SymMatrix,
    pub dual_sfull: SymMatrix,
    pub inverse_y: Vector,
    pub dual_sfull_temp: SymMatrix,
}

/// The constraint operator A of the setup applied to the packed matrix x: the dense rows,
/// then the zero-pattern entries (constraint_action_r, L04).
pub open spec fn setup_action_r(s: &SdpadSetup, x: Seq<real>) -> Seq<real> {
    constraint_action_r(rvs(s.mat_dense@), s.c_len as int, s.zeros@, x)
}

/// The shape facts of the setup the update relies on (those of sdpad_setup_ok), and zero
/// constraints only together with dense rows (without dense rows constraint_action ignores
/// the zero pattern).
pub open spec fn wgy_setup_ok(s: &SdpadSetup) -> bool {
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let rows = s.c_len as int;
    let nz = s.zeros@.len() as int;
    &&& d >= 1
    &&& dim_fits(d)
    &&& dim_fits(rows)
    &&& s.obj_frob@.len() == cols
    &&& s.point@.len() == rows + nz
    &&& s.mat_dense@.len() == rows * cols
    &&& valid_zero_list(s.zeros@, d)
    &&& rows > 0 ==> is_constraint_gram(rvs(s.gram_small@), rvs(s.mat_dense@), rows, d)
    &&& rows == 0 ==> s.gram_small@.len() == 0
    &&& rows == 0 ==> nz == 0
}

/// One update from the iterates (Z, S) of st0 to those of st1, for the penalty mu and the
/// step rho (sdpad_spec.rs): y solves the dual system with right-hand side
/// mu (b - A(Z)) + A(C - S); diff = C - A*(y); V = diff - mu Z; S' = V_+; and
/// Z' = (1 - rho) Z + (rho / mu) (S' - V).
pub open spec fn wgy_update_ok(s: &SdpadSetup, check: bool, mu: real, rho: real, st0: &WgyState, st1: &WgyState) -> bool {
    let d = s.var_dim as int;
    let cf = rvs(s.obj_frob@);
    let z0 = rvs(st0.prim_z@);
    let y = rvs(st1.dual_y@);
    let v = rvs(st1.dual_sfull@);
    let rhs = wgy_rhs_r(mu, rvs(s.point@), setup_action_r(s, z0), setup_action_r(s, vsub_r(cf, rvs(st0.dual_s@))));
    &&& dual_solve_ok(rvs(s.gram_small@), rhs, s.c_len as int, check, y)
    &&& is_dual_slack(rvs(st1.diff@), cf, rvs(s.mat_dense@), s.c_len as int, s.zeros@, y, d)
    &&& is_shifted_slack(v, rvs(st1.diff@), mu, z0, d)
    &&& is_pos_part(v, rvs(st1.dual_s@), d)
    &&& is_primal_step(rvs(st1.prim_z@), rho, mu, z0, rvs(st1.dual_s@), v, d)
}

/// The packed column count of the setup is positive and the dense rows are read back from the
/// length of mat_dense (the row count constraint_action and constraint_lin_comb compute).
pub proof fn lemma_setup_rows(s: &SdpadSetup)
    requires
        wgy_setup_ok(s),
    ensures
        pk_len(s.var_dim as int) >= 1,
        s.mat_dense@.len() as int % pk_len(s.var_dim as int) == 0,
        rows_of(s.mat_dense@.len() as int, pk_len(s.var_dim as int)) == s.c_len as int,
        s.mat_dense@.len() == 0 ==> s.c_len == 0,
{
    let d = s.var_dim as int;
    let cols = pk_len(d);
    let rows = s.c_len as int;
    lemma_pk_col_start_zero();
    lemma_pk_col_start_step(0);
    lemma_pk_col_start_monotone(1, d);
    lemma_mod_multiples_basic(rows, cols);
    lemma_div_multiples_vanish(rows, cols);
    assert(rows * cols == cols * rows);
    assert(s.mat_dense@.len() == 0 ==> rows == 0);
}

// the constraint operator of the setup applied to x (contract: setup_action_r), with one entry
// per constraint row
fn setup_action(s: &SdpadSetup, x: &SymMatrix) -> (r: Vector)
    requires
        wgy_setup_ok(s),
        x@.len() == pk_len(s.var_dim as int),
    ensures
        r@.len() == s.c_len + s.zeros@.len(),
        rvs(r@) == setup_action_r(s, rvs(x@)),
{
    let ghost d = s.var_dim as int;
    proof {
        lemma_setup_rows(s);
        assert(dim_fits(d) && pk_len(d) == x@.len() && valid_zero_list(s.zeros@, d));
        assert(zero_list_fits(s.zeros@, x@.len() as int));
    }
    let constr = (&s.mat_dense, &s.zeros);
    let r = constraint_action(&constr, x);
    proof {
        if s.mat_dense@.len() == 0 {
            assert(rvs(r@) =~= setup_action_r(s, rvs(x@)));
        }
    }
    r
}

// one Wen-Goldfarb-Yin update of the iterates for the penalty mu and the step rho
// (contract: wgy_update_ok): the dual solve for y, V = C - A*(y) - mu Z, S = V_+ through
// poseigendecomp, and Z <- (1 - rho) Z + (rho / mu) (S - V)
pub fn wgy_update(s: &SdpadSetup, check: bool, var: &JacobiVariant, penalty: f64, step: f64, st: &mut WgyState)
    requires
        wgy_setup_ok(s),
        old(st).prim_z@.len() == pk_len(s.var_dim as int),
        old(st).dual_s@.len() == pk_len(s.var_dim as int),
        rv(penalty) != 0.0real,
    ensures
        wgy_update_ok(s, check, rv(penalty), rv(step), old(st), final(st)),
{
    let ghost d = s.var_dim as int;
    let ghost cols = pk_len(d);
    let ghost rows = s.c_len as int;
    let ghost mu = rv(penalty);
    let ghost rho = rv(step);
    let ghost z0 = rvs(old(st).prim_z@);
    let ghost s0 = rvs(old(st).dual_s@);
    let ghost cf = rvs(s.obj_frob@);
    let var_dim = s.var_dim;
    proof {
        lemma_dim_fits_idx(d, d);
        lemma_pk_col_start_zero();
    }
    let var_dim_tot = (var_dim * (var_dim + 1)) / 2;
    // the dual system (A A*) y = A(C - S) + mu (b - A(Z))
    let action = setup_action(s, &st.prim_z);
    let c_minus_s = vect_subt(&s.obj_frob, &st.dual_s);
    let acs = setup_action(s, &c_minus_s);
    let ghost rhs = wgy_rhs_r(mu, rvs(s.point@), setup_action_r(s, z0), setup_action_r(s, vsub_r(cf, s0)));
    let n = acs.len();
    st.inverse_y.clear();
    for i in 0..n
        invariant
            n == acs@.len() == action@.len() == s.point@.len(),
            st.prim_z@ == old(st).prim_z@,
            st.dual_s@ == old(st).dual_s@,
            st.inverse_y@.len() == i,
            forall|k: int| 0 <= k < i
                ==> rv(#[trigger] st.inverse_y@[k]) == rv(penalty) * (rv(s.point@[k]) - rv(action@[k])) + rv(acs@[k]),
    {
        st.inverse_y.push(penalty * (s.point[i] - action[i]) + acs[i]);
    }
    proof {
        assert(rvs(acs@) == setup_action_r(s, vsub_r(cf, s0)));
        assert(rvs(st.inverse_y@) =~= rhs);
    }
    st.dual_y = dual_solve(&s.gram_small, &mut st.inverse_y, s.c_len, check);
    let ghost y = rvs(st.dual_y@);
    // the dual slack C - A*(y) and V = C - A*(y) - mu Z
    proof {
        lemma_setup_rows(s);
    }
    let constr = (&s.mat_dense, &s.zeros);
    let lin_comb = constraint_lin_comb(var_dim, &constr, &st.dual_y);
    st.diff = vect_subt(&s.obj_frob, &lin_comb);
    proof {
        let m = rvs(s.mat_dense@);
        let lc = rvs(lin_comb@);
        assert forall|r: int, c: int| 0 <= r <= c < d
            implies #[trigger] rvs(st.diff@)[pk_idx(r, c)] == cf[pk_idx(r, c)] - adjoint_entry_r(m, rows, s.zeros@, y, r, c) by {
            lemma_pk_idx_bound(r, c, d);
            if s.mat_dense@.len() == 0 {
                lemma_adjoint_entry_empty(m, s.zeros@, y, r, c);
            }
        }
        assert(is_dual_slack(rvs(st.diff@), cf, m, rows, s.zeros@, y, d));
    }
    st.dual_sfull.clear();
    st.dual_sfull_temp.clear();
    for i in 0..var_dim_tot
        invariant
            var_dim_tot == cols,
            st.prim_z@ == old(st).prim_z@,
            st.prim_z@.len() == cols,
            st.diff@.len() == cols,
            st.dual_sfull@.len() == i,
            st.dual_sfull_temp@ == st.dual_sfull@,
            forall|t: int| 0 <= t < i ==> rv(#[trigger] st.dual_sfull@[t]) == rv(st.diff@[t]) - rv(penalty) * rv(st.prim_z@[t]),
    {
        let val = st.diff[i] - penalty * st.prim_z[i];
        st.dual_sfull.push(val);
        st.dual_sfull_temp.push(val);
    }
    let ghost v = rvs(st.dual_sfull@);
    let ghost diff = rvs(st.diff@);
    proof {
        assert forall|t: int| 0 <= t < cols implies #[trigger] v[t] == diff[t] - mu * z0[t] by {
            assert(rv(st.dual_sfull@[t]) == rv(st.diff@[t]) - rv(penalty) * rv(st.prim_z@[t]));
        }
        assert(is_shifted_slack(v, diff, mu, z0, d));
        assert(rvs(st.dual_sfull_temp@) == v);
    }
    // S = V_+, the positive spectral part of V
    let (egnvals, egnvects) = poseigendecomp(&mut st.dual_sfull_temp, var_dim, var);
    let ghost b = rvs(st.dual_sfull_temp@);
    let ghost vals = sym_diag_r(b, d);
    let ghost q = choose|q: Seq<real>| #[trigger] spectral_part_ok(v, b, q, pos_sel(vals), rvs(egnvals@), rvs(egnvects@), d);
    if egnvals.is_empty() {
        st.dual_s = vec![0.0; var_dim_tot];
        proof {
            assert forall|t: int| 0 <= t < st.dual_s@.len() implies #[trigger] rvs(st.dual_s@)[t] == 0.0real by {
                assert(rv(st.dual_s@[t]) == 0.0real);
            }
            lemma_zero_pos_part(v, b, q, rvs(st.dual_s@), d);
        }
    } else {
        let ghost m = egnvals@.len() as int;
        proof {
            lemma_rows_of_product(m, d);
        }
        st.dual_s = diag_scale_gram(&egnvals, &egnvects);
        proof {
            lemma_selected_pos_part(v, b, q, rvs(egnvals@), rvs(egnvects@), rvs(st.dual_s@), d);
        }
    }
    let ghost s1 = rvs(st.dual_s@);
    proof {
        assert(pos_part_of(v, b, q, s1, d));
        assert(is_pos_part(v, s1, d));
    }
    // Z <- (1 - rho) Z + (rho / mu) (S - V)
    for i in 0..var_dim_tot
        invariant
            var_dim_tot == cols,
            rv(penalty) != 0.0real,
            st.prim_z@.len() == cols,
            st.dual_s@.len() == cols,
            st.dual_sfull@.len() == cols,
            z0 == rvs(old(st).prim_z@),
            s1 == rvs(st.dual_s@),
            v == rvs(st.dual_sfull@),
            forall|t: int| 0 <= t < i
                ==> rv(#[trigger] st.prim_z@[t]) == (1.0real - rv(step)) * z0[t] + (rv(step) / rv(penalty)) * (s1[t] - v[t]),
            forall|t: int| i <= t < cols ==> rv(#[trigger] st.prim_z@[t]) == z0[t],
    {
        st.prim_z[i] = (1.0 - step) * st.prim_z[i] + (step / penalty) * (st.dual_s[i] - st.dual_sfull[i]);
    }
    proof {
        assert forall|t: int| 0 <= t < cols
            implies #[trigger] rvs(st.prim_z@)[t] == (1.0real - rho) * z0[t] + (rho / mu) * (s1[t] - v[t]) by {
            assert(rv(st.prim_z@[t]) == (1.0real - rv(step)) * z0[t] + (rv(step) / rv(penalty)) * (s1[t] - v[t]));
        }
        assert(is_primal_step(rvs(st.prim_z@), rho, mu, z0, s1, v, d));
    }
}

// the penalty update of one iteration from the primal and dual infeasibilities (contract:
// penalty_update_ok): the counts of consecutive primal- and dual-dominated iterations stay
// below PEN_COUNT, and the penalty stays within [PEN_MIN, PEN_MAX]
pub fn penalty_update(penalty: f64, pinf: f64, dinf: f64, it_pinf: i32, it_dinf: i32) -> (res: (f64, i32, i32))
    requires
        0 <= it_pinf < PEN_COUNT,
        0 <= it_dinf < PEN_COUNT,
    ensures
        penalty_update_ok(rv(penalty), rv(pinf), rv(dinf), it_pinf as int, it_dinf as int, rv(res.0), res.1 as int, res.2 as int),
        0 <= res.1 < PEN_COUNT,
        0 <= res.2 < PEN_COUNT,
        rv(PEN_MIN) <= rv(penalty) <= rv(PEN_MAX) ==> rv(PEN_MIN) <= rv(res.0) <= rv(PEN_MAX),
{
    let ghost mu0 = rv(penalty);
    let ghost ip0 = it_pinf as int;
    let ghost id0 = it_dinf as int;
    let i_ratio = pinf / dinf;
    let mut penalty = penalty;
    let mut it_pinf = it_pinf;
    let mut it_dinf = it_dinf;
    if !(i_ratio > PEN_RATIO1) {
        it_pinf += 1;
        it_dinf = 0;
        if it_pinf >= PEN_COUNT {
            penalty = (PEN_FACT * penalty).max(PEN_MIN);
            it_pinf = 0;
        }
    } else if i_ratio > PEN_RATIO2 {
        it_dinf += 1;
        it_pinf = 0;
        if it_dinf >= PEN_COUNT {
            penalty = (penalty / PEN_FACT).min(PEN_MAX);
            it_dinf = 0;
        }
    }
    let res = (penalty, it_pinf, it_dinf);
    proof {
        assert(penalty_step_ok(mu0, rv(i_ratio), ip0, id0, rv(res.0), res.1 as int, res.2 as int));
    }
    res
}

/* ---------- residuals, KKT quantities and termination ---------- */

/// Why the loop stopped.
pub enum Termination {
    Accuracy,
    Stagnation,
    MaxIter,
}

/// The outcome of the loop (outcome_ok, sdpad_spec.rs): the user's solution read from an
/// iterate, that iterate's objective value and residual quantities, the termination reason,
/// and (ghost) the trajectory of the loop (its states and penalties), the index of the
/// returned iterate in it, the deltas of all iterations, which of them were recorded as
/// candidates, and the stagnation bookkeeping.
pub struct SdpadOutcome {
    pub result: Vec<SymMatrix>,
    pub value: f64,
    pub pinf: f64,
    pub dinf: f64,
    pub gap: f64,
    pub termination: Termination,
    pub states: Ghost<Seq<WgyState>>,
    pub pens: Ghost<Seq<Penalty>>,
    pub iterate: Ghost<int>,
    pub deltas: Ghost<Seq<f64>>,
    pub feasible: Ghost<Seq<bool>>,
    pub best_delta: Ghost<f64>,
    pub it_stag: Ghost<i32>,
}

/// The setup contract contains the shape facts of the update.
pub proof fn lemma_setup_wgy(sdp: &SDP, check: bool, s: &SdpadSetup)
    requires
        sdpad_setup_ok(sdp, check, *s),
    ensures
        wgy_setup_ok(s),
{
}

// the residuals and KKT quantities of the iterate st (contract: kkt_ok): the primal
// infeasibility ||A(Z) - b|| / (1 + ||b||), the dual infeasibility
// ||(C - A*(y)) - S||_F / (1 + ||C||_1), the objective values <C, Z>_F and b . y, the relative
// duality gap, and delta = max(pinf, dinf, gap)
fn kkt_quantities(s: &SdpadSetup, st: &WgyState) -> (q: (f64, f64, f64, f64, f64))
    requires
        wgy_setup_ok(s),
        st.prim_z@.len() == pk_len(s.var_dim as int),
        st.dual_s@.len() == pk_len(s.var_dim as int),
        st.diff@.len() == pk_len(s.var_dim as int),
        st.dual_y@.len() == s.point@.len(),
    ensures
        kkt_ok(s, st, rv(q.0), rv(q.1), rv(q.2), dual_val_r(s, rvs(st.dual_y@)), rv(q.3), rv(q.4)),
{
    let var_dim = s.var_dim;
    let ghost d = var_dim as int;
    let ghost b = rvs(s.point@);
    let ghost cf = rvs(s.obj_frob@);
    proof {
        // the denominators are at least 1
        lemma_dot_self_nonneg(b);
        assert(norm_r(b) >= 0.0real);
        lemma_sym_one_norm_nonneg(cf, d);
    }
    let action = setup_action(s, &st.prim_z);
    let pinf = euclid_distance(&action, &s.point) / (1.0 + euclid_norm(&s.point));
    let dinf = frob_dist_sym(&st.diff, &st.dual_s, var_dim) / (1.0 + sym_mat_one_norm(&s.obj_frob, var_dim));
    let prim_val = frob_prod_sym(&s.obj_frob, &st.prim_z, var_dim);
    let dual_val = dot_prod(&s.point, &st.dual_y);
    let gap = (dual_val - prim_val).abs() / (1.0 + dual_val.abs() + prim_val.abs());
    let delta = pinf.max(dinf).max(gap);
    (pinf, dinf, prim_val, gap, delta)
}

// the user's solution read from the standard-form variable z (contract: solution_ok): z itself
// without LMIs, otherwise the difference of the two diagonal blocks of every user variable,
// which follow the LMI blocks
fn read_solution(s: &SdpadSetup, z: &SymMatrix) -> (result: Vec<SymMatrix>)
    requires
        dim_fits(s.var_dim as int),
        z@.len() == pk_len(s.var_dim as int),
        s.block_size_sum as int == block_sum(dims_of(s.block_dims@)),
        s.block_size_sum > 0 ==> std_dim(dims_of(s.block_dims@), dims_of(s.symm_dims@)) == s.var_dim as int,
    ensures
        solution_ok(s, rvs(z@), result@),
{
    let ghost d = s.var_dim as int;
    let ghost vd = dims_of(s.symm_dims@);
    let ghost bsum = s.block_size_sum as int;
    let mut result: Vec<SymMatrix> = Vec::new();
    if s.block_size_sum == 0 {
        result.push(z.clone());
        return result;
    }
    let nv = s.symm_dims.len();
    let mut symm_offset: usize = 0;
    proof {
        assert forall|i: int| 0 <= i < vd.len() implies #[trigger] vd[i] >= 0 by {}
    }
    for n in 0..nv
        invariant
            nv == s.symm_dims@.len(),
            vd == dims_of(s.symm_dims@),
            bsum == s.block_size_sum as int,
            bsum > 0,
            d == s.var_dim as int,
            dim_fits(d),
            z@.len() == pk_len(d),
            forall|i: int| 0 <= i < vd.len() ==> #[trigger] vd[i] >= 0,
            bsum + 2 * sum_upto(vd, nv as int) == d,
            symm_offset as int == 2 * sum_upto(vd, n as int),
            result@.len() == n,
            forall|k: int| 0 <= k < n ==> rvs(#[trigger] result@[k]@) == zvar_r(rvs(z@), var_off(bsum, vd, k), vd[k]),
    {
        let dn = s.symm_dims[n];
        proof {
            assert(sum_upto(vd, n as int + 1) == sum_upto(vd, n as int) + vd[n as int]);
            lemma_sum_upto_monotone(vd, n as int + 1, nv as int);
            lemma_dim_fits_monotone(bsum + symm_offset + dn, d);
            lemma_dim_fits_monotone(bsum + symm_offset + 2 * dn, d);
            lemma_pk_col_start_monotone(bsum + symm_offset + dn, d);
            lemma_pk_col_start_monotone(bsum + symm_offset + 2 * dn, d);
        }
        let start = s.block_size_sum + symm_offset;
        let b1 = sym_matrix_diag_block(start, dn, z);
        let b2 = sym_matrix_diag_block(start + dn, dn, z);
        proof {
            lemma_diag_block_view(b1@, z@, start as int, dn as int);
            lemma_diag_block_view(b2@, z@, start as int + dn as int, dn as int);
        }
        result.push(vect_subt(&b1, &b2));
        symm_offset = symm_offset + 2 * dn;
    }
    result
}

// the solver loop (contract: outcome_ok): Wen-Goldfarb-Yin updates with the penalty
// adaptation until the current iterate has delta < TOL and PSD-verified blocks, the stagnation
// rule fires (the best PSD-verified candidate is returned), or TOTAL_STEPS iterations were run;
// the ghost trajectory (states, pens) records every iterate and the penalty that produced it
// (the problem sdp is used by the specification only)
#[allow(unused_variables)]
pub fn sdpad_loop(sdp: &SDP, s: &SdpadSetup, check: bool, var: &JacobiVariant) -> (o: SdpadOutcome)
    requires
        sdpad_setup_ok(sdp, check, *s),
    ensures
        outcome_ok(sdp, s, check, o),
{
    let ghost d = s.var_dim as int;
    let ghost cols = pk_len(d);
    proof {
        lemma_setup_wgy(sdp, check, s);
        lemma_dim_fits_idx(d, d);
    }
    let var_dim = s.var_dim;
    let (mut it_stag, mut it_pinf, mut it_dinf): (i32, i32, i32) = (0, 0, 0);
    let mut penalty = PEN_INIT;
    let var_dim_tot = (var_dim * (var_dim + 1)) / 2;
    let (prim_z, dual_s): (SymMatrix, SymMatrix) = initial_iterates(var_dim);
    let mut st = WgyState {
        prim_z, dual_s, dual_y: Vec::new(), diff: Vec::new(), dual_sfull: Vec::with_capacity(var_dim),
        inverse_y: Vec::with_capacity(s.c_len), dual_sfull_temp: Vec::with_capacity(var_dim),
    };
    let mut best_delta = pos_infinity();
    let mut best_prim_z: SymMatrix = Vec::with_capacity(var_dim_tot);
    let (mut best_feas_delta, mut best_pinf, mut best_dinf, mut best_gap): (f64, f64, f64, f64) =
        (pos_infinity(), pos_infinity(), pos_infinity(), pos_infinity());
    let mut best_prim_val: Option<f64> = None;
    // the trajectory so far (states, pens), the index in it of the recorded best candidate,
    // the deltas of the iterations so far and which of them were recorded as candidates
    let ghost mut states: Seq<WgyState> = Seq::empty().push(st);
    let ghost mut pens: Seq<Penalty> = Seq::empty();
    let ghost mut best_idx: int = 0;
    let ghost mut deltas: Seq<f64> = Seq::empty();
    let ghost mut feasible: Seq<bool> = Seq::empty();

    // begin main computation
    let mut count: usize = 0;
    loop
        invariant
            sdpad_setup_ok(sdp, check, *s),
            wgy_setup_ok(s),
            d == s.var_dim as int,
            cols == pk_len(d),
            st.prim_z@.len() == cols,
            st.dual_s@.len() == cols,
            rv(PEN_MIN) <= rv(penalty) <= rv(PEN_MAX),
            0 <= it_pinf < PEN_COUNT,
            0 <= it_dinf < PEN_COUNT,
            count < TOTAL_STEPS,
            trajectory_ok(s, check, states, pens),
            pens.len() == count,
            states.last() == st,
            count == 0 ==> rv(penalty) == rv(PEN_INIT) && it_pinf == 0 && it_dinf == 0,
            count >= 1 ==> penalty_follows(s, pens.last(), &states.last(),
                Penalty { mu: rv(penalty), ip: it_pinf as int, id: it_dinf as int }),
            deltas.len() == count,
            forall|j: int| 0 <= j < count ==> rv(#[trigger] deltas[j]) == iterate_delta_r(s, &states[j + 1]),
            feasible.len() == count,
            it_stag <= count,
            stagnation_ok(rvs(deltas), rv(best_delta), it_stag as int),
            best_feasible_ok(rvs(deltas), feasible, rv(best_feas_delta), best_prim_val is Some),
            best_prim_val matches Some(v)
                ==> best_record_ok(s, states, feasible, best_idx, best_prim_z@, rv(best_pinf), rv(best_dinf), rv(best_gap), rv(v), rv(best_feas_delta)),
        decreases TOTAL_STEPS - count,
    {
        count += 1;

        let ghost st0: WgyState = st;
        let ghost pen = Penalty { mu: rv(penalty), ip: it_pinf as int, id: it_dinf as int };
        wgy_update(s, check, var, penalty, STEP, &mut st);
        proof {
            assert(wgy_update_ok(s, check, pen.mu, rv(STEP), &st0, &st));
            lemma_trajectory_step(s, check, states, pens, pen, st);
            states = states.push(st);
            pens = pens.push(pen);
            lemma_pos_part_psd_sym(rvs(st.dual_sfull@), rvs(st.dual_s@), d);
        }
        let (pinf, dinf, prim_val, gap, delta) = kkt_quantities(s, &st);
        let ghost deltas0 = deltas;
        proof {
            deltas = deltas.push(delta);
            assert(rvs(deltas) =~= rvs(deltas0).push(rv(delta)));
            lemma_stag_step(rvs(deltas0), rv(best_delta), it_stag as int, rv(delta));
            assert forall|j: int| 0 <= j < count implies rv(#[trigger] deltas[j]) == iterate_delta_r(s, &states[j + 1]) by {
                if j < count - 1 {
                    assert(rv(deltas0[j]) == iterate_delta_r(s, &states[j + 1]));
                }
            }
        }

        if residual_is_small(delta, TOL) && psd_block_check(&st.prim_z, &s.psd_blocks) {
            let result = read_solution(s, &st.prim_z);
            let o = SdpadOutcome {
                result, value: prim_val, pinf, dinf, gap, termination: Termination::Accuracy,
                states: Ghost(states), pens: Ghost(pens), iterate: Ghost(count as int),
                deltas: Ghost(deltas), feasible: Ghost(feasible.push(true)),
                best_delta: Ghost(if rv(delta) > rv(best_delta) { best_delta } else { delta }),
                it_stag: Ghost(if rv(delta) > rv(best_delta) { (it_stag + 1) as i32 } else { 0 }),
            };
            proof {
                lemma_objective_value(sdp, check, s, rvs(st.prim_z@));
                assert(states[count as int] == st && solution_ok(s, rvs(st.prim_z@), o.result@));
            }
            return o;
        }

        // record best solution so far
        let record = !(delta > best_feas_delta) && psd_block_check(&st.prim_z, &s.psd_blocks);
        proof {
            lemma_best_feasible_step(rvs(deltas0), feasible, rv(best_feas_delta), best_prim_val is Some, rv(delta), record);
            feasible = feasible.push(record);
        }
        if record {
            best_feas_delta = delta;
            best_pinf = pinf;
            best_dinf = dinf;
            best_gap = gap;
            best_prim_z.clear();
            best_prim_z.extend_from_slice(st.prim_z.as_slice());
            best_prim_val = Some(prim_val);
            proof {
                best_idx = count as int;
                assert(best_prim_z@ =~= st.prim_z@);
                assert(best_record_ok(s, states, feasible, best_idx, best_prim_z@, rv(best_pinf), rv(best_dinf), rv(best_gap), rv(prim_val), rv(best_feas_delta)));
            }
        }

        // record general stagnation
        if delta > best_delta {
            it_stag += 1;
        } else {
            best_delta = delta;
            it_stag = 0;
        }
        if let Some(best_val) = best_prim_val {
            if (it_stag > STAG1 && residual_is_small(delta, 1e-5)) ||
                (it_stag > STAG2 && residual_is_small(delta, 1e-4)) ||
                (it_stag > STAG3 && residual_is_small(delta, 1e-3)) {
                proof {
                    let j = best_idx - 1;
                    assert(wgy_update_ok(s, check, pens[j].mu, rv(STEP), &states[j], &states[j + 1]));
                    assert(states[j + 1] == states[best_idx]);
                }
                let result = read_solution(s, &best_prim_z);
                let o = SdpadOutcome {
                    result, value: best_val, pinf: best_pinf, dinf: best_dinf, gap: best_gap, termination: Termination::Stagnation,
                    states: Ghost(states), pens: Ghost(pens), iterate: Ghost(best_idx),
                    deltas: Ghost(deltas), feasible: Ghost(feasible), best_delta: Ghost(best_delta), it_stag: Ghost(it_stag),
                };
                proof {
                    let stb = states[best_idx];
                    lemma_objective_value(sdp, check, s, rvs(stb.prim_z@));
                    assert(solution_ok(s, rvs(stb.prim_z@), o.result@));
                }
                return o;
            }
        }

        if count == TOTAL_STEPS {
            let result = read_solution(s, &st.prim_z);
            let o = SdpadOutcome {
                result, value: prim_val, pinf, dinf, gap, termination: Termination::MaxIter,
                states: Ghost(states), pens: Ghost(pens), iterate: Ghost(count as int),
                deltas: Ghost(deltas), feasible: Ghost(feasible), best_delta: Ghost(best_delta), it_stag: Ghost(it_stag),
            };
            proof {
                lemma_objective_value(sdp, check, s, rvs(st.prim_z@));
                assert(states[count as int] == st && solution_ok(s, rvs(st.prim_z@), o.result@));
            }
            return o;
        }

        let (p, ip, id) = penalty_update(penalty, pinf, dinf, it_pinf, it_dinf);
        penalty = p;
        it_pinf = ip;
        it_dinf = id;
        proof {
            assert(penalty_follows(s, pen, &st, Penalty { mu: rv(penalty), ip: it_pinf as int, id: it_dinf as int }));
        }
    }
}

} // verus!

// the alternating direction dual augmented Lagrangian method: the setup, the verified loop,
// and the report of how it terminated
pub fn sdpad(sdp : &SDP, check_constraints : bool, jac_variant : &JacobiVariant) -> (Vec<SymMatrix>, f64) {
    let setup = sdpad_setup(sdp, check_constraints);
    let o = sdpad_loop(sdp, &setup, check_constraints, jac_variant);
    match o.termination {
        Termination::Accuracy => {
            println!("\nsdpad terminated with desired accuracy\n");
        }
        Termination::Stagnation => {
            println!("\nsdpad terminated due to stagnation but with reasonable accuracy");
            println!("solution quality of best solution found: pinf={:.3e} dinf={:.3e} gap={:.3e}", o.pinf, o.dinf, o.gap);
        }
        Termination::MaxIter => {
            println!("\nsdpad terminated after {} iterations", TOTAL_STEPS);
            println!("solution quality of final iteration: pinf={:.3e} dinf={:.3e} gap={:.3e}", o.pinf, o.dinf, o.gap);
        }
    }
    (o.result, o.value)
}

// testing

#[cfg(test)]
#[path = "sdp_tests.rs"]
mod tests;
