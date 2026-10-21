use vstd::prelude::*;
pub use crate::lin_alg::sym_matrix::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::reduce_spec::*;
#[cfg(verus_keep_ghost)]
use crate::sdp_spec::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_mod_multiples_basic, lemma_div_multiples_vanish};
use crate::trusted::abort;
use vstd::slice::slice_to_vec;

// linear map on symmetric matrices that is designed for user input
type MatrixSym = Vec<Vec<Vector>>;

verus! {

broadcast use crate::trusted::group_real_model;

// semidefinite programming problem
// (the fields are public so that the verified contracts can refer to the user data)
pub struct SDP {
    pub objective: MatrixSym,
    pub constraint: (Vec<MatrixSym>, Vector),
    pub lmi: Vec<(Vec<MatrixSym>, Matrix)>,
}

impl SDP {
    pub fn new(
        objective: MatrixSym,
        constraint: (Vec<MatrixSym>, Vector),
        lmi: Vec<(Vec<MatrixSym>, Matrix)>,
    ) -> (s: Self)
        ensures
            s.objective == objective,
            s.constraint == constraint,
            s.lmi == lmi,
    {
        Self { objective, constraint, lmi }
    }

    pub fn objective(&self) -> (r: &MatrixSym)
        ensures
            *r == self.objective,
    {
        &self.objective
    }

    pub fn constraint(&self) -> (r: &(Vec<MatrixSym>, Vector))
        ensures
            *r == self.constraint,
    {
        &self.constraint
    }

    pub fn lmi(&self) -> (r: &Vec<(Vec<MatrixSym>, Matrix)>)
        ensures
            *r == self.lmi,
    {
        &self.lmi
    }
}



/// The dimension of the symmetric matrices a MatrixSym acts on: the length of its first row.
pub open spec fn symmap_dim(m: Seq<Vec<Vector>>) -> int {
    if m.len() == 0 { 0int } else { m[0]@.len() as int }
}

// flatten the rows of a MatrixSym
// (the column-major matrix with one row per row of mat and one column per packed entry:
// entry (r, pk_idx(j, i)) is mat[r][i][j])
fn matrixsym_flatten(mat: &MatrixSym) -> (result: Matrix)
    requires
        symmap_shape(mat@, symmap_dim(mat@)),
        dim_fits(symmap_dim(mat@)),
        mat@.len() * pk_len(symmap_dim(mat@)) <= usize::MAX,
    ensures
        result@.len() == mat@.len() * pk_len(symmap_dim(mat@)),
        forall|r: int, i: int, j: int| 0 <= r < mat@.len() && 0 <= j <= i < symmap_dim(mat@)
            ==> rv(#[trigger] result@[cm_idx(mat@.len() as int, r, pk_idx(j, i))]) == rv(mat@[r]@[i]@[j]),
{
    let dim_c = mat.first().unwrap_or(&vec![]).len();
    let ghost (dc, rows) = (dim_c as int, mat@.len() as int);
    proof {
        lemma_dim_fits_idx(dc, dc);
        lemma_pk_col_start_zero();
    }
    let mut result: Matrix = Vec::with_capacity(mat.len() * ((dim_c * (dim_c + 1)) / 2));
    for i in 0..dim_c
        invariant
            dc == dim_c as int,
            rows == mat@.len() as int,
            symmap_shape(mat@, dc),
            dim_fits(dc),
            rows * pk_len(dc) <= usize::MAX,
            result@.len() == pk_col_start(i as int) * rows,
            forall|r: int, i2: int, j: int| 0 <= r < rows && 0 <= j <= i2 < i
                ==> rv(#[trigger] result@[cm_idx(rows, r, pk_idx(j, i2))]) == rv(mat@[r]@[i2]@[j]),
    {
        proof {
            lemma_dim_fits_idx(dc, i as int);
            lemma_pk_col_start_step(i as int);
            lemma_pk_col_start_monotone(i as int + 1, dc);
            // the entries filled so far are exactly those of the columns before pk_idx(0, i)
            assert forall|r: int, i2: int, j2: int| 0 <= r < rows && 0 <= j2 <= i2 < dc && pk_idx(j2, i2) < pk_idx(0, i as int)
                implies rv(#[trigger] result@[cm_idx(rows, r, pk_idx(j2, i2))]) == rv(mat@[r]@[i2]@[j2]) by {
                lemma_pk_idx_order(j2, i2, 0, i as int);
            }
        }
        for j in 0..=i
            invariant
                i < dim_c,
                dc == dim_c as int,
                rows == mat@.len() as int,
                symmap_shape(mat@, dc),
                rows * pk_len(dc) <= usize::MAX,
                pk_col_start(i as int + 1) == pk_col_start(i as int) + i + 1,
                pk_col_start(i as int + 1) <= pk_len(dc),
                result@.len() == pk_idx(j as int, i as int) * rows,
                forall|r: int, i2: int, j2: int| 0 <= r < rows && 0 <= j2 <= i2 < dc && pk_idx(j2, i2) < pk_idx(j as int, i as int)
                    ==> rv(#[trigger] result@[cm_idx(rows, r, pk_idx(j2, i2))]) == rv(mat@[r]@[i2]@[j2]),
        {
            let ghost t = pk_idx(j as int, i as int);
            proof {
                assert((t + 1) * rows == t * rows + rows) by (nonlinear_arith);
                assert((t + 1) * rows <= pk_len(dc) * rows) by (nonlinear_arith)
                    requires t + 1 <= pk_len(dc), rows >= 0;
                assert(pk_len(dc) * rows == rows * pk_len(dc)) by (nonlinear_arith);
            }
            for r in 0..mat.len()
                invariant
                    i < dim_c,
                    j <= i,
                    dc == dim_c as int,
                    rows == mat@.len() as int,
                    symmap_shape(mat@, dc),
                    t == pk_idx(j as int, i as int),
                    (t + 1) * rows == t * rows + rows,
                    (t + 1) * rows <= usize::MAX,
                    result@.len() == t * rows + r,
                    forall|r2: int, i2: int, j2: int| 0 <= r2 < rows && 0 <= j2 <= i2 < dc && pk_idx(j2, i2) < t
                        ==> rv(#[trigger] result@[cm_idx(rows, r2, pk_idx(j2, i2))]) == rv(mat@[r2]@[i2]@[j2]),
                    forall|r2: int| 0 <= r2 < r ==> rv(#[trigger] result@[cm_idx(rows, r2, t)]) == rv(mat@[r2]@[i as int]@[j as int]),
            {
                let ghost prev = result@;
                result.push(mat[r][i][j]);
                proof {
                    assert forall|r2: int, i2: int, j2: int| 0 <= r2 < rows && 0 <= j2 <= i2 < dc && pk_idx(j2, i2) < t
                        implies rv(#[trigger] result@[cm_idx(rows, r2, pk_idx(j2, i2))]) == rv(mat@[r2]@[i2]@[j2]) by {
                        lemma_cm_idx_bound(rows, r2, pk_idx(j2, i2), t);
                        assert(result@[cm_idx(rows, r2, pk_idx(j2, i2))] == prev[cm_idx(rows, r2, pk_idx(j2, i2))]);
                    }
                    assert forall|r2: int| 0 <= r2 < r + 1
                        implies rv(#[trigger] result@[cm_idx(rows, r2, t)]) == rv(mat@[r2]@[i as int]@[j as int]) by {
                        if r2 < r {
                            assert(result@[cm_idx(rows, r2, t)] == prev[cm_idx(rows, r2, t)]);
                        }
                    }
                }
            }
            proof {
                assert forall|r: int, i2: int, j2: int| 0 <= r < rows && 0 <= j2 <= i2 < dc && pk_idx(j2, i2) < t + 1
                    implies rv(#[trigger] result@[cm_idx(rows, r, pk_idx(j2, i2))]) == rv(mat@[r]@[i2]@[j2]) by {
                    if pk_idx(j2, i2) == t {
                        lemma_pk_idx_injective(j2, i2, j as int, i as int);
                    }
                }
            }
        }
        proof {
            assert forall|r: int, i2: int, j: int| 0 <= r < rows && 0 <= j <= i2 < i + 1
                implies rv(#[trigger] result@[cm_idx(rows, r, pk_idx(j, i2))]) == rv(mat@[r]@[i2]@[j]) by {
                lemma_pk_idx_bound(j, i2, i as int + 1);
                assert(pk_idx(j, i2) < pk_idx(i as int + 1, i as int));
            }
        }
    }
    proof {
        assert(symmap_dim(mat@) == dc);
        assert(pk_col_start(dc) * rows == rows * pk_len(dc)) by (nonlinear_arith)
            requires pk_col_start(dc) == pk_len(dc);
    }
    result
}

/* ---------- conversion to standard form ---------- */

/// Number of user variables: one objective map per variable.
pub open spec fn n_vars(sdp: &SDP) -> int {
    sdp.objective@.len() as int
}

/// The LMI block sizes: the integer square roots of the lengths of the constant matrices.
pub open spec fn bdims_r(lmis: Seq<(Vec<MatrixSym>, Matrix)>) -> Seq<int> {
    Seq::new(lmis.len(), |i: int| isqrt_r(lmis[i].1@.len() as int))
}

/// The orders of the user variables, read from the maps of the first LMI (the number of
/// columns of the first row of map n).
pub open spec fn vdims_r(lmis: Seq<(Vec<MatrixSym>, Matrix)>, nv: int) -> Seq<int> {
    Seq::new(nv as nat, |n: int| lmis[0].0@[n]@[0]@.len() as int)
}

/// The number of maps of the first LMI (one per user variable).
pub open spec fn n_maps(lmis: Seq<(Vec<MatrixSym>, Matrix)>) -> int {
    lmis[0].0@.len() as int
}

/// The orders stored in a dimension vector, as integers.
pub open spec fn dims_of(v: Seq<usize>) -> Seq<int> {
    Seq::new(v.len(), |n: int| v[n] as int)
}

/// Shape requirements of the LMI data: every LMI has one map per user variable, the constant
/// matrices are nonempty squares, and map n of every LMI is a MatrixSym of the variable's order
/// with one row per entry of the constant matrix.
pub open spec fn lmi_shapes_ok(lmis: Seq<(Vec<MatrixSym>, Matrix)>, nv: int) -> bool {
    let bd = bdims_r(lmis);
    let vd = vdims_r(lmis, nv);
    &&& lmis.len() >= 1
    &&& forall|i: int| 0 <= i < lmis.len() ==> (#[trigger] lmis[i]).0@.len() == nv
    &&& forall|i: int| 0 <= i < lmis.len() ==> #[trigger] bd[i] >= 1 && bd[i] * bd[i] == lmis[i].1@.len()
    &&& forall|n: int| 0 <= n < nv ==> #[trigger] vd[n] >= 1
    &&& forall|i: int, n: int| 0 <= i < lmis.len() && 0 <= n < nv
            ==> symmap_shape((#[trigger] lmis[i].0@[n])@, vd[n]) && lmis[i].0@[n]@.len() == bd[i] * bd[i]
}

/// Well-formed input with LMIs: shapes of the LMI data, the objective maps and the equality
/// constraint maps, and the standard-form sizes fitting in usize.
pub open spec fn sdp_lmi_input_ok(sdp: &SDP) -> bool {
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let sd = std_dim(bd, vd);
    let m = sdp.constraint.1@.len() as int;
    &&& lmi_shapes_ok(sdp.lmi@, nv)
    &&& forall|n: int| 0 <= n < nv ==> part_shape((#[trigger] sdp.objective@[n])@, vd[n])
    &&& sdp.constraint.0@.len() == 0 || sdp.constraint.0@.len() == nv
    &&& sdp.constraint.0@.len() == 0 ==> m == 0
    &&& forall|n: int| 0 <= n < sdp.constraint.0@.len()
            ==> symmap_shape((#[trigger] sdp.constraint.0@[n])@, vd[n]) && sdp.constraint.0@[n]@.len() == m
    &&& dim_fits(sd)
    &&& (pk_len(sd) + 1) * (lmi_rows(bd) + m + 1) <= usize::MAX
}

/// The objective maps have the shapes of the variable orders symm_dims, which are at least 1.
pub open spec fn objective_shapes_ok(objective: Seq<Vec<Vec<f64>>>, symm_dims: Seq<usize>) -> bool {
    forall|n: int| 0 <= n < objective.len()
        ==> part_shape((#[trigger] objective[n])@, symm_dims[n] as int) && symm_dims[n] >= 1
}

/// The objective maps have the shapes of their variables (sdp_lmi_input_ok), stated on the
/// returned variable orders.
proof fn lemma_objective_shapes(sdp: &SDP, symm_dims: Seq<usize>)
    requires
        sdp_lmi_input_ok(sdp),
        dims_of(symm_dims) == vdims_r(sdp.lmi@, n_vars(sdp)),
    ensures
        objective_shapes_ok(sdp.objective@, symm_dims),
{
    let nv = n_vars(sdp);
    let vd = vdims_r(sdp.lmi@, nv);
    assert forall|n: int| 0 <= n < sdp.objective@.len()
        implies part_shape((#[trigger] sdp.objective@[n])@, symm_dims[n] as int) && symm_dims[n] >= 1 by {
        assert(part_shape(sdp.objective@[n]@, vd[n]));
        assert(vd[n] >= 1);
        assert(dims_of(symm_dims)[n] == vd[n]);
    }
}

/// The single-variable case (no LMI): one objective map and at most one constraint map, both of
/// the variable's order d (the number of columns of the objective map); the stacked constraint
/// system fits in usize.
pub open spec fn sdp_single_input_ok(sdp: &SDP) -> bool {
    let d = sdp.objective@[0]@.len() as int;
    let m = sdp.constraint.1@.len() as int;
    &&& sdp.objective@.len() == 1
    &&& d >= 1
    &&& part_shape(sdp.objective@[0]@, d)
    &&& dim_fits(d)
    &&& sdp.constraint.0@.len() <= 1
    &&& sdp.constraint.0@.len() == 1
            ==> symmap_shape(sdp.constraint.0@[0]@, d) && sdp.constraint.0@[0]@.len() == m && pk_len(d) * (m + 1) <= usize::MAX
}

/// Some entry of the single constraint map or of its right-hand side is nonzero (needed for
/// the tolerance of the constraint reduction).
pub open spec fn single_constraint_nonzero(sdp: &SDP) -> bool {
    let d = sdp.objective@[0]@.len() as int;
    ||| exists|e: int, i: int, j: int| 0 <= e < sdp.constraint.1@.len() && 0 <= j <= i < d
            && rv(#[trigger] sdp.constraint.0@[0]@[e]@[i]@[j]) != 0.0real
    ||| exists|e: int| 0 <= e < sdp.constraint.1@.len() && rv(#[trigger] sdp.constraint.1@[e]) != 0.0real
}

/// red is the reduced system of the augmented matrix aug (through some echelon form).
pub open spec fn reduced_from(aug: Seq<real>, red: Seq<real>, dim: int) -> bool {
    exists|e: Seq<real>| reduced_system(aug, e, red, dim)
}

/// The augmented system returned by the conversion: the constraint matrix followed by the
/// right-hand side column (the first rows entries of the point vector).
pub open spec fn result_aug(mat: Seq<f64>, point: Seq<f64>, rows: int) -> Seq<real> {
    rvs(mat) + rvs(point.subrange(0, rows))
}

/// The contract of the conversion with LMIs: the returned orders, the objective identity
/// <obj, Z> = C_1(Z_1) + .. + C_n(Z_n), the zero pattern, and the constraint rows (reduced when
/// check_constraints holds).
pub open spec fn std_form_lmi_ok(sdp: &SDP, res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>), check: bool) -> bool {
    let nv = n_vars(sdp);
    let bd = bdims_r(sdp.lmi@);
    let vd = vdims_r(sdp.lmi@, nv);
    let bsum = block_sum(bd);
    let sd = std_dim(bd, vd);
    let cols = pk_len(sd);
    let nz = res.1.1@.len() as int;
    let rows = res.2@.len() as int - nz;
    &&& res.3 as int == sd
    &&& dims_of(res.4@) == vd
    &&& dims_of(res.5@) == bd
    &&& res.0@.len() == cols
    &&& forall|z: Seq<real>| z.len() == cols ==> #[trigger] dot_r(rvs(res.0@), z) == obj_r(sdp.objective@, bsum, vd, z)
    &&& zeros_ok(res.1.1@, block_sizes(bd, vd), sd)
    &&& nz == zeros_len(block_sizes(bd, vd), sd)
    &&& rows >= 0
    &&& res.1.0@.len() == rows * cols
    &&& forall|q: int| rows <= q < res.2@.len() ==> rv(#[trigger] res.2@[q]) == 0.0real
    &&& exists|aug: Seq<real>| #[trigger] std_rows_ok(sdp.lmi@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug)
            && (if check { reduced_from(aug, result_aug(res.1.0@, res.2@, rows), sd) } else { result_aug(res.1.0@, res.2@, rows) == aug })
}

/// The contract of the conversion without LMIs: the objective is the packed objective map, the
/// variable keeps its order, and the constraint rows read A(Z) = b (reduced when check holds).
pub open spec fn std_form_single_ok(sdp: &SDP, res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>), check: bool) -> bool {
    let d = sdp.objective@[0]@.len() as int;
    let cols = pk_len(d);
    let rows = res.2@.len() as int;
    &&& res.3 as int == d
    &&& res.4@.len() == 0
    &&& res.5@.len() == 0
    &&& res.1.1@.len() == 0
    &&& res.0@.len() == cols
    &&& rvs(res.0@) == part_r(sdp.objective@[0]@, d)
    &&& res.1.0@.len() == rows * cols
    &&& sdp.constraint.0@.len() == 0 ==> rows == 0
    &&& sdp.constraint.0@.len() == 1 ==> exists|aug: Seq<real>| #[trigger] single_rows_ok(sdp.constraint.0@[0]@, sdp.constraint.1@, d, aug)
            && (if check { reduced_from(aug, result_aug(res.1.0@, res.2@, rows), d) } else { result_aug(res.1.0@, res.2@, rows) == aug })
}

/// A partitioned symmetric matrix as a packed vector (what `concat` computes on a row of a MatrixSym).
fn part_concat(v: &Vec<Vector>, dim: usize) -> (result: Vector)
    requires
        part_shape(v@, dim as int),
        dim_fits(dim as int),
    ensures
        result@.len() == pk_len(dim as int),
        rvs(result@) == part_r(v@, dim as int),
{
    let ghost d = dim as int;
    proof {
        lemma_dim_fits_idx(d, d);
        lemma_pk_col_start_zero();
    }
    let mut result: Vector = Vec::with_capacity((dim * (dim + 1)) / 2);
    for c in 0..dim
        invariant
            d == dim as int,
            part_shape(v@, d),
            dim_fits(d),
            result@.len() == pk_col_start(c as int),
            forall|r2: int, c2: int| 0 <= r2 <= c2 < c ==> rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(v@[c2]@[r2]),
    {
        let ghost ci = c as int;
        proof {
            lemma_dim_fits_idx(d, ci);
            lemma_pk_col_start_step(ci);
            lemma_pk_col_start_monotone(ci + 1, d);
        }
        for r in 0..=c
            invariant
                d == dim as int,
                ci == c as int,
                c < dim,
                part_shape(v@, d),
                pk_col_start(ci + 1) == pk_col_start(ci) + ci + 1,
                pk_col_start(ci + 1) <= pk_len(d),
                pk_len(d) <= usize::MAX,
                result@.len() == pk_idx(r as int, ci),
                forall|r2: int, c2: int| 0 <= r2 <= c2 < c ==> rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(v@[c2]@[r2]),
                forall|r2: int| 0 <= r2 < r ==> rv(#[trigger] result@[pk_idx(r2, ci)]) == rv(v@[ci]@[r2]),
        {
            let ghost prev = result@;
            result.push(v[c][r]);
            proof {
                assert forall|r2: int, c2: int| 0 <= r2 <= c2 < c implies rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(v@[c2]@[r2]) by {
                    lemma_pk_idx_bound(r2, c2, ci);
                    assert(result@[pk_idx(r2, c2)] == prev[pk_idx(r2, c2)]);
                }
                assert forall|r2: int| 0 <= r2 < r + 1 implies rv(#[trigger] result@[pk_idx(r2, ci)]) == rv(v@[ci]@[r2]) by {
                    if r2 < r {
                        assert(result@[pk_idx(r2, ci)] == prev[pk_idx(r2, ci)]);
                    }
                }
            }
        }
    }
    proof {
        assert forall|t: int| 0 <= t < pk_len(d) implies #[trigger] rvs(result@)[t] == part_r(v@, d)[t] by {
            lemma_pk_unpack(t);
            let (r, c) = pk_unpack(t);
            if c >= d {
                lemma_pk_col_start_monotone(d, c);
            }
            assert(rv(result@[pk_idx(r, c)]) == rv(v@[c]@[r]));
        }
        assert(rvs(result@) =~= part_r(v@, d));
    }
    result
}

/// The LMI block sizes and the orders of the user variables (checked against the data), with
/// their sums.
fn lmi_dims(lmis: &Vec<(Vec<MatrixSym>, Matrix)>) -> (res: (Vec<usize>, Vec<usize>, usize, usize))
    requires
        lmi_shapes_ok(lmis@, n_maps(lmis@)),
        std_dim(bdims_r(lmis@), vdims_r(lmis@, n_maps(lmis@))) <= usize::MAX,
    ensures
        dims_of(res.0@) == bdims_r(lmis@),
        dims_of(res.1@) == vdims_r(lmis@, n_maps(lmis@)),
        res.2 as int == block_sum(bdims_r(lmis@)),
        res.3 as int == sum_upto(vdims_r(lmis@, n_maps(lmis@)), n_maps(lmis@)),
{
    let ghost nvars = n_maps(lmis@);
    let ghost bd = bdims_r(lmis@);
    let ghost vd = vdims_r(lmis@, nvars);
    let ghost s = lmis@.len() as int;
    let (mut block_dims, mut symm_dims): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let (mut block_size_sum, mut symm_size_sum) = (0, 0);
    let mut map_len_old = 0;
    proof {
        assert forall|i: int| 0 <= i < s implies #[trigger] bd[i] >= 0 by {}
        lemma_sum_upto_monotone(bd, s, s);
        assert forall|n: int| 0 <= n < nvars implies #[trigger] vd[n] >= 0 by {}
    }
    for i in 0..lmis.len()
        invariant
            lmi_shapes_ok(lmis@, nvars),
            bd == bdims_r(lmis@),
            vd == vdims_r(lmis@, nvars),
            s == lmis@.len() as int,
            forall|k: int| 0 <= k < s ==> #[trigger] bd[k] >= 0,
            forall|n: int| 0 <= n < nvars ==> #[trigger] vd[n] >= 0,
            std_dim(bd, vd) <= usize::MAX,
            block_dims@.len() == i,
            forall|k: int| 0 <= k < i ==> #[trigger] block_dims@[k] as int == bd[k],
            block_size_sum as int == sum_upto(bd, i as int),
            i >= 1 ==> symm_dims@.len() == nvars && symm_size_sum as int == sum_upto(vd, nvars)
                && forall|n: int| 0 <= n < nvars ==> #[trigger] symm_dims@[n] as int == vd[n],
            i == 0 ==> symm_dims@.len() == 0 && symm_size_sum == 0,
    {
        let ghost ii = i as int;
        let maps = &lmis[i].0;
        let mat = &lmis[i].1;
        let block_size = mat.len().isqrt();
        proof {
            assert(is_isqrt(block_size as int, mat@.len() as int));
            lemma_isqrt_unique(block_size as int, mat@.len() as int);
            lemma_sum_upto_monotone(bd, ii + 1, s);
        }
        if !(block_size > 0 && block_size * block_size == mat.len()) {
            abort("the constant matrix for each LMI must be nonempty and square");
        }
        block_dims.push(block_size);
        block_size_sum += block_size;
        if i == 0 {
            map_len_old = maps.len();
            for n in 0..maps.len()
                invariant
                    lmi_shapes_ok(lmis@, nvars),
                    vd == vdims_r(lmis@, nvars),
                    s == lmis@.len() as int,
                    i == 0,
                    maps@ == lmis@[0].0@,
                    maps@.len() == nvars,
                    forall|n2: int| 0 <= n2 < nvars ==> #[trigger] vd[n2] >= 0,
                    std_dim(bd, vd) <= usize::MAX,
                    block_size_sum as int == sum_upto(bd, 1),
                    bd == bdims_r(lmis@),
                    forall|k: int| 0 <= k < s ==> #[trigger] bd[k] >= 0,
                    symm_dims@.len() == n,
                    forall|n2: int| 0 <= n2 < n ==> #[trigger] symm_dims@[n2] as int == vd[n2],
                    symm_size_sum as int == sum_upto(vd, n as int),
            {
                proof {
                    assert(lmis@[0].0@[n as int]@.len() == bd[0] * bd[0]);
                    assert(bd[0] * bd[0] >= 1) by (nonlinear_arith)
                        requires bd[0] >= 1;
                    assert(symmap_shape(lmis@[0].0@[n as int]@, vd[n as int]));
                }
                let row0 = &maps[n][0];
                if row0.len() == 0 {
                    abort("first row of some matrix in affine combination is empty");
                }
                let symm_size = row0[row0.len() - 1].len();
                proof {
                    assert(part_shape(maps@[n as int]@[0]@, vd[n as int]));
                    assert(symm_size as int == vd[n as int]);
                    lemma_sum_upto_monotone(vd, n as int + 1, nvars);
                    lemma_sum_upto_monotone(bd, 1, s);
                }
                symm_dims.push(symm_size);
                symm_size_sum += symm_size;
            }
        } else if maps.len() != map_len_old {
            abort("each LMI must have the same number of decision variables");
        }
    }
    proof {
        assert(dims_of(block_dims@) =~= bd);
        assert(dims_of(symm_dims@) =~= vd);
    }
    (block_dims, symm_dims, block_size_sum, symm_size_sum)
}

/// One map per user variable, each a MatrixSym of the variable's order with the given number of rows.
pub open spec fn maps_shape_ok(maps: Seq<Vec<Vec<Vector>>>, symm_dims: Seq<usize>, rows: int) -> bool {
    &&& maps.len() == symm_dims.len()
    &&& forall|n: int| 0 <= n < maps.len()
            ==> symmap_shape((#[trigger] maps[n])@, symm_dims[n] as int) && maps[n]@.len() == rows && symm_dims[n] >= 1
}

/// The sum over the user variables of the maps embedded at the variable's two blocks (with
/// opposite signs): row e of the result reads sum_n maps[n][e](Z_n) on the standard-form variable.
/// Columns before the first variable block are zero.
fn maps_sum(maps: &Vec<MatrixSym>, symm_dims: &Vec<usize>, block_size_sum: usize, total_dim: usize, rows: usize) -> (sum: Matrix)
    requires
        maps_shape_ok(maps@, symm_dims@, rows as int),
        dim_fits(total_dim as int),
        var_off(block_size_sum as int, dims_of(symm_dims@), symm_dims@.len() as int) <= total_dim,
        rows * (pk_len(total_dim as int) + 1) <= usize::MAX,
    ensures
        sum@.len() == rows * pk_len(total_dim as int),
        forall|e: int, z: Seq<real>| 0 <= e < rows && z.len() == pk_len(total_dim as int)
            ==> #[trigger] row_dot_r(rvs(sum@), rows as int, e, z) == maps_row_r(maps@, block_size_sum as int, dims_of(symm_dims@), e, z),
        forall|e: int, t: int| 0 <= e < rows && 0 <= t < pk_col_start(block_size_sum as int)
            ==> rv(#[trigger] sum@[cm_idx(rows as int, e, t)]) == 0.0real,
{
    let ghost (sd, bsum, nv) = (total_dim as int, block_size_sum as int, maps@.len() as int);
    let ghost vd = dims_of(symm_dims@);
    let ghost cols = pk_len(sd);
    proof {
        lemma_dim_fits_idx(sd, sd);
        assert forall|n: int| 0 <= n < nv implies #[trigger] vd[n] >= 1 by {
            assert(maps@[n]@.len() == rows);
            assert(vd[n] == symm_dims@[n] as int);
        }
        lemma_sum_upto_monotone(vd, 0, nv);
        lemma_pk_col_start_monotone(0, sd);
        lemma_pk_col_start_monotone(bsum, sd);
        assert(rows * (cols + 1) == rows * cols + rows) by (nonlinear_arith);
        assert(pk_len(sd) * rows == rows * pk_len(sd)) by (nonlinear_arith);
    }
    let total_size = (total_dim * (total_dim + 1)) / 2;
    let mut sum = zero_mat(total_size, rows);
    let mut symm_offset = 0;
    proof {
        assert forall|e: int, z: Seq<real>| 0 <= e < rows && z.len() == cols
            implies #[trigger] row_dot_r(rvs(sum@), rows as int, e, z) == sum_fn_r(|n2: int| map_term_r(maps@, bsum, vd, e, z, n2), 0) by {
            lemma_row_dot_zero(rvs(sum@), rows as int, e, z);
        }
        let z0 = rvs(sum@);
        assert(z0 == const_r((total_size * rows) as nat, 0.0real));
        assert forall|e: int, t: int| 0 <= e < rows && 0 <= t < pk_col_start(bsum)
            implies rv(#[trigger] sum@[cm_idx(rows as int, e, t)]) == 0.0real by {
            lemma_cm_idx_bound(rows as int, e, t, cols);
            assert(cm_idx(rows as int, e, t) < total_size * rows);
            assert(z0[cm_idx(rows as int, e, t)] == 0.0real);
        }
    }
    for n in 0..maps.len()
        invariant
            sd == total_dim as int,
            bsum == block_size_sum as int,
            nv == maps@.len() as int,
            vd == dims_of(symm_dims@),
            cols == pk_len(sd),
            maps_shape_ok(maps@, symm_dims@, rows as int),
            forall|n2: int| 0 <= n2 < nv
                ==> symmap_shape((#[trigger] maps@[n2])@, vd[n2]) && maps@[n2]@.len() == rows && vd[n2] >= 1,
            dim_fits(sd),
            var_off(bsum, vd, nv) <= sd,
            rows * cols <= usize::MAX,
            rows * (cols + 1) <= usize::MAX,
            sum@.len() == rows * cols,
            symm_offset as int == 2 * sum_upto(vd, n as int),
            var_off(bsum, vd, n as int) <= var_off(bsum, vd, nv),
            forall|e: int, z: Seq<real>| 0 <= e < rows && z.len() == cols
                ==> #[trigger] row_dot_r(rvs(sum@), rows as int, e, z) == sum_fn_r(|n2: int| map_term_r(maps@, bsum, vd, e, z, n2), n as nat),
            forall|e: int, t: int| 0 <= e < rows && 0 <= t < pk_col_start(bsum)
                ==> rv(#[trigger] sum@[cm_idx(rows as int, e, t)]) == 0.0real,
    {
        let ghost ni = n as int;
        let dim = symm_dims[n];
        let ghost d = dim as int;
        let map = &maps[n];
        if map.len() != rows {
            abort("the size of a map in some LMI does not match the size of the constant matrix");
        }
        let ghost st1 = bsum + symm_offset as int;
        let ghost st2 = st1 + d;
        let ghost dd = symmap_dim(map@);
        let ghost p = pk_len(d);
        proof {
            lemma_sum_upto_monotone(vd, ni + 1, nv);
            if rows >= 1 {
                assert(part_shape(map@[0]@, d));
            }
            assert(dd == if rows == 0 { 0int } else { d });
            lemma_dim_fits_monotone(dd, sd);
            lemma_dim_fits_monotone(d, sd);
            lemma_dim_fits_idx(d, d);
            lemma_pk_col_start_monotone(dd, sd);
            lemma_pk_col_start_monotone(bsum, sd);
            assert(rows * pk_len(dd) <= rows * cols) by (nonlinear_arith)
                requires pk_len(dd) <= cols, rows >= 0;
            assert(p >= 1) by {
                lemma_pk_col_start_step(d - 1);
                lemma_pk_col_start_monotone(0, d - 1);
            }
            assert(rows * pk_len(dd) == rows * p) by (nonlinear_arith)
                requires rows == 0 || pk_len(dd) == p;
            lemma_mod_multiples_basic(rows as int, p);
            lemma_div_multiples_vanish(rows as int, p);
            assert(rows * p == p * rows) by (nonlinear_arith);
            assert(pk_len(sd) * rows == rows * pk_len(sd)) by (nonlinear_arith);
        }
        let map_flatten = matrixsym_flatten(map);
        proof {
            assert(map_flatten@.len() == rows * p);
            assert((map_flatten@.len() as int) % p == 0);
            assert((map_flatten@.len() as int) / p == rows);
        }
        let d1 = matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim);
        let d2 = matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim);
        let ghost prev = rvs(sum@);
        sum = matrix_add_diff(&sum, &d1, &d2);
        proof {
            let (m1, m2) = (rvs(d1@), rvs(d2@));
            assert(d1@.len() == rows * cols && d2@.len() == rows * cols);
            assert forall|e: int, z: Seq<real>| 0 <= e < rows && z.len() == cols
                implies #[trigger] row_dot_r(rvs(sum@), rows as int, e, z) == sum_fn_r(|n2: int| map_term_r(maps@, bsum, vd, e, z, n2), (n + 1) as nat) by {
                let w = symrow_r(map@, e, d);
                lemma_row_dot_add_sub(rvs(sum@), prev, m1, m2, rows as int, e, z);
                // the two embeddings of row e of the map
                assert forall|a: int, b: int| 0 <= a <= b < d implies m1[cm_idx(rows as int, e, #[trigger] pk_idx(st1 + a, st1 + b))] == w[pk_idx(a, b)] by {
                    lemma_part_entry(map@[e]@, d, a, b);
                    lemma_pk_idx_bound(st1 + a, st1 + b, sd);
                    lemma_cm_idx_bound(rows as int, e, pk_idx(st1 + a, st1 + b), cols);
                    lemma_cm_idx_bound(rows as int, e, pk_idx(a, b), pk_len(d));
                    assert(rv(d1@[cm_idx(rows as int, e, pk_idx(st1 + a, st1 + b))]) == rv(map_flatten@[cm_idx(rows as int, e, pk_idx(a, b))]));
                    assert(rv(map_flatten@[cm_idx(rows as int, e, pk_idx(a, b))]) == rv(map@[e]@[b]@[a]));
                }
                assert forall|a: int, b: int| 0 <= a <= b < d implies m2[cm_idx(rows as int, e, #[trigger] pk_idx(st2 + a, st2 + b))] == w[pk_idx(a, b)] by {
                    lemma_part_entry(map@[e]@, d, a, b);
                    lemma_pk_idx_bound(st2 + a, st2 + b, sd);
                    lemma_cm_idx_bound(rows as int, e, pk_idx(st2 + a, st2 + b), cols);
                    lemma_cm_idx_bound(rows as int, e, pk_idx(a, b), pk_len(d));
                    assert(rv(d2@[cm_idx(rows as int, e, pk_idx(st2 + a, st2 + b))]) == rv(map_flatten@[cm_idx(rows as int, e, pk_idx(a, b))]));
                    assert(rv(map_flatten@[cm_idx(rows as int, e, pk_idx(a, b))]) == rv(map@[e]@[b]@[a]));
                }
                assert forall|t: int| 0 <= t < cols && !is_block_idx(t, st1, d) implies m1[#[trigger] cm_idx(rows as int, e, t)] == 0.0real by {
                    lemma_cm_idx_bound(rows as int, e, t, cols);
                }
                assert forall|t: int| 0 <= t < cols && !is_block_idx(t, st2, d) implies m2[#[trigger] cm_idx(rows as int, e, t)] == 0.0real by {
                    lemma_cm_idx_bound(rows as int, e, t, cols);
                }
                lemma_embed_row_dot(m1, rows as int, e, w, z, sd, st1, d);
                lemma_embed_row_dot(m2, rows as int, e, w, z, sd, st2, d);
                lemma_dot_sub_right(w, blk_r(z, st1, d), blk_r(z, st2, d));
                assert(var_off(bsum, vd, ni) == st1);
                assert(map_term_r(maps@, bsum, vd, e, z, ni) == dot_r(w, zvar_r(z, st1, d)));
            }
            assert forall|e: int, t: int| 0 <= e < rows && 0 <= t < pk_col_start(bsum)
                implies rv(#[trigger] sum@[cm_idx(rows as int, e, t)]) == 0.0real by {
                lemma_cm_idx_bound(rows as int, e, t, cols);
                lemma_pk_col_start_monotone(bsum, sd);
                assert(!is_block_idx(t, st1, d)) by {
                    if is_block_idx(t, st1, d) {
                        let (a, b) = choose|a: int, b: int| 0 <= a <= b < d && t == #[trigger] pk_idx(st1 + a, st1 + b);
                        lemma_pk_idx_bound(st1 + a, st1 + b, st1 + b + 1);
                        lemma_pk_col_start_monotone(bsum, st1 + b);
                    }
                }
                assert(!is_block_idx(t, st2, d)) by {
                    if is_block_idx(t, st2, d) {
                        let (a, b) = choose|a: int, b: int| 0 <= a <= b < d && t == #[trigger] pk_idx(st2 + a, st2 + b);
                        lemma_pk_idx_bound(st2 + a, st2 + b, st2 + b + 1);
                        lemma_pk_col_start_monotone(bsum, st2 + b);
                    }
                }
                let idx = cm_idx(rows as int, e, t);
                assert(idx < rows * cols);
                assert(rvs(sum@) == vadd_r(prev, vsub_r(m1, m2)));
                assert(rvs(sum@)[idx] == prev[idx] + vsub_r(m1, m2)[idx]);
                assert(vsub_r(m1, m2)[idx] == m1[idx] - m2[idx]);
                assert(rv(sum@[idx]) == rvs(sum@)[idx]);
            }
        }
        symm_offset += 2 * dim;
    }
    sum
}

/// The rows of one LMI: H_i(Z_1, .., Z_n)[e] - B_i[e] for each entry e = c k + r of its k x k
/// block (B_i the diagonal block of the standard-form variable at block_offset).
fn lmi_block(maps: &Vec<MatrixSym>, symm_dims: &Vec<usize>, block_size_sum: usize, total_dim: usize, block_offset: usize, block_size: usize) -> (sum: Matrix)
    requires
        block_size >= 1,
        maps_shape_ok(maps@, symm_dims@, block_size * block_size),
        dim_fits(total_dim as int),
        var_off(block_size_sum as int, dims_of(symm_dims@), symm_dims@.len() as int) <= total_dim,
        block_offset + block_size <= block_size_sum,
        block_size * block_size * (pk_len(total_dim as int) + 1) <= usize::MAX,
    ensures
        sum@.len() == block_size * block_size * pk_len(total_dim as int),
        forall|e: int, z: Seq<real>| 0 <= e < block_size * block_size && z.len() == pk_len(total_dim as int)
            ==> #[trigger] row_dot_r(rvs(sum@), (block_size * block_size) as int, e, z)
                == maps_row_r(maps@, block_size_sum as int, dims_of(symm_dims@), e, z)
                    - sym_entry(z, block_offset + lmi_row_lo(block_size as int, e), block_offset + lmi_row_hi(block_size as int, e)),
        rv(sum@[cm_idx((block_size * block_size) as int, 0, pk_idx(block_offset as int, block_offset as int))]) == -1.0real,
{
    let ghost (sd, bsum, bo, k) = (total_dim as int, block_size_sum as int, block_offset as int, block_size as int);
    let ghost vd = dims_of(symm_dims@);
    let ghost cols = pk_len(sd);
    proof {
        assert(k * k <= k * k * (cols + 1)) by (nonlinear_arith)
            requires k >= 0, cols >= 0;
    }
    let rows = block_size * block_size;
    let ghost rr = rows as int;
    proof {
        assert(rr == k * k);
        assert forall|n: int| 0 <= n < symm_dims@.len() implies #[trigger] vd[n] >= 1 by {
            assert(maps@[n]@.len() == rows);
            assert(vd[n] == symm_dims@[n] as int);
        }
        lemma_sum_upto_monotone(vd, 0, symm_dims@.len() as int);
        assert(bsum <= sd);
        assert(maps_shape_ok(maps@, symm_dims@, rows as int));
    }
    let mut sum = maps_sum(maps, symm_dims, block_size_sum, total_dim, rows);
    proof {
        lemma_dim_fits_monotone(bo + k, sd);
        lemma_dim_fits_idx(sd, sd);
        lemma_pk_col_start_monotone(bsum, sd);
        assert(rr * cols == cols * rr) by (nonlinear_arith);
        assert(rr * (cols + 1) == rr * cols + rr) by (nonlinear_arith);
        assert(0 * block_size == 0) by (nonlinear_arith);
        assert(k * k >= 1) by (nonlinear_arith)
            requires k >= 1;
    }
    let block_current = sym_matrix_diag_block_map(block_offset, block_size);
    let ghost base = rvs(sum@);
    proof {
        assert forall|e: int, t: int| 0 <= e < rr && 0 <= t < pk_col_start(bsum) implies #[trigger] base[cm_idx(rr, e, t)] == 0.0real by {
            lemma_cm_idx_bound(rr, e, t, cols);
        }
        assert forall|e: int, t: int| 0 <= e < rr && 0 <= t < cols implies rv(#[trigger] sum@[cm_idx(rr, e, t)]) == base[cm_idx(rr, e, t)] by {
            lemma_cm_idx_bound(rr, e, t, cols);
        }
    }
    let mut row = 0;
    for c in 0..block_size
        invariant
            sd == total_dim as int,
            bsum == block_size_sum as int,
            bo == block_offset as int,
            k == block_size as int,
            rr == rows as int,
            rows == block_size * block_size,
            cols == pk_len(sd),
            k >= 1,
            bo + k <= bsum <= sd,
            dim_fits(sd),
            rr * cols <= usize::MAX,
            rr * cols == cols * rr,
            sum@.len() == rr * cols,
            base.len() == rr * cols,
            block_current@.len() == pk_len(k),
            forall|r2: int, c2: int| 0 <= r2 <= c2 < k ==> #[trigger] block_current@[pk_idx(r2, c2)] == pk_idx(bo + r2, bo + c2),
            row == c * block_size,
            forall|e: int, t: int| 0 <= e < rr && 0 <= t < pk_col_start(bsum) ==> #[trigger] base[cm_idx(rr, e, t)] == 0.0real,
            forall|e: int, z: Seq<real>| 0 <= e < row && z.len() == cols
                ==> #[trigger] row_dot_r(rvs(sum@), rr, e, z) == row_dot_r(base, rr, e, z) - sym_entry(z, bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)),
            forall|e: int, z: Seq<real>| row <= e < rr && z.len() == cols
                ==> #[trigger] row_dot_r(rvs(sum@), rr, e, z) == row_dot_r(base, rr, e, z),
            forall|e: int| 0 <= e < row ==> rv(#[trigger] sum@[cm_idx(rr, e, pk_idx(bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)))]) == -1.0real,
            forall|e: int, t: int| row <= e < rr && 0 <= t < cols ==> rv(#[trigger] sum@[cm_idx(rr, e, t)]) == base[cm_idx(rr, e, t)],
    {
        let ghost ci = c as int;
        proof {
            assert(c * block_size + block_size == (c + 1) * block_size) by (nonlinear_arith);
            assert((c + 1) * block_size <= block_size * block_size) by (nonlinear_arith)
                requires c + 1 <= block_size;
        }
        for r in 0..block_size
            invariant
                sd == total_dim as int,
                bsum == block_size_sum as int,
                bo == block_offset as int,
                k == block_size as int,
                rr == rows as int,
                rows == block_size * block_size,
                cols == pk_len(sd),
                ci == c as int,
                k >= 1,
                c < block_size,
                bo + k <= bsum <= sd,
                dim_fits(sd),
                rr * cols <= usize::MAX,
                rr * cols == cols * rr,
                sum@.len() == rr * cols,
                base.len() == rr * cols,
                block_current@.len() == pk_len(k),
                forall|r2: int, c2: int| 0 <= r2 <= c2 < k ==> #[trigger] block_current@[pk_idx(r2, c2)] == pk_idx(bo + r2, bo + c2),
                row == c * block_size + r,
                (c + 1) * block_size <= block_size * block_size,
                c * block_size + block_size == (c + 1) * block_size,
                forall|e: int, t: int| 0 <= e < rr && 0 <= t < pk_col_start(bsum) ==> #[trigger] base[cm_idx(rr, e, t)] == 0.0real,
                forall|e: int, z: Seq<real>| 0 <= e < row && z.len() == cols
                    ==> #[trigger] row_dot_r(rvs(sum@), rr, e, z) == row_dot_r(base, rr, e, z) - sym_entry(z, bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)),
                forall|e: int, z: Seq<real>| row <= e < rr && z.len() == cols
                    ==> #[trigger] row_dot_r(rvs(sum@), rr, e, z) == row_dot_r(base, rr, e, z),
                forall|e: int| 0 <= e < row ==> rv(#[trigger] sum@[cm_idx(rr, e, pk_idx(bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)))]) == -1.0real,
                forall|e: int, t: int| row <= e < rr && 0 <= t < cols ==> rv(#[trigger] sum@[cm_idx(rr, e, t)]) == base[cm_idx(rr, e, t)],
        {
            let ghost (ri, ei) = (r as int, row as int);
            let ghost (lo, hi) = if r <= c { (ri, ci) } else { (ci, ri) };
            proof {
                lemma_dim_fits_monotone(bo + k, sd);
                lemma_dim_fits_idx(bo + k, lo);
                lemma_dim_fits_idx(bo + k, hi);
                lemma_pk_idx_bound(lo, hi, k);
                lemma_pk_idx_bound(bo + lo, bo + hi, bsum);
                lemma_pk_col_start_monotone(bsum, sd);
                // the entry e = c k + r addresses rows r and c of the block
                lemma_lmi_row_of(k, ci, ri);
                assert(lmi_row_lo(k, ei) == lo);
                assert(lmi_row_hi(k, ei) == hi);
            }
            let packed = if r <= c {
                block_current[(c * (c + 1)) / 2 + r]
            } else {
                block_current[(r * (r + 1)) / 2 + c]
            };
            proof {
                assert(packed as int == pk_idx(bo + lo, bo + hi));
                lemma_cm_idx_bound(rr, ei, packed as int, cols);
            }
            let idx = packed * rows + row;
            let ghost prev = rvs(sum@);
            let ghost t0 = packed as int;
            sum[idx] = sum[idx] - 1.0;
            proof {
                assert(idx as int == cm_idx(rr, ei, t0));
                let cur = rvs(sum@);
                assert(cur =~= prev.update(cm_idx(rr, ei, t0), prev[cm_idx(rr, ei, t0)] + (-1.0real)));
                assert forall|e: int, z: Seq<real>| 0 <= e < rr && z.len() == cols
                    implies #[trigger] row_dot_r(cur, rr, e, z) == row_dot_r(prev, rr, e, z) + (if e == ei { (-1.0real) * z[t0] } else { 0.0real }) by {
                    lemma_row_dot_update(prev, cur, rr, ei, t0, -1.0real, e, z);
                }
                assert forall|e: int, z: Seq<real>| 0 <= e < row + 1 && z.len() == cols
                    implies #[trigger] row_dot_r(cur, rr, e, z) == row_dot_r(base, rr, e, z) - sym_entry(z, bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)) by {
                    if e == ei {
                        assert(row_dot_r(prev, rr, ei, z) == row_dot_r(base, rr, ei, z));
                        assert(sym_entry(z, bo + lo, bo + hi) == z[t0]);
                    }
                }
                assert forall|e: int| 0 <= e < row + 1 implies rv(#[trigger] sum@[cm_idx(rr, e, pk_idx(bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)))]) == -1.0real by {
                    if e == ei {
                        // entry (ei, t0) was untouched before this write and is zero in the base
                        assert(base[cm_idx(rr, ei, t0)] == 0.0real);
                        assert(prev[cm_idx(rr, ei, t0)] == base[cm_idx(rr, ei, t0)]);
                    } else {
                        lemma_lmi_row_bounds(k, e);
                        lemma_pk_idx_bound(bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e), sd);
                        lemma_cm_idx_bound(rr, e, pk_idx(bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)), cols);
                        lemma_cm_idx_injective(rr, e, pk_idx(bo + lmi_row_lo(k, e), bo + lmi_row_hi(k, e)), ei, t0);
                    }
                }
                assert forall|e: int, t: int| row + 1 <= e < rr && 0 <= t < cols implies rv(#[trigger] sum@[cm_idx(rr, e, t)]) == base[cm_idx(rr, e, t)] by {
                    lemma_cm_idx_bound(rr, e, t, cols);
                    lemma_cm_idx_injective(rr, e, t, ei, t0);
                }
            }
            row += 1;
        }
    }
    proof {
        lemma_lmi_row_of(k, 0, 0);
        assert(0 * k + 0 == 0);
        assert(rv(sum@[cm_idx(rr, 0, pk_idx(bo + lmi_row_lo(k, 0), bo + lmi_row_hi(k, 0)))]) == -1.0real);
    }
    sum
}

/// The standard-form objective: sum_n [embed(C_n, first block of n) - embed(C_n, second block of n)],
/// so that <obj, Z> = C_1(Z_1) + .. + C_n(Z_n).
fn objective_sum(objective: &MatrixSym, symm_dims: &Vec<usize>, block_size_sum: usize, total_dim: usize) -> (obj_sum: SymMatrix)
    requires
        objective@.len() == symm_dims@.len(),
        objective_shapes_ok(objective@, symm_dims@),
        dim_fits(total_dim as int),
        var_off(block_size_sum as int, dims_of(symm_dims@), symm_dims@.len() as int) <= total_dim,
    ensures
        obj_sum@.len() == pk_len(total_dim as int),
        forall|z: Seq<real>| z.len() == pk_len(total_dim as int)
            ==> #[trigger] dot_r(rvs(obj_sum@), z) == obj_r(objective@, block_size_sum as int, dims_of(symm_dims@), z),
{
    let ghost (sd, bsum, nv) = (total_dim as int, block_size_sum as int, objective@.len() as int);
    let ghost vd = dims_of(symm_dims@);
    let ghost cols = pk_len(sd);
    proof {
        lemma_dim_fits_idx(sd, sd);
        assert forall|n: int| 0 <= n < nv implies #[trigger] vd[n] >= 1 by {
            assert(part_shape(objective@[n]@, symm_dims@[n] as int));
            assert(vd[n] == symm_dims@[n] as int);
        }
        lemma_sum_upto_monotone(vd, 0, nv);
    }
    let total_size = (total_dim * (total_dim + 1)) / 2;
    let mut obj_sum: SymMatrix = vec![0.0; total_size];
    let mut symm_offset = 0;
    proof {
        assert forall|z: Seq<real>| z.len() == cols
            implies #[trigger] dot_r(rvs(obj_sum@), z) == sum_fn_r(|n2: int| obj_term_r(objective@, bsum, vd, z, n2), 0) by {
            lemma_dot_zero(rvs(obj_sum@), z);
        }
    }
    for n in 0..objective.len()
        invariant
            sd == total_dim as int,
            bsum == block_size_sum as int,
            nv == objective@.len() as int,
            vd == dims_of(symm_dims@),
            cols == pk_len(sd),
            total_size == cols,
            objective@.len() == symm_dims@.len(),
            forall|n2: int| 0 <= n2 < nv ==> part_shape((#[trigger] objective@[n2])@, vd[n2]) && vd[n2] >= 1,
            dim_fits(sd),
            var_off(bsum, vd, nv) <= sd,
            obj_sum@.len() == cols,
            symm_offset as int == 2 * sum_upto(vd, n as int),
            var_off(bsum, vd, n as int) <= var_off(bsum, vd, nv),
            forall|z: Seq<real>| z.len() == cols
                ==> #[trigger] dot_r(rvs(obj_sum@), z) == sum_fn_r(|n2: int| obj_term_r(objective@, bsum, vd, z, n2), n as nat),
    {
        let ghost ni = n as int;
        let dim = symm_dims[n];
        let ghost d = dim as int;
        let ghost st1 = bsum + symm_offset as int;
        let ghost st2 = st1 + d;
        proof {
            lemma_sum_upto_monotone(vd, ni + 1, nv);
            lemma_dim_fits_monotone(d, sd);
        }
        let obj_flatten = part_concat(&objective[n], dim);
        let db1 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim);
        let db2 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim);
        let ghost prev = rvs(obj_sum@);
        for t in 0..total_size
            invariant
                total_size == cols,
                obj_sum@.len() == cols,
                db1@.len() == cols,
                db2@.len() == cols,
                prev.len() == cols,
                forall|t2: int| 0 <= t2 < t ==> rv(#[trigger] obj_sum@[t2]) == prev[t2] + (rv(db1@[t2]) - rv(db2@[t2])),
                forall|t2: int| t <= t2 < cols ==> rv(#[trigger] obj_sum@[t2]) == prev[t2],
        {
            obj_sum[t] = obj_sum[t] + (db1[t] - db2[t]);
        }
        proof {
            let (v1, v2) = (rvs(db1@), rvs(db2@));
            let w = rvs(obj_flatten@);
            assert forall|z: Seq<real>| z.len() == cols
                implies #[trigger] dot_r(rvs(obj_sum@), z) == sum_fn_r(|n2: int| obj_term_r(objective@, bsum, vd, z, n2), (n + 1) as nat) by {
                lemma_dot_add_sub(rvs(obj_sum@), prev, v1, v2, z);
                assert forall|a: int, b: int| 0 <= a <= b < d implies v1[#[trigger] pk_idx(st1 + a, st1 + b)] == w[pk_idx(a, b)] by {
                    lemma_pk_idx_bound(st1 + a, st1 + b, sd);
                    lemma_pk_idx_bound(a, b, d);
                }
                assert forall|a: int, b: int| 0 <= a <= b < d implies v2[#[trigger] pk_idx(st2 + a, st2 + b)] == w[pk_idx(a, b)] by {
                    lemma_pk_idx_bound(st2 + a, st2 + b, sd);
                    lemma_pk_idx_bound(a, b, d);
                }
                assert forall|t: int| 0 <= t < cols && !is_block_idx(t, st1, d) implies #[trigger] v1[t] == 0.0real by {}
                assert forall|t: int| 0 <= t < cols && !is_block_idx(t, st2, d) implies #[trigger] v2[t] == 0.0real by {}
                lemma_embed_dot(v1, w, z, sd, st1, d);
                lemma_embed_dot(v2, w, z, sd, st2, d);
                lemma_dot_sub_right(w, blk_r(z, st1, d), blk_r(z, st2, d));
                assert(var_off(bsum, vd, ni) == st1);
                assert(obj_term_r(objective@, bsum, vd, z, ni) == dot_r(w, zvar_r(z, st1, d)));
            }
        }
        symm_offset += 2 * dim;
    }
    obj_sum
}

/// The upper off-block-diagonal positions (r, c) of the standard-form variable, listed block
/// by block: for every diagonal block, the rows of the block against every later column.
fn zero_positions(block_dims: &Vec<usize>, symm_dims: &Vec<usize>, total_dim: usize) -> (zeros: Vec<(usize, usize)>)
    requires
        sum_upto(block_sizes(dims_of(block_dims@), dims_of(symm_dims@)), (block_dims@.len() + 2 * symm_dims@.len()) as int) == total_dim,
        forall|j: int| 0 <= j < block_dims@.len() + 2 * symm_dims@.len()
            ==> #[trigger] block_sizes(dims_of(block_dims@), dims_of(symm_dims@))[j] >= 1,
        dim_fits(total_dim as int),
    ensures
        zeros_ok(zeros@, block_sizes(dims_of(block_dims@), dims_of(symm_dims@)), total_dim as int),
        zeros@.len() == zeros_len(block_sizes(dims_of(block_dims@), dims_of(symm_dims@)), total_dim as int),
{
    let ghost bs = block_sizes(dims_of(block_dims@), dims_of(symm_dims@));
    let ghost sd = total_dim as int;
    let ghost nb = bs.len() as int;
    let ghost pkf = |j: int| pk_len(bs[j]);
    proof {
        lemma_dim_fits_idx(sd, sd);
        assert forall|j: int| 0 <= j < nb implies #[trigger] bs[j] >= 0 by {}
        lemma_sum_upto_ge(bs, nb);
        lemma_pk_col_start_zero();
    }
    let nblocks = block_dims.len() + 2 * symm_dims.len();
    // the number of positions: the upper triangle minus the block triangles
    let total_size = (total_dim * (total_dim + 1)) / 2;
    let mut block_total = 0;
    for j in 0..nblocks
        invariant
            nblocks == block_dims@.len() + 2 * symm_dims@.len(),
            bs == block_sizes(dims_of(block_dims@), dims_of(symm_dims@)),
            nb == bs.len() as int,
            sd == total_dim as int,
            dim_fits(sd),
            sum_upto(bs, nb) == sd,
            forall|j2: int| 0 <= j2 < nb ==> #[trigger] bs[j2] >= 0,
            pkf == (|j2: int| pk_len(bs[j2])),
            block_total as int == sum_fn_int(pkf, j as nat),
            block_total <= total_size,
            total_size == pk_len(sd),
    {
        let k = if j < block_dims.len() { block_dims[j] } else { symm_dims[(j - block_dims.len()) / 2] };
        proof {
            assert(k as int == bs[j as int]);
            lemma_sum_upto_monotone(bs, 0, j as int);
            lemma_sum_upto_monotone(bs, j as int + 1, nb);
            lemma_dim_fits_idx(sd, k as int);
            lemma_block_triangles_le(bs, j as int + 1);
            lemma_pk_col_start_monotone(sum_upto(bs, j as int + 1), sd);
        }
        block_total += (k * (k + 1)) / 2;
    }
    let zeros_count = total_size - block_total;
    let mut zeros: Vec<(usize, usize)> = Vec::with_capacity(zeros_count);
    let mut start = 0;
    for j in 0..nblocks
        invariant
            nblocks == block_dims@.len() + 2 * symm_dims@.len(),
            bs == block_sizes(dims_of(block_dims@), dims_of(symm_dims@)),
            nb == bs.len() as int,
            sd == total_dim as int,
            dim_fits(sd),
            sum_upto(bs, nb) == sd,
            forall|j2: int| 0 <= j2 < nb ==> #[trigger] bs[j2] >= 0,
            pkf == (|j2: int| pk_len(bs[j2])),
            start as int == sum_upto(bs, j as int),
            start <= total_dim,
            zeros@.len() as int == pk_len(sd) - pk_len(sd - start) - sum_fn_int(pkf, j as nat),
            valid_zero_list(zeros@, sd),
            forall|q: int| 0 <= q < zeros@.len() ==> in_zero_region(bs, (#[trigger] zeros@[q]).0 as int, zeros@[q].1 as int) && zeros@[q].0 < start,
            forall|r: int, c: int| 0 <= r < c < sd && in_zero_region(bs, r, c) && r < start ==> #[trigger] listed(zeros@, r, c),
            forall|q1: int, q2: int| 0 <= q1 < zeros@.len() && 0 <= q2 < zeros@.len() && q1 != q2
                ==> #[trigger] zeros@[q1] != #[trigger] zeros@[q2],
    {
        let ghost ji = j as int;
        let k = if j < block_dims.len() { block_dims[j] } else { symm_dims[(j - block_dims.len()) / 2] };
        let ghost (s0, ki) = (start as int, k as int);
        proof {
            assert(ki == bs[ji]);
            lemma_sum_upto_monotone(bs, ji + 1, nb);
            lemma_pk_len_diff(sd - s0, ki);
            lemma_pk_col_start_monotone(0, sd - s0 - ki);
        }
        let ghost len0 = zeros@.len() as int;
        proof {
            assert(((start + k) as int - (s0 + ki)) * ki == 0) by (nonlinear_arith)
                requires (start + k) as int == s0 + ki;
        }
        for c in start + k..total_dim
            invariant
                sd == total_dim as int,
                nb == bs.len() as int,
                ji == j as int,
                j < nb,
                ki == k as int,
                ki == bs[ji],
                s0 == start as int,
                s0 == sum_upto(bs, ji),
                s0 + ki == sum_upto(bs, ji + 1),
                s0 + ki <= sd,
                zeros@.len() as int == len0 + (c - (start + k)) * ki,
                len0 >= 0,
                dim_fits(sd),
                valid_zero_list(zeros@, sd),
                forall|q: int| 0 <= q < len0 ==> in_zero_region(bs, (#[trigger] zeros@[q]).0 as int, zeros@[q].1 as int) && zeros@[q].0 < s0,
                forall|q: int| len0 <= q < zeros@.len() ==> s0 <= (#[trigger] zeros@[q]).0 < s0 + ki && s0 + ki <= zeros@[q].1 < c,
                forall|r: int, c2: int| 0 <= r < c2 < sd && in_zero_region(bs, r, c2) && r < s0 ==> #[trigger] listed(zeros@, r, c2),
                forall|r: int, c2: int| s0 <= r < s0 + ki && s0 + ki <= c2 < c ==> #[trigger] listed(zeros@, r, c2),
                forall|q1: int, q2: int| 0 <= q1 < zeros@.len() && 0 <= q2 < zeros@.len() && q1 != q2
                    ==> #[trigger] zeros@[q1] != #[trigger] zeros@[q2],
        {
            let ghost cc = c as int;
            let ghost len1 = zeros@.len() as int;
            proof {
                assert((cc + 1 - (s0 + ki)) * ki == (cc - (s0 + ki)) * ki + ki) by (nonlinear_arith);
                assert((cc - (s0 + ki)) * ki >= 0) by (nonlinear_arith)
                    requires cc >= s0 + ki, ki >= 0;
                assert((sd - (s0 + ki)) * ki >= (cc + 1 - (s0 + ki)) * ki) by (nonlinear_arith)
                    requires cc + 1 <= sd, ki >= 0;
            }
            for r in start..start + k
                invariant
                    sd == total_dim as int,
                    nb == bs.len() as int,
                    ji == j as int,
                    j < nb,
                    ki == k as int,
                    ki == bs[ji],
                    s0 == start as int,
                    s0 == sum_upto(bs, ji),
                    s0 + ki == sum_upto(bs, ji + 1),
                    cc == c as int,
                    s0 + ki <= cc < sd,
                    len0 >= 0,
                    len1 == len0 + (cc - (s0 + ki)) * ki,
                    zeros@.len() as int == len1 + (r - start),
                    valid_zero_list(zeros@, sd),
                    forall|q: int| 0 <= q < len0 ==> in_zero_region(bs, (#[trigger] zeros@[q]).0 as int, zeros@[q].1 as int) && zeros@[q].0 < s0,
                    forall|q: int| len0 <= q < len1 ==> s0 <= (#[trigger] zeros@[q]).0 < s0 + ki && s0 + ki <= zeros@[q].1 < cc,
                    forall|q: int| len1 <= q < zeros@.len() ==> s0 <= (#[trigger] zeros@[q]).0 < r && zeros@[q].1 == cc,
                    forall|r2: int, c2: int| 0 <= r2 < c2 < sd && in_zero_region(bs, r2, c2) && r2 < s0 ==> #[trigger] listed(zeros@, r2, c2),
                    forall|r2: int, c2: int| s0 <= r2 < s0 + ki && s0 + ki <= c2 < cc ==> #[trigger] listed(zeros@, r2, c2),
                    forall|r2: int| s0 <= r2 < r ==> #[trigger] listed(zeros@, r2, cc),
                    forall|q1: int, q2: int| 0 <= q1 < zeros@.len() && 0 <= q2 < zeros@.len() && q1 != q2
                        ==> #[trigger] zeros@[q1] != #[trigger] zeros@[q2],
            {
                let ghost prev = zeros@;
                zeros.push((r, c));
                proof {
                    let q0 = prev.len() as int;
                    assert(zeros@ == prev.push((r, c)));
                    assert(zeros@[q0] == (r, c));
                    assert(in_zero_region(bs, r as int, cc)) by {
                        assert(sum_upto(bs, ji) <= (r as int) < sum_upto(bs, ji + 1) <= cc);
                    }
                    assert forall|r2: int, c2: int| listed(prev, r2, c2) || (r2 == r as int && c2 == cc) implies #[trigger] listed(zeros@, r2, c2) by {
                        lemma_listed_push(prev, (r, c), r2, c2);
                    }
                    assert forall|q1: int, q2: int| 0 <= q1 < zeros@.len() && 0 <= q2 < zeros@.len() && q1 != q2
                        implies #[trigger] zeros@[q1] != #[trigger] zeros@[q2] by {
                        if q1 == q0 || q2 == q0 {
                            let q = if q1 == q0 { q2 } else { q1 };
                            assert(zeros@[q] == prev[q]);
                            if q < len0 {
                                assert(prev[q].0 < s0);
                            } else if q < len1 {
                                assert(prev[q].1 < cc);
                            } else {
                                assert(prev[q].0 < r);
                            }
                        } else {
                            assert(zeros@[q1] == prev[q1]);
                            assert(zeros@[q2] == prev[q2]);
                        }
                    }
                }
            }
        }
        proof {
            // after the block: every region pair with r < start + k is listed, exactly once
            assert forall|r: int, c2: int| 0 <= r < c2 < sd && in_zero_region(bs, r, c2) && r < s0 + ki
                implies #[trigger] listed(zeros@, r, c2) by {
                if r >= s0 {
                    let jr = choose|jr: int| 0 <= jr < bs.len() && #[trigger] sum_upto(bs, jr) <= r < sum_upto(bs, jr + 1) <= c2;
                    if jr < ji {
                        lemma_sum_upto_monotone(bs, jr + 1, ji);
                    } else if jr > ji {
                        lemma_sum_upto_monotone(bs, ji + 1, jr);
                    }
                    assert(jr == ji);
                }
            }
            assert((sd - (s0 + ki)) * ki == ki * (sd - s0 - ki)) by (nonlinear_arith);
        }
        start += k;
    }
    proof {
        assert(start == total_dim);
        assert(pk_len(0) == 0);
        assert(zeros.len() == zeros_count);
    }
    zeros
}

/// Stacks the rows of the blocks (each a column-major matrix with total_size columns together
/// with its right-hand side) into one column-major matrix and one right-hand side vector.
fn stack_blocks(blocks: &Vec<(Matrix, Vector)>, total_size: usize) -> (res: (Matrix, Vector))
    requires
        total_size >= 1,
        forall|b: int| 0 <= b < blocks@.len() ==> (#[trigger] blocks@[b]).0@.len() == blocks@[b].1@.len() * total_size,
        rows_upto(blocks@, blocks@.len() as int) * total_size <= usize::MAX,
    ensures
        res.0@.len() == rows_upto(blocks@, blocks@.len() as int) * total_size,
        res.1@.len() == rows_upto(blocks@, blocks@.len() as int),
        forall|b: int, e: int, t: int| 0 <= b < blocks@.len() && 0 <= e < blocks@[b].1@.len() && 0 <= t < total_size
            ==> rv(res.0@[cm_idx(rows_upto(blocks@, blocks@.len() as int), rows_upto(blocks@, b) + e, t)])
                == rv((#[trigger] blocks@[b]).0@[#[trigger] cm_idx(blocks@[b].1@.len() as int, e, t)]),
        forall|b: int, e: int| 0 <= b < blocks@.len() && 0 <= e < blocks@[b].1@.len()
            ==> rv(res.1@[rows_upto(blocks@, b) + e]) == rv(#[trigger] blocks@[b].1@[e]),
{
    let ghost nb = blocks@.len() as int;
    let ghost rr = rows_upto(blocks@, nb);
    let ghost cols = total_size as int;
    let mut constraint_rows = 0;
    for b in 0..blocks.len()
        invariant
            nb == blocks@.len() as int,
            rr == rows_upto(blocks@, nb),
            rr * cols <= usize::MAX,
            cols == total_size as int,
            cols >= 1,
            constraint_rows as int == rows_upto(blocks@, b as int),
    {
        proof {
            lemma_rows_upto_monotone(blocks@, 0, b as int);
            lemma_rows_upto_monotone(blocks@, b as int + 1, nb);
            assert(rows_upto(blocks@, b as int + 1) <= rr);
            assert(rr <= rr * cols) by (nonlinear_arith)
                requires rr >= 0, cols >= 1;
        }
        constraint_rows += blocks[b].1.len();
    }
    let mut mat: Matrix = Vec::with_capacity(total_size * constraint_rows);
    let mut vect: Vector = Vec::with_capacity(constraint_rows);
    proof {
        assert(0 * rr == 0) by (nonlinear_arith);
        assert(rr * cols == cols * rr) by (nonlinear_arith);
    }
    for col in 0..total_size
        invariant
            nb == blocks@.len() as int,
            rr == rows_upto(blocks@, nb),
            cols == total_size as int,
            rr * cols <= usize::MAX,
            rr * cols == cols * rr,
            constraint_rows as int == rr,
            forall|b: int| 0 <= b < nb ==> (#[trigger] blocks@[b]).0@.len() == blocks@[b].1@.len() * cols,
            mat@.len() == col * rr,
            forall|b: int, e: int, t: int| 0 <= b < nb && 0 <= e < blocks@[b].1@.len() && 0 <= t < col
                ==> rv(mat@[cm_idx(rr, rows_upto(blocks@, b) + e, t)]) == rv((#[trigger] blocks@[b]).0@[#[trigger] cm_idx(blocks@[b].1@.len() as int, e, t)]),
    {
        let ghost ci = col as int;
        proof {
            assert((ci + 1) * rr == ci * rr + rr) by (nonlinear_arith);
            assert((ci + 1) * rr <= cols * rr) by (nonlinear_arith)
                requires ci + 1 <= cols, rr >= 0;
        }
        for b in 0..blocks.len()
            invariant
                nb == blocks@.len() as int,
                rr == rows_upto(blocks@, nb),
                cols == total_size as int,
                ci == col as int,
                col < total_size,
                (ci + 1) * rr == ci * rr + rr,
                (ci + 1) * rr <= cols * rr,
                cols * rr <= usize::MAX,
                forall|b2: int| 0 <= b2 < nb ==> (#[trigger] blocks@[b2]).0@.len() == blocks@[b2].1@.len() * cols,
                mat@.len() == ci * rr + rows_upto(blocks@, b as int),
                forall|b2: int, e: int, t: int| 0 <= b2 < nb && 0 <= e < blocks@[b2].1@.len() && 0 <= t < ci
                    ==> rv(mat@[cm_idx(rr, rows_upto(blocks@, b2) + e, t)]) == rv((#[trigger] blocks@[b2]).0@[#[trigger] cm_idx(blocks@[b2].1@.len() as int, e, t)]),
                forall|b2: int, e: int| 0 <= b2 < b && 0 <= e < blocks@[b2].1@.len()
                    ==> rv(mat@[cm_idx(rr, rows_upto(blocks@, b2) + e, ci)]) == rv((#[trigger] blocks@[b2]).0@[#[trigger] cm_idx(blocks@[b2].1@.len() as int, e, ci)]),
        {
            let ghost bi = b as int;
            let row_dim = blocks[b].1.len();
            let ghost rd = row_dim as int;
            proof {
                lemma_rows_upto_monotone(blocks@, 0, bi);
                lemma_rows_upto_monotone(blocks@, bi + 1, nb);
                assert(rd <= rr);
                assert(ci * rd + rd == (ci + 1) * rd) by (nonlinear_arith);
                assert((ci + 1) * rd <= cols * rd) by (nonlinear_arith)
                    requires ci + 1 <= cols, rd >= 0;
                assert(cols * rd == rd * cols) by (nonlinear_arith);
                assert(rd * cols <= rr * cols) by (nonlinear_arith)
                    requires rd <= rr, cols >= 0;
                assert(ci * rr + rows_upto(blocks@, bi) + rd <= (ci + 1) * rr);
            }
            let ghost prev = mat@;
            mat.extend_from_slice(&blocks[b].0[col * row_dim..(col + 1) * row_dim]);
            proof {
                assert(mat@.len() == prev.len() + rd);
                assert forall|b2: int, e: int, t: int| 0 <= b2 < nb && 0 <= e < blocks@[b2].1@.len() && 0 <= t < ci
                    implies rv(mat@[cm_idx(rr, rows_upto(blocks@, b2) + e, t)]) == rv((#[trigger] blocks@[b2]).0@[#[trigger] cm_idx(blocks@[b2].1@.len() as int, e, t)]) by {
                    lemma_rows_upto_monotone(blocks@, 0, b2);
                    lemma_rows_upto_monotone(blocks@, b2 + 1, nb);
                    lemma_cm_idx_bound(rr, rows_upto(blocks@, b2) + e, t, ci);
                    let idx = cm_idx(rr, rows_upto(blocks@, b2) + e, t);
                    assert(idx < ci * rr);
                    assert(idx < prev.len());
                    assert(mat@[idx] == prev[idx]);
                }
                assert forall|b2: int, e: int| 0 <= b2 < b + 1 && 0 <= e < blocks@[b2].1@.len()
                    implies rv(mat@[cm_idx(rr, rows_upto(blocks@, b2) + e, ci)]) == rv((#[trigger] blocks@[b2]).0@[#[trigger] cm_idx(blocks@[b2].1@.len() as int, e, ci)]) by {
                    lemma_rows_upto_monotone(blocks@, 0, b2);
                    let idx = cm_idx(rr, rows_upto(blocks@, b2) + e, ci);
                    if b2 < bi {
                        lemma_rows_upto_monotone(blocks@, b2 + 1, bi);
                        assert(idx < prev.len());
                        assert(mat@[idx] == prev[idx]);
                    } else {
                        assert(idx == prev.len() + e);
                        assert(mat@[prev.len() + e] == blocks@[bi].0@[ci * rd + e]);
                    }
                }
            }
        }
    }
    for b in 0..blocks.len()
        invariant
            nb == blocks@.len() as int,
            rr == rows_upto(blocks@, nb),
            vect@.len() == rows_upto(blocks@, b as int),
            forall|b2: int, e: int| 0 <= b2 < b && 0 <= e < blocks@[b2].1@.len()
                ==> rv(vect@[rows_upto(blocks@, b2) + e]) == rv(#[trigger] blocks@[b2].1@[e]),
    {
        let ghost prev = vect@;
        proof {
            lemma_rows_upto_monotone(blocks@, b as int + 1, nb);
        }
        vect.extend_from_slice(&blocks[b].1);
        proof {
            assert forall|b2: int, e: int| 0 <= b2 < b + 1 && 0 <= e < blocks@[b2].1@.len()
                implies rv(vect@[rows_upto(blocks@, b2) + e]) == rv(#[trigger] blocks@[b2].1@[e]) by {
                lemma_rows_upto_monotone(blocks@, 0, b2);
                if b2 < b {
                    lemma_rows_upto_monotone(blocks@, b2 + 1, b as int);
                    assert(rows_upto(blocks@, b2) + e < prev.len());
                    assert(vect@[rows_upto(blocks@, b2) + e] == prev[rows_upto(blocks@, b2) + e]);
                }
            }
        }
    }
    proof {
        assert(cols * rr == rr * cols) by (nonlinear_arith);
    }
    (mat, vect)
}


// converting a user-given SDP to one in standard form (c, A, b)
// also stores the dimension of the final decision variable,
// the dimensions of any smaller input decision variables,
// and the dimensions of the block constraints

// takes a boolean flag "check_constraints" to decide whether to
// run Gaussian elimination on dense constraint matrix
// (contract: `std_form_single_ok` without LMIs, `std_form_lmi_ok` with LMIs)
pub fn sdp_to_standard(sdp: &SDP, check_constraints: bool)
    -> (res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>))
    requires
        sdp.lmi@.len() == 0 ==> sdp_single_input_ok(sdp)
            && (check_constraints && sdp.constraint.0@.len() == 1 ==> single_constraint_nonzero(sdp)),
        sdp.lmi@.len() >= 1 ==> sdp_lmi_input_ok(sdp),
    ensures
        sdp.lmi@.len() == 0 ==> std_form_single_ok(sdp, res, check_constraints),
        sdp.lmi@.len() >= 1 ==> std_form_lmi_ok(sdp, res, check_constraints),
{
    if sdp.lmi.is_empty() {
        standard_single(sdp, check_constraints)
    } else {
        standard_lmi(sdp, check_constraints)
    }
}

/// The conversion without LMIs: a single decision variable.
fn standard_single(sdp: &SDP, check_constraints: bool)
    -> (res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>))
    requires
        sdp.lmi@.len() == 0,
        sdp_single_input_ok(sdp),
        check_constraints && sdp.constraint.0@.len() == 1 ==> single_constraint_nonzero(sdp),
    ensures
        std_form_single_ok(sdp, res, check_constraints),
{
    {
        if sdp.objective.len() != 1 {
            abort("you must supply exactly one objective map since you have exactly one decision variable");
        }
        let obj = &sdp.objective[0];
        let ghost d = obj@.len() as int;
        if obj.len() == 0 {
            abort("you have not provided an objective function");
        }
        let total_dim = obj[obj.len() - 1].len();
        let ghost cols = pk_len(d);
        let obj_packed = part_concat(obj, total_dim);
        if sdp.constraint.0.is_empty() {
            let res = (obj_packed, (vec![], vec![]), vec![], total_dim, vec![], vec![]);
            proof {
                assert(0 * cols == 0) by (nonlinear_arith);
                assert(res.3 as int == d);
                assert(res.0@.len() == cols);
                assert(rvs(res.0@) == part_r(sdp.objective@[0]@, d));
                assert(res.1.0@.len() == 0);
                assert(res.2@.len() == 0);
            }
            res
        } else if sdp.constraint.0.len() == 1 {
            let constr = &sdp.constraint.0[0];
            let ghost m = sdp.constraint.1@.len() as int;
            let ghost dd = symmap_dim(constr@);
            proof {
                lemma_dim_fits_idx(d, d);
                lemma_pk_col_start_monotone(0, d);
                if m >= 1 {
                    assert(part_shape(constr@[0]@, d));
                }
                assert(dd == if m == 0 { 0int } else { d });
                lemma_dim_fits_monotone(dd, d);
                lemma_pk_col_start_monotone(dd, d);
                assert(m * pk_len(dd) <= m * cols) by (nonlinear_arith)
                    requires pk_len(dd) <= cols, m >= 0;
                assert(m * cols <= cols * (m + 1)) by (nonlinear_arith)
                    requires cols >= 0, m >= 0;
            }
            let mut constraints = matrixsym_flatten(constr);
            let ghost flat = constraints@;
            proof {
                assert(cols * m == m * cols) by (nonlinear_arith);
                assert(flat.len() == m * pk_len(dd));
                assert(m * pk_len(dd) == m * cols) by (nonlinear_arith)
                    requires m == 0 || pk_len(dd) == cols;
                lemma_single_rows(constr@, sdp.constraint.1@, d, flat);
            }
            if check_constraints {
                // make augmented constraint matrix full rank
                constraints.extend_from_slice(&sdp.constraint.1);
                let ghost aug = rvs(constraints@);
                proof {
                    assert(aug =~= rvs(flat) + rvs(sdp.constraint.1@));
                    assert(m * cols + m == m * (cols + 1)) by (nonlinear_arith);
                    lemma_mod_multiples_basic(m, cols + 1);
                    // some entry of the map or of the right-hand side is nonzero
                    if exists|e: int, i: int, j: int| 0 <= e < m && 0 <= j <= i < d
                        && rv(#[trigger] sdp.constraint.0@[0]@[e]@[i]@[j]) != 0.0real {
                        let (e, i, j) = choose|e: int, i: int, j: int| 0 <= e < m && 0 <= j <= i < d
                            && rv(#[trigger] sdp.constraint.0@[0]@[e]@[i]@[j]) != 0.0real;
                        lemma_pk_idx_bound(j, i, d);
                        lemma_cm_idx_bound(m, e, pk_idx(j, i), cols);
                        lemma_abs_max_entry(aug, cm_idx(m, e, pk_idx(j, i)));
                    } else {
                        let e = choose|e: int| 0 <= e < m && rv(#[trigger] sdp.constraint.1@[e]) != 0.0real;
                        lemma_abs_max_entry(aug, m * cols + e);
                    }
                }
                let mut constraints_red = linear_remove_redundant_sym(&mut constraints, total_dim);
                let ghost red = rvs(constraints_red@);
                let total_size = (total_dim * (total_dim + 1)) / 2;
                let rank = constraints_red.len() / (total_size + 1);
                proof {
                    let pivots = choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(aug, rvs(constraints@), red, d, pivots);
                    assert(red.len() == (cols + 1) * pivots.len());
                    assert(((cols + 1) * pivots.len()) / (cols + 1) == pivots.len()) by (nonlinear_arith)
                        requires cols + 1 >= 1;
                    assert(rank as int == pivots.len());
                    assert(cols * rank + rank == (cols + 1) * rank) by (nonlinear_arith);
                }
                let point = constraints_red.split_off(total_size * rank);
                let res = (obj_packed, (constraints_red, vec![]), point, total_dim, vec![], vec![]);
                proof {
                    assert(result_aug(res.1.0@, res.2@, rank as int) =~= red);
                    assert(reduced_system(aug, rvs(constraints@), red, d));
                    assert(reduced_from(aug, result_aug(res.1.0@, res.2@, rank as int), d));
                    assert(res.2@.len() == rank);
                    assert(res.3 as int == d);
                    assert(res.0@.len() == cols);
                    assert(rvs(res.0@) == part_r(sdp.objective@[0]@, d));
                    assert(res.1.0@.len() == (rank as int) * cols) by {
                        assert(cols * rank == rank * cols) by (nonlinear_arith);
                    }
                    assert(single_rows_ok(sdp.constraint.0@[0]@, sdp.constraint.1@, d, aug));
                }
                res
            } else {
                let point = sdp.constraint.1.clone();
                let res = (obj_packed, (constraints, vec![]), point, total_dim, vec![], vec![]);
                proof {
                    assert(result_aug(res.1.0@, res.2@, m) =~= rvs(flat) + rvs(sdp.constraint.1@));
                    assert(res.2@.len() == m);
                    assert(res.3 as int == d);
                    assert(res.0@.len() == cols);
                    assert(rvs(res.0@) == part_r(sdp.objective@[0]@, d));
                    assert(res.1.0@.len() == m * cols);
                    assert(single_rows_ok(sdp.constraint.0@[0]@, sdp.constraint.1@, d, rvs(flat) + rvs(sdp.constraint.1@)));
                }
                res
            }
        } else {
            abort("you have one decision variable but have supplied constraint maps for more than one decision variable");
        }
    }
}

/// The conversion with LMIs: block-diagonal standard-form variable.
fn standard_lmi(sdp: &SDP, check_constraints: bool)
    -> (res: (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>))
    requires
        sdp.lmi@.len() >= 1,
        sdp_lmi_input_ok(sdp),
    ensures
        std_form_lmi_ok(sdp, res, check_constraints),
{
    {
        let lmis = &sdp.lmi;
        let ghost nv = sdp.objective@.len() as int;
        let ghost bd = bdims_r(lmis@);
        let ghost vd = vdims_r(lmis@, nv);
        let ghost s = lmis@.len() as int;
        let ghost bsum = block_sum(bd);
        let ghost sd = std_dim(bd, vd);
        let ghost cols = pk_len(sd);
        let ghost m = sdp.constraint.1@.len() as int;
        let ghost rr = lmi_rows(bd) + m;
        proof {
            assert forall|i: int| 0 <= i < s implies #[trigger] bd[i] >= 1 by {}
            assert forall|n: int| 0 <= n < nv implies #[trigger] vd[n] >= 1 by {}
            lemma_sum_upto_monotone(bd, 0, s);
            lemma_sum_upto_monotone(vd, 0, nv);
            lemma_sum_upto_ge(bd, s);
            lemma_sq_sum_upto_monotone(bd, 0, s);
            assert(bsum <= sd);
            assert(sd >= 1);
            lemma_dim_fits_idx(sd, sd);
            lemma_pk_col_start_monotone(0, sd);
            assert((cols + 1) * (rr + 1) == cols * (rr + 1) + (rr + 1)) by (nonlinear_arith);
            assert(cols * (rr + 1) == cols * rr + cols) by (nonlinear_arith);
            assert(cols * rr == rr * cols) by (nonlinear_arith);
            assert(rr * (cols + 1) <= (rr + 1) * (cols + 1)) by (nonlinear_arith)
                requires rr >= 0, cols >= 0;
            assert((rr + 1) * (cols + 1) == (cols + 1) * (rr + 1)) by (nonlinear_arith);
        }
        proof {
            assert(n_maps(lmis@) == nv);
        }
        let (block_dims, symm_dims, block_size_sum, symm_size_sum) = lmi_dims(lmis);
        let total_dim = block_size_sum + 2 * symm_size_sum;
        let total_size = (total_dim * (total_dim + 1)) / 2;
        proof {
            lemma_lmi_rows_ge(bd, s);
            assert(cols >= 1) by {
                lemma_pk_col_start_step(sd - 1);
                lemma_pk_col_start_monotone(0, sd - 1);
                assert(pk_col_start(sd) == pk_col_start(sd - 1) + sd);
            }
            assert(rr + 1 <= cols * (rr + 1)) by (nonlinear_arith)
                requires cols >= 1, rr + 1 >= 0;
            lemma_block_sizes_sum(bd, vd, nv);
        }
        let mut blocks: Vec<(Matrix, Vector)> = Vec::with_capacity(lmis.len() + 1);
        let mut block_offset = 0;
        for b in 0..lmis.len()
            invariant
                sdp_lmi_input_ok(sdp),
                lmis@ == sdp.lmi@,
                nv == sdp.objective@.len() as int,
                bd == bdims_r(lmis@),
                vd == vdims_r(lmis@, nv),
                s == lmis@.len() as int,
                bsum == block_sum(bd),
                sd == std_dim(bd, vd),
                cols == pk_len(sd),
                m == sdp.constraint.1@.len() as int,
                rr == lmi_rows(bd) + m,
                total_dim as int == sd,
                total_size == cols,
                block_size_sum as int == bsum,
                dims_of(block_dims@) == bd,
                dims_of(symm_dims@) == vd,
                block_dims@.len() == s,
                symm_dims@.len() == nv,
                forall|i: int| 0 <= i < s ==> #[trigger] bd[i] >= 0,
                forall|n: int| 0 <= n < nv ==> #[trigger] vd[n] >= 0,
                cols * (rr + 1) <= usize::MAX,
                rr * (cols + 1) <= usize::MAX,
                cols * rr == rr * cols,
                blocks@.len() == b,
                block_offset as int == block_off(bd, b as int),
                forall|i: int| 0 <= i < b ==> (#[trigger] blocks@[i]).1@ == lmis@[i].1@,
                forall|i: int| 0 <= i < b ==> (#[trigger] blocks@[i]).0@.len() == bd[i] * bd[i] * cols,
                forall|i: int, e: int, z: Seq<real>| 0 <= i < b && 0 <= e < bd[i] * bd[i] && z.len() == cols
                    ==> #[trigger] row_dot_r(rvs(blocks@[i].0@), bd[i] * bd[i], e, z)
                        == maps_row_r(lmis@[i].0@, bsum, vd, e, z) - sym_entry(z, block_off(bd, i) + lmi_row_lo(bd[i], e), block_off(bd, i) + lmi_row_hi(bd[i], e)),
                b >= 1 ==> rv(blocks@[0].0@[cm_idx(bd[0] * bd[0], 0, pk_idx(block_off(bd, 0), block_off(bd, 0)))]) == -1.0real,
        {
            let ghost bi = b as int;
            let block_size = block_dims[b];
            let ghost k = block_size as int;
            proof {
                assert(k == bd[bi]);
                lemma_sum_upto_monotone(bd, bi + 1, s);
                lemma_sq_sum_upto_monotone(bd, 0, bi);
                lemma_sq_sum_upto_monotone(bd, bi + 1, s);
                assert(sq_sum_upto(bd, bi + 1) == sq_sum_upto(bd, bi) + bd[bi] * bd[bi]);
                assert(k * k == bd[bi] * bd[bi]);
                assert(k * k <= sq_sum_upto(bd, bi + 1));
                assert(k * k * (cols + 1) <= rr * (cols + 1)) by (nonlinear_arith)
                    requires k * k <= rr, cols >= 0;
                assert(lmi_shapes_ok(sdp.lmi@, nv));
                assert(sdp.lmi@[bi].0@.len() == nv);
                let maps_b = lmis@[bi].0@;
                assert(maps_b.len() == symm_dims@.len());
                assert forall|n: int| 0 <= n < maps_b.len() implies symmap_shape((#[trigger] maps_b[n])@, symm_dims@[n] as int) by {
                    assert(symmap_shape(sdp.lmi@[bi].0@[n]@, vd[n]));
                    assert(vd[n] == symm_dims@[n] as int);
                }
                assert forall|n: int| 0 <= n < maps_b.len() implies (#[trigger] maps_b[n])@.len() == k * k by {
                    assert(sdp.lmi@[bi].0@[n]@.len() == bd[bi] * bd[bi]);
                    assert(k * k == bd[bi] * bd[bi]);
                }
                assert forall|n: int| 0 <= n < maps_b.len() implies symm_dims@[n] >= 1 by {
                    assert(vd[n] >= 1);
                    assert(vd[n] == symm_dims@[n] as int);
                }
                assert(maps_shape_ok(maps_b, symm_dims@, k * k)) by {
                    assert forall|n: int| 0 <= n < maps_b.len() implies symmap_shape((#[trigger] maps_b[n])@, symm_dims@[n] as int)
                        && maps_b[n]@.len() == k * k && symm_dims@[n] >= 1 by {
                        assert(symmap_shape(maps_b[n]@, symm_dims@[n] as int));
                        assert(maps_b[n]@.len() == k * k);
                        assert(symm_dims@[n] >= 1);
                    }
                }
                assert(k * k == block_size * block_size);
            }
            let sum = lmi_block(&lmis[b].0, &symm_dims, block_size_sum, total_dim, block_offset, block_size);
            blocks.push((sum, lmis[b].1.clone()));
            block_offset += block_size;
        }
        proof {
            assert(n_maps(lmis@) == nv);
            lemma_objective_shapes(sdp, symm_dims@);
        }
        let obj_sum = objective_sum(&sdp.objective, &symm_dims, block_size_sum, total_dim);
        let constr_len = sdp.constraint.1.len();
        let constr_sum: Matrix = if sdp.constraint.0.is_empty() {
            if !sdp.constraint.1.is_empty() {
                abort("empty constraints but nonempty vector");
            }
            proof {
                assert(0 * cols == 0) by (nonlinear_arith);
            }
            vec![]
        } else {
            proof {
                assert(m * (cols + 1) <= rr * (cols + 1)) by (nonlinear_arith)
                    requires m <= rr, cols >= 0;
                assert(sdp.constraint.0@.len() == nv);
                assert forall|n: int| 0 <= n < sdp.constraint.0@.len() implies symmap_shape((#[trigger] sdp.constraint.0@[n])@, symm_dims@[n] as int)
                    && sdp.constraint.0@[n]@.len() == constr_len && symm_dims@[n] >= 1 by {
                    assert(symmap_shape(sdp.constraint.0@[n]@, vd[n]));
                    assert(vd[n] >= 1);
                    assert(vd[n] == symm_dims@[n] as int);
                }
                assert(maps_shape_ok(sdp.constraint.0@, symm_dims@, constr_len as int));
            }
            maps_sum(&sdp.constraint.0, &symm_dims, block_size_sum, total_dim, constr_len)
        };
        proof {
            assert(constr_sum@.len() == m * cols);
            assert forall|e: int, z: Seq<real>| 0 <= e < m && z.len() == cols
                implies #[trigger] row_dot_r(rvs(constr_sum@), m, e, z) == maps_row_r(sdp.constraint.0@, bsum, vd, e, z) by {}
        }
        blocks.push((constr_sum, sdp.constraint.1.clone()));
        proof {
            lemma_rows_upto_sq(blocks@, bd, s);
            assert(rows_upto(blocks@, s + 1) == rr);
            lemma_lmi_rows_ge(bd, s);
        }
        proof {
            let bs = block_sizes(bd, vd);
            assert forall|j: int| 0 <= j < s + 2 * nv implies #[trigger] bs[j] >= 1 by {
                if j >= s {
                    assert(bs[j] == vd[(j - s) / 2]);
                }
            }
        }
        let zeros = zero_positions(&block_dims, &symm_dims, total_dim);
        let (mut mat, mut vect) = stack_blocks(&blocks, total_size);
        let ghost aug0 = rvs(mat@) + rvs(vect@);
        proof {
            // the stacked rows are the block rows, with the right-hand sides in the last column
            assert(aug0.len() == rr * (cols + 1)) by {
                assert(rr * cols + rr == rr * (cols + 1)) by (nonlinear_arith);
            }
            assert(rvs(mat@).len() == rr * cols);
            assert(rvs(vect@).len() == rr);
            assert forall|i: int, e: int, z: Seq<real>| 0 <= i < s && 0 <= e < bd[i] * bd[i] && z.len() == cols
                implies #[trigger] row_dot_r(aug0, rr, row_off(bd, i) + e, z)
                    == maps_row_r(lmis@[i].0@, bsum, vd, e, z) - sym_entry(z, block_off(bd, i) + lmi_row_lo(bd[i], e), block_off(bd, i) + lmi_row_hi(bd[i], e)) by {
                lemma_rows_upto_sq(blocks@, bd, i);
                lemma_sq_sum_upto_monotone(bd, 0, i);
                lemma_sq_sum_upto_monotone(bd, i + 1, s);
                assert(sq_sum_upto(bd, i + 1) == sq_sum_upto(bd, i) + bd[i] * bd[i]);
                let brows = blocks@[i].1@.len() as int;
                assert(brows == bd[i] * bd[i]);
                let roff = row_off(bd, i);
                assert(rows_upto(blocks@, i) == roff);
                assert(rows_upto(blocks@, blocks@.len() as int) == rr);
                assert(blocks@[i].0@.len() == brows * cols);
                assert(brows * cols == cols * brows) by (nonlinear_arith);
                assert forall|t: int| 0 <= t < cols implies rvs(mat@)[#[trigger] cm_idx(rr, roff + e, t)] == rvs(blocks@[i].0@)[cm_idx(brows, e, t)] by {
                    lemma_cm_idx_bound(rr, roff + e, t, cols);
                    lemma_cm_idx_bound(brows, e, t, cols);
                    let idx = cm_idx(rr, roff + e, t);
                    let idx2 = cm_idx(brows, e, t);
                    assert(idx < mat@.len());
                    assert(idx2 < blocks@[i].0@.len());
                    assert(rv(mat@[cm_idx(rows_upto(blocks@, blocks@.len() as int), rows_upto(blocks@, i) + e, t)]) == rv(blocks@[i].0@[cm_idx(blocks@[i].1@.len() as int, e, t)]));
                    assert(rvs(mat@)[idx] == rv(mat@[idx]));
                    assert(rvs(blocks@[i].0@)[idx2] == rv(blocks@[i].0@[idx2]));
                }
                lemma_stacked_row(aug0, rvs(mat@), rvs(blocks@[i].0@), rr, brows, roff, e, z);
            }
            assert forall|i: int, e: int| 0 <= i < s && 0 <= e < bd[i] * bd[i]
                implies #[trigger] cm_entry(aug0, rr, row_off(bd, i) + e, cols) == rv(lmis@[i].1@[e]) by {
                lemma_rows_upto_sq(blocks@, bd, i);
                lemma_sq_sum_upto_monotone(bd, 0, i);
                lemma_sq_sum_upto_monotone(bd, i + 1, s);
                assert(sq_sum_upto(bd, i + 1) == sq_sum_upto(bd, i) + bd[i] * bd[i]);
                let rho = row_off(bd, i) + e;
                let idx = cm_idx(rr, rho, cols);
                assert(idx == cols * rr + rho);
                assert(idx - rvs(mat@).len() == rho);
                assert(aug0[idx] == rvs(vect@)[rho]);
                assert(rv(vect@[rows_upto(blocks@, i) + e]) == rv(blocks@[i].1@[e]));
            }
            assert forall|e: int, z: Seq<real>| 0 <= e < m && z.len() == cols
                implies #[trigger] row_dot_r(aug0, rr, lmi_rows(bd) + e, z) == maps_row_r(sdp.constraint.0@, bsum, vd, e, z) by {
                let brows = blocks@[s].1@.len() as int;
                assert(brows == m);
                let roff = lmi_rows(bd);
                assert(rows_upto(blocks@, s) == roff);
                assert(rows_upto(blocks@, blocks@.len() as int) == rr);
                assert(blocks@[s].0@.len() == brows * cols);
                assert(brows * cols == cols * brows) by (nonlinear_arith);
                assert forall|t: int| 0 <= t < cols implies rvs(mat@)[#[trigger] cm_idx(rr, roff + e, t)] == rvs(blocks@[s].0@)[cm_idx(brows, e, t)] by {
                    lemma_cm_idx_bound(rr, roff + e, t, cols);
                    lemma_cm_idx_bound(brows, e, t, cols);
                    let idx = cm_idx(rr, roff + e, t);
                    let idx2 = cm_idx(brows, e, t);
                    assert(idx < mat@.len());
                    assert(idx2 < blocks@[s].0@.len());
                    assert(rv(mat@[cm_idx(rows_upto(blocks@, blocks@.len() as int), rows_upto(blocks@, s) + e, t)]) == rv(blocks@[s].0@[cm_idx(blocks@[s].1@.len() as int, e, t)]));
                    assert(rvs(mat@)[idx] == rv(mat@[idx]));
                    assert(rvs(blocks@[s].0@)[idx2] == rv(blocks@[s].0@[idx2]));
                }
                lemma_stacked_row(aug0, rvs(mat@), rvs(blocks@[s].0@), rr, brows, roff, e, z);
            }
            assert forall|e: int| 0 <= e < m implies #[trigger] cm_entry(aug0, rr, lmi_rows(bd) + e, cols) == rv(sdp.constraint.1@[e]) by {
                let rho = lmi_rows(bd) + e;
                let idx = cm_idx(rr, rho, cols);
                assert(idx == cols * rr + rho);
                assert(idx - rvs(mat@).len() == rho);
                assert(aug0[idx] == rvs(vect@)[rho]);
                assert(rv(vect@[rows_upto(blocks@, s) + e]) == rv(blocks@[s].1@[e]));
            }
            assert(std_rows_ok(lmis@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug0));
            // the first LMI row holds the entry -1 of its own block: the system is nonzero
            lemma_rows_upto_sq(blocks@, bd, 0);
            lemma_sum_upto_monotone(bd, 0, 0);
            lemma_pk_col_start_zero();
            assert(block_off(bd, 0) == 0);
            assert(row_off(bd, 0) == 0);
            assert(aug0[cm_idx(rr, 0, 0)] == -1.0real) by {
                assert(blocks@[0].1@.len() as int == bd[0] * bd[0]);
                assert(bd[0] * bd[0] >= 1) by (nonlinear_arith)
                    requires bd[0] >= 1;
                assert(rows_upto(blocks@, blocks@.len() as int) == rr);
                assert(pk_idx(0, 0) == 0);
                assert(rows_upto(blocks@, 0) == 0);
                assert(rv(mat@[cm_idx(rr, rows_upto(blocks@, 0) + 0, 0)]) == rv(blocks@[0].0@[cm_idx(blocks@[0].1@.len() as int, 0, 0)]));
                assert(rv(blocks@[0].0@[cm_idx(bd[0] * bd[0], 0, pk_idx(block_off(bd, 0), block_off(bd, 0)))]) == -1.0real);
                lemma_cm_idx_bound(rr, 0, 0, cols);
                assert(aug0[cm_idx(rr, 0, 0)] == rvs(mat@)[cm_idx(rr, 0, 0)]);
            }
        }
        if check_constraints {
            // make augmented constraint matrix full rank
            mat.extend_from_slice(&vect);
            let ghost aug = rvs(mat@);
            proof {
                assert(aug =~= aug0);
                assert(cols * rr + rr == rr * (cols + 1)) by (nonlinear_arith);
                lemma_mod_multiples_basic(rr, cols + 1);
                assert(aug.len() == rr * cols + rr);
                assert(rr >= 1);
                assert(0 * rr == 0);
                assert(cm_idx(rr, 0, 0) == 0);
                assert(aug[cm_idx(rr, 0, 0)] == -1.0real);
                lemma_abs_max_entry(aug, cm_idx(rr, 0, 0));
            }
            let mut constraints_red = linear_remove_redundant_sym(&mut mat, total_dim);
            let ghost red = rvs(constraints_red@);
            let rank = constraints_red.len() / (total_size + 1);
            proof {
                let pivots = choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(aug, rvs(mat@), red, sd, pivots);
                assert(red.len() == (cols + 1) * pivots.len());
                assert(((cols + 1) * pivots.len()) / (cols + 1) == pivots.len()) by (nonlinear_arith)
                    requires cols + 1 >= 1;
                assert(rank as int == pivots.len());
                assert(cols * rank + rank == (cols + 1) * rank) by (nonlinear_arith);
            }
            let mut point = constraints_red.split_off(total_size * rank);
            let ghost head = point@;
            point.append(&mut vec![0.0; zeros.len()]);
            let res = (obj_sum, (constraints_red, zeros), point, total_dim, symm_dims, block_dims);
            proof {
                assert(res.2@.subrange(0, rank as int) =~= head);
                assert(result_aug(res.1.0@, res.2@, rank as int) =~= red);
                assert(reduced_system(aug, rvs(mat@), red, sd));
                assert(reduced_from(aug, result_aug(res.1.0@, res.2@, rank as int), sd));
                assert(std_rows_ok(lmis@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug));
                let nz = res.1.1@.len() as int;
                assert(res.2@.len() as int - nz == rank as int);
                assert(res.3 as int == sd);
                assert(dims_of(res.4@) == vd);
                assert(dims_of(res.5@) == bd);
                assert(res.0@.len() == cols);
                assert(forall|z: Seq<real>| z.len() == cols ==> #[trigger] dot_r(rvs(res.0@), z) == obj_r(sdp.objective@, bsum, vd, z));
                assert(zeros_ok(res.1.1@, block_sizes(bd, vd), sd));
                assert(nz == zeros_len(block_sizes(bd, vd), sd));
                assert(res.1.0@.len() == (rank as int) * cols) by {
                    assert(cols * rank == rank * cols) by (nonlinear_arith);
                }
                assert(forall|q: int| rank <= q < res.2@.len() ==> rv(#[trigger] res.2@[q]) == 0.0real);
            }
            res
        } else {
            let ghost head = vect@;
            vect.append(&mut vec![0.0; zeros.len()]);
            let res = (obj_sum, (mat, zeros), vect, total_dim, symm_dims, block_dims);
            proof {
                assert(res.2@.subrange(0, rr) =~= head);
                assert(result_aug(res.1.0@, res.2@, rr) =~= aug0);
                let nz = res.1.1@.len() as int;
                assert(res.2@.len() as int - nz == rr);
                assert(res.3 as int == sd);
                assert(dims_of(res.4@) == vd);
                assert(dims_of(res.5@) == bd);
                assert(res.0@.len() == cols);
                assert(forall|z: Seq<real>| z.len() == cols ==> #[trigger] dot_r(rvs(res.0@), z) == obj_r(sdp.objective@, bsum, vd, z));
                assert(zeros_ok(res.1.1@, block_sizes(bd, vd), sd));
                assert(nz == zeros_len(block_sizes(bd, vd), sd));
                assert(res.1.0@.len() == rr * cols);
                assert(forall|q: int| rr <= q < res.2@.len() ==> rv(#[trigger] res.2@[q]) == 0.0real);
                assert(std_rows_ok(lmis@, sdp.constraint.0@, sdp.constraint.1@, bd, vd, aug0));
            }
            res
        }
    }
}


} // verus!

verus! {

// make an instance of MatrixSym with specified input dimension and size of output vector (i.e, number of constraints)
// (row k is the packed symmetric matrix stored at entries[k * pk_len(sym_dim) ..], in partitioned form)
pub fn mk_symmap(sym_dim: usize, output_size: usize, entries: Vector) -> (result: MatrixSym)
    requires
        dim_fits(sym_dim as int),
        entries@.len() >= output_size * pk_len(sym_dim as int),
    ensures
        result@.len() == output_size,
        symmap_shape(result@, sym_dim as int),
        forall|k: int, n: int, j: int| 0 <= k < output_size && 0 <= j <= n < sym_dim
            ==> rv(#[trigger] result@[k]@[n]@[j]) == rv(entries@[k * pk_len(sym_dim as int) + pk_idx(j, n)]),
{
    let mut result: MatrixSym = Vec::with_capacity(output_size);
    let mut start = 0;
    let ghost (sd, tot) = (sym_dim as int, pk_len(sym_dim as int));
    proof {
        lemma_dim_fits_idx(sd, sd);
        assert(entries.len() >= output_size * tot);
        assert(0 * tot == 0) by (nonlinear_arith);
    }
    let total = (sym_dim * (sym_dim + 1)) / 2;
    for _k in 0..output_size
        invariant
            sd == sym_dim as int,
            tot == pk_len(sd),
            total == tot,
            dim_fits(sd),
            entries@.len() >= output_size * tot,
            start == _k * tot,
            result@.len() == _k,
            symmap_shape(result@, sd),
            forall|k2: int, n: int, j: int| 0 <= k2 < _k && 0 <= j <= n < sd
                ==> rv(#[trigger] result@[k2]@[n]@[j]) == rv(entries@[k2 * tot + pk_idx(j, n)]),
    {
        proof {
            assert((_k + 1) * tot == _k * tot + tot) by (nonlinear_arith);
            assert((_k + 1) * tot <= output_size * tot) by (nonlinear_arith)
                requires _k + 1 <= output_size, tot >= 0;
            lemma_pk_col_start_zero();
        }
        let mut v: Vec<Vector> = Vec::with_capacity(sym_dim);
        let mut diff = 0;
        for n in 0..sym_dim
            invariant
                sd == sym_dim as int,
                tot == pk_len(sd),
                dim_fits(sd),
                entries@.len() >= output_size * tot,
                _k < output_size,
                start == _k * tot,
                start as int + tot <= output_size * tot,
                diff == pk_col_start(n as int),
                part_shape(v@, n as int),
                forall|n2: int, j: int| 0 <= j <= n2 < n ==> rv(#[trigger] v@[n2]@[j]) == rv(entries@[start + pk_idx(j, n2)]),
        {
            proof {
                lemma_dim_fits_idx(sd, n as int);
                lemma_pk_idx_bound(n as int, n as int, sd);
                lemma_pk_col_start_step(n as int);
                assert(entries.len() >= output_size * tot);
            }
            let offset = start + diff;
            v.push(slice_to_vec(&entries[offset..=offset + n]));
            diff += n + 1;
        }
        result.push(v);
        proof {
            assert(entries.len() >= output_size * tot);
        }
        start += total;
    }
    result
}

// zero MatrixSym
pub fn zero_symmap(sym_dim: usize, output_size: usize) -> (result: MatrixSym)
    ensures
        result@.len() == output_size,
        symmap_shape(result@, sym_dim as int),
        forall|k: int, n: int, j: int| 0 <= k < output_size && 0 <= j <= n < sym_dim
            ==> rv(#[trigger] result@[k]@[n]@[j]) == 0.0real,
{
    let mut result: MatrixSym = Vec::with_capacity(output_size);
    for _k in 0..output_size
        invariant
            result@.len() == _k,
            symmap_shape(result@, sym_dim as int),
            forall|k2: int, n: int, j: int| 0 <= k2 < _k && 0 <= j <= n < sym_dim
                ==> rv(#[trigger] result@[k2]@[n]@[j]) == 0.0real,
    {
        let mut v: Vec<Vector> = Vec::with_capacity(sym_dim);
        for n in 0..sym_dim
            invariant
                part_shape(v@, n as int),
                forall|n2: int, j: int| 0 <= j <= n2 < n ==> rv(#[trigger] v@[n2]@[j]) == 0.0real,
        {
            v.push(vec![0.0; n + 1]);
        }
        result.push(v);
    }
    result
}

/// Row n * dim + i of the identity MatrixSym picks entry (min(n, i), max(n, i)): the map Z -> vec(Z).
pub open spec fn ident_symmap_entry(n: int, i: int, c: int, r: int) -> real {
    if c == (if n <= i { i } else { n }) && r == (if n <= i { n } else { i }) { 1.0real } else { 0.0real }
}

// identity MatrixSym
pub fn ident_symmap(dim: usize) -> (result: MatrixSym)
    requires
        dim * dim <= usize::MAX,
    ensures
        result@.len() == dim * dim,
        symmap_shape(result@, dim as int),
        forall|n: int, i: int, c: int, r: int| 0 <= n < dim && 0 <= i < dim && 0 <= r <= c < dim
            ==> rv(#[trigger] result@[n * dim + i]@[c]@[r]) == ident_symmap_entry(n, i, c, r),
{
    let mut result: MatrixSym = Vec::new();
    let ghost d = dim as int;
    for n in 0..dim
        invariant
            d == dim as int,
            dim * dim <= usize::MAX,
            result@.len() == n * d,
            symmap_shape(result@, d),
            forall|n2: int, i: int, c: int, r: int| 0 <= n2 < n && 0 <= i < d && 0 <= r <= c < d
                ==> rv(#[trigger] result@[n2 * d + i]@[c]@[r]) == ident_symmap_entry(n2, i, c, r),
    {
        proof {
            assert((n + 1) * d == n * d + d) by (nonlinear_arith);
            assert((n + 1) * d <= d * d) by (nonlinear_arith)
                requires n + 1 <= d;
        }
        for i in 0..n
            invariant
                n < dim,
                d == dim as int,
                (n + 1) * d == n * d + d,
                (n + 1) * d <= d * d,
                dim * dim <= usize::MAX,
                result@.len() == n * d + i,
                symmap_shape(result@, d),
                forall|n2: int, i2: int, c: int, r: int| 0 <= n2 < n && 0 <= i2 < d && 0 <= r <= c < d
                    ==> rv(#[trigger] result@[n2 * d + i2]@[c]@[r]) == ident_symmap_entry(n2, i2, c, r),
                forall|i2: int, c: int, r: int| 0 <= i2 < i && 0 <= r <= c < d
                    ==> rv(#[trigger] result@[n * d + i2]@[c]@[r]) == ident_symmap_entry(n as int, i2, c, r),
        {
            let ghost prev = result@;
            result.push(sym_matrix_proj(dim, i, n));
            proof {
                assert forall|n2: int, i2: int, c: int, r: int| 0 <= n2 < n && 0 <= i2 < d && 0 <= r <= c < d
                    implies rv(#[trigger] result@[n2 * d + i2]@[c]@[r]) == ident_symmap_entry(n2, i2, c, r) by {
                    assert(n2 * d + d <= n * d) by (nonlinear_arith)
                        requires n2 + 1 <= n, d >= 0;
                    assert(result@[n2 * d + i2] == prev[n2 * d + i2]);
                }
                assert forall|i2: int, c: int, r: int| 0 <= i2 < i + 1 && 0 <= r <= c < d
                    implies rv(#[trigger] result@[n * d + i2]@[c]@[r]) == ident_symmap_entry(n as int, i2, c, r) by {
                    if i2 < i {
                        assert(result@[n * d + i2] == prev[n * d + i2]);
                    }
                }
            }
        }
        for i in n..dim
            invariant
                n < dim,
                n <= i,
                d == dim as int,
                (n + 1) * d == n * d + d,
                (n + 1) * d <= d * d,
                dim * dim <= usize::MAX,
                result@.len() == n * d + i,
                symmap_shape(result@, d),
                forall|n2: int, i2: int, c: int, r: int| 0 <= n2 < n && 0 <= i2 < d && 0 <= r <= c < d
                    ==> rv(#[trigger] result@[n2 * d + i2]@[c]@[r]) == ident_symmap_entry(n2, i2, c, r),
                forall|i2: int, c: int, r: int| 0 <= i2 < i && 0 <= r <= c < d
                    ==> rv(#[trigger] result@[n * d + i2]@[c]@[r]) == ident_symmap_entry(n as int, i2, c, r),
        {
            let ghost prev = result@;
            result.push(sym_matrix_proj(dim, n, i));
            proof {
                assert forall|n2: int, i2: int, c: int, r: int| 0 <= n2 < n && 0 <= i2 < d && 0 <= r <= c < d
                    implies rv(#[trigger] result@[n2 * d + i2]@[c]@[r]) == ident_symmap_entry(n2, i2, c, r) by {
                    assert(n2 * d + d <= n * d) by (nonlinear_arith)
                        requires n2 + 1 <= n, d >= 0;
                    assert(result@[n2 * d + i2] == prev[n2 * d + i2]);
                }
                assert forall|i2: int, c: int, r: int| 0 <= i2 < i + 1 && 0 <= r <= c < d
                    implies rv(#[trigger] result@[n * d + i2]@[c]@[r]) == ident_symmap_entry(n as int, i2, c, r) by {
                    if i2 < i {
                        assert(result@[n * d + i2] == prev[n * d + i2]);
                    }
                }
            }
        }
        proof {
            assert forall|n2: int, i: int, c: int, r: int| 0 <= n2 < n + 1 && 0 <= i < d && 0 <= r <= c < d
                implies rv(#[trigger] result@[n2 * d + i]@[c]@[r]) == ident_symmap_entry(n2, i, c, r) by {
                if n2 < n {
                    assert(n2 * d + d <= n * d) by (nonlinear_arith)
                        requires n2 + 1 <= n, d >= 0;
                }
            }
        }
    }
    result
}

} // verus!
