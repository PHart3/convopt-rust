use vstd::prelude::*;
pub use crate::lin_alg::sym_matrix::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::lemma_pk_idx_injective;
use vstd::slice::slice_to_vec;

// linear map on symmetric matrices that is designed for user input
type MatrixSym = Vec<Vec<Vector>>;

// semidefinite programming problem
pub struct SDP {
    objective : MatrixSym,
    constraint : (Vec<MatrixSym>, Vector),
    lmi : Vec<(Vec<MatrixSym>, Matrix)>
}

impl SDP {   
    pub fn new(
	objective : MatrixSym,
	constraint : (Vec<MatrixSym>, Vector),
	lmi : Vec<(Vec<MatrixSym>, Matrix)>
    ) -> Self {
	Self { objective, constraint, lmi }
    }
    pub fn objective(&self) -> &MatrixSym {
        &self.objective
    }
    pub fn constraint(&self) -> &(Vec<MatrixSym>, Vector) {
        &self.constraint
    }
    pub fn lmi(&self) -> &Vec<(Vec<MatrixSym>, Matrix)> {
        &self.lmi
    }
}

verus! {

broadcast use crate::trusted::group_real_model;

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
        mat@.len() * (symmap_dim(mat@) * (symmap_dim(mat@) + 1)) <= usize::MAX,
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
        assert(dc * (dc + 1) >= 0) by (nonlinear_arith)
            requires dc >= 0;
        assert(pk_len(dc) <= dc * (dc + 1));
        assert(rows * pk_len(dc) <= rows * (dc * (dc + 1))) by (nonlinear_arith)
            requires pk_len(dc) <= dc * (dc + 1), rows >= 0;
    }
    let mut result: Matrix = Vec::with_capacity(mat.len() * (dim_c * (dim_c + 1)) / 2);
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

} // verus!

// converting a user-given SDP to one in standard form (c, A, b)
// also stores the dimension of the final decision variable,
// the dimensions of any smaller input decision variables,
// and the dimensions of the block constraints

// takes a boolean flag "check_constraints" to decide whether to
// run Gaussian elimination on dense constraint matrix
pub fn sdp_to_standard(sdp : &SDP, check_constraints : bool) ->
    (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>)
{
    if sdp.lmi.is_empty() {
	assert!(sdp.objective.len() == 1,
		"you must supply exactly one objective map since you have exactly one decision variable");
	if sdp.constraint.0.is_empty() {
	    let obj = &sdp.objective[0];
	    (obj.concat(), (vec![], vec![]), vec![], obj.last().expect("you have not provided an objective function").len(),
	     vec![], vec![])
	} else if sdp.constraint.0.len() == 1 {
	    let obj = &sdp.objective[0];
	    let total_dim = obj.last().expect("you have not provided an objective function").len();
	    if check_constraints {
		// make augmented constraint matrix full rank
		let mut constraints = matrixsym_flatten(&(sdp.constraint.0)[0]);
		constraints.extend_from_slice(&sdp.constraint.1);
		let mut constraints_red = linear_remove_redundant_sym(&mut constraints, total_dim);
		let rank = constraints_red.len() / (((total_dim * (total_dim + 1)) / 2) + 1);
		let point = constraints_red.split_off(((total_dim * (total_dim + 1)) / 2) * rank);

		(obj.concat(), (constraints_red, vec![]), point, total_dim, vec![], vec![])
	    } else {
		let (constraints, point) = (matrixsym_flatten(&(sdp.constraint.0)[0]), sdp.constraint.1.clone());
		(obj.concat(), (constraints, vec![]), point, total_dim, vec![], vec![])
	    }
	} else {
	    panic!("you have one decision variable but have supplied constraint maps for more than one decision variable");
	}
    } else {
	let lmis = &sdp.lmi;
	let (mut block_dims, mut symm_dims) = (Vec::new(), Vec::new());
	let mut first = true;
	let mut sequence_len = 0;
	let (mut block_size, mut symm_size);
	let (mut block_size_sum, mut symm_size_sum) = (0, 0);
	let mut map_len_old : Option<usize> = None;
	for (maps, mat) in lmis {
	    block_size = mat.len().isqrt();
	    assert!(block_size > 0 && block_size * block_size == mat.len(),
		    "the constant matrix for each LMI must be nonempty and square");
	    sequence_len += 1;
	    block_dims.push(block_size);
	    block_size_sum += block_size;
	    if first {
		map_len_old = Some(maps.len());
		for map in maps {		    
		    symm_size = map[0].last().unwrap_or_else(|| {
			panic!("first row of some matrix in affine combination {} is empty", sequence_len) }).len();
		    symm_dims.push(symm_size);
		    symm_size_sum += symm_size; 
		}
	    } else {
		assert_eq!(maps.len(), map_len_old.expect("never initialized"),
			   "each LMI must have the same number of decision variables");
		map_len_old = Some(maps.len());
	    }
	    first = false;
	}
	let total_dim = block_size_sum + 2 * symm_size_sum;
	let total_size = (total_dim * (total_dim + 1)) / 2;
	
	let mut blocks : Vec<(Matrix, Vector)> = Vec::with_capacity(sequence_len + 1);
	let mut sum : Matrix;
	let (mut symm_offset, mut dim);
	let (mut map_flatten, mut block_current);
	let (mut block_offset, mut b) = (0, 0);
	let mut row;
	for (maps, mat) in lmis {
	    symm_offset = 0;
	    block_size = block_dims[b];
	    sum = zero_mat(total_size, block_size * block_size);
	    for (n, map) in maps.iter().enumerate() {
		if block_size * block_size != map.len() {
		    println!("block size = {} and rows = {}", block_size, map.len());
		    panic!("the size of map {} in LMI {} does not match the size of the constant matrix", n, b);
		}
		dim = symm_dims[n];
		map_flatten = matrixsym_flatten(map);
		sum = matrix_add_diff(&sum,
				      &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim),
				      &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim));
		symm_offset += 2 * dim;
	    }
	    block_current = sym_matrix_diag_block_map(block_offset, block_size);
	    
	    row = 0;
	    for c in 0..block_size {
		for r in 0..block_size {
		    let packed = if r <= c {
			block_current[(c * (c + 1)) / 2 + r]
		    } else {
			block_current[(r * (r + 1)) / 2 + c]
		    };
		    sum[packed * block_size * block_size + row] -= 1.0;
		    row += 1;
		}
	    }
	    blocks.push((sum, mat.clone()));
	    block_offset += block_size;
	    b += 1;
	}

	let constr_len = sdp.constraint.1.len();
	let mut obj_sum : SymMatrix = vec![0.0; total_size];
	let mut constr_sum : Matrix;
	let mut obj_flatten : SymMatrix;
	symm_offset = 0;
	if sdp.constraint.0.is_empty() {
	    assert!(sdp.constraint.1.is_empty(), "empty constraints but nonempty vector");
	    constr_sum = vec![];
	    for (n, obj) in sdp.objective.iter().enumerate() {
		dim = symm_dims[n];

		obj_flatten = obj.concat();
		let db1 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim);
		let db2 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim);
		for ((s, v1), v2) in obj_sum.iter_mut().zip(db1.iter()).zip(db2.iter()) {
		    *s += *v1 - *v2;
		}
		symm_offset += 2 * dim;
	    }
	} else {
	    constr_sum = zero_mat(total_size, constr_len);
	    for (n, (obj, constr)) in (sdp.objective).iter().zip(sdp.constraint.0.iter()).enumerate() {
		dim = symm_dims[n];

		obj_flatten = obj.concat();
		let db1 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim);
		let db2 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim);
		for ((s, v1), v2) in obj_sum.iter_mut().zip(db1.iter()).zip(db2.iter()) {
		    *s += *v1 - *v2;
		}

		map_flatten = matrixsym_flatten(constr);
		constr_sum = matrix_add_diff(&constr_sum,
					     &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim),
					     &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim));

		symm_offset += 2 * dim;
	    }
	}
	blocks.push((constr_sum, sdp.constraint.1.clone()));

	// compute zero regions of constraint matrix
	let block_total = block_dims.iter().fold(0, |acc, k| acc + (k * (k + 1)) / 2);
	let symm_total = symm_dims.iter().fold(0, |acc, k| acc + (k * (k + 1)) / 2);
	let zeros_len = total_size - (block_total + 2 * symm_total);
	let mut zeros : Vec<(usize, usize)> = Vec::with_capacity(zeros_len);
	let (mut start, mut count) = (0, 0);
	for k in block_dims.iter().chain(symm_dims.iter()) {
	    for _ in 0..2 {
		for c in start + k..total_dim {
		    for r in start..start + k {
			zeros.push((r, c));
		    }
		}
		start += k;
		if count < sequence_len {
		    break;
		}
	    }
	    count += 1;
	}
	assert_eq!(start, total_dim);
	assert_eq!(zeros.len(), zeros_len);

	// compute augmented constraint matrix by stacking blocks
	let constraint_rows = blocks.iter().map(|(_, v)| v.len()).sum();
	let mut constraints : (Matrix, Vector) =
	    (Vec::with_capacity(total_size * constraint_rows), Vec::with_capacity(constraint_rows));
	for col in 0..total_size {
	    for (mat, vect) in &blocks {
		let row_dim = vect.len();
		constraints.0.extend_from_slice(&mat[col * row_dim..(col + 1) * row_dim]);
	    }
	}
	for (_, vect) in &blocks {
	    constraints.1.extend_from_slice(vect);
	}

	if check_constraints {
	    // make augmented constraint matrix full rank
	    constraints.0.extend_from_slice(&constraints.1);
	    let mut constraints_red = linear_remove_redundant_sym(&mut constraints.0, total_dim);
	    let rank = constraints_red.len() / (total_size + 1);
	    let mut point = constraints_red.split_off(total_size * rank);
	    point.append(&mut vec![0.0; zeros.len()]);

	    (obj_sum, (constraints_red, zeros), point, total_dim, symm_dims, block_dims)
	} else {
	    constraints.1.append(&mut vec![0.0; zeros.len()]);

	    (obj_sum, (constraints.0, zeros), constraints.1, total_dim, symm_dims, block_dims)
	}
    }
}

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
