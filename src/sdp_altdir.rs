use crate::lin_alg::jacobi::*;
use crate::sdp::*;

// maximum number of iterations to run
const TOTAL_STEPS: usize = 5000;

// the alternating direction dual augmented Lagrangian method
pub fn sdpad(sdp : &SDP) -> (Vec<SymMatrix>, f64) {
    let (mut obj, (mat_red, zeros), point, var_dim, symm_dims, block_dims) = sdp_to_standard(sdp);
    let (c_len, block_size_sum) : (usize, usize) = (mat_red.first().unwrap_or(&vec![]).len(), block_dims.iter().sum());
    // adjust objective function so that it acts via the Frobenius product instead of packed dot product
    let obj_frob = scale_off_diag(0.5, var_dim, &mut obj);
    let constr_mat = (&mat_red, &zeros);
    // form the Gram matrix for the linear constraints (excluding the zero constraints)
    let gram_small = constraint_gram(var_dim, &mat_red);
    let (mut it_stag, mut it_pinf, mut it_dinf) = (0, 0, 0);
    // tuning parameters
    let (mut penalty, pen_min, pen_max, pen_fact, step) = (5.0, 1e-4, 1e4, 0.5, 1.6);
    let (stag1, stag2, stag3, pen_count, pen_ratio1, pen_ratio2) : (i32, i32, i32, i32, f64, f64) = (20, 150, 300, 50, 1.0, 1.0);
    let mut i_ratio : f64;
    let mut dual_y : Vector;
    let var_dim_tot = (var_dim * (var_dim + 1)) / 2;
    let (mut prim_z, mut dual_s) : (SymMatrix, SymMatrix) = (ident_sym_mat(var_dim), vec![0.0; var_dim_tot]);
    let (mut pinf, mut dinf, mut gap) : (f64, f64, f64);
    let mut delta : f64;
    let mut best = f64::INFINITY;
    let (mut inverse_y, mut dual_sfull, mut dual_sfull_temp) : (Vector, SymMatrix, SymMatrix) =
	(Vec::with_capacity(c_len), Vec::with_capacity(var_dim), Vec::with_capacity(var_dim));
    let mut diff : SymMatrix;
    let (mut symm_offset, mut result) : (usize, Vec<SymMatrix>) = (0, Vec::new());
    let (mut action, mut lin_comb) : (Vector, SymMatrix);
    let (mut prim_val, mut dual_val) : (f64, f64);

    // begin main computation
    let mut count = 0;
    loop {
	count += 1;

	action = constraint_action(&constr_mat, &prim_z);
	inverse_y.clear();
	for (i, a) in constraint_action(&constr_mat, &vect_subt(obj_frob, &dual_s)).iter().enumerate() {
	    inverse_y.push(penalty * (point[i] - action[i]) + a);
	}
	dual_y = if point.is_empty() {
	    vec![]
	} else {
	    pos_def_solver_upperleft(&gram_small, &mut inverse_y, zeros.len())
	};
	lin_comb = constraint_lin_comb(var_dim, &constr_mat, &dual_y);
	diff = vect_subt(obj_frob, &lin_comb);
	dual_sfull.clear();
	dual_sfull_temp.clear();
	for i in 0..var_dim_tot {
	    let val = diff[i] - penalty * prim_z[i];
	    dual_sfull.push(val);
	    dual_sfull_temp.push(val);
	}
	let (egnvals, egnvects) : (Vector, Matrix) = nonnegeigendecomp(&mut dual_sfull_temp, var_dim);
	if egnvals.is_empty() {
	    dual_s = vec![0.0; var_dim_tot];
	} else {
	    dual_s = diag_scale_gram(&egnvals, &egnvects)
	}

	for i in 0..var_dim_tot {
	    prim_z[i] = (1.0 - step) * prim_z[i] + (step / penalty) * (dual_s[i] - dual_sfull[i]);
	}
	
	pinf = euclid_distance(&constraint_action(&constr_mat, &prim_z), &point) / (1.0 + euclid_norm(&point));
	dinf = frob_dist_sym(&diff, &dual_s, var_dim) / (1.0 + sym_mat_one_norm(obj_frob, var_dim));

	(prim_val, dual_val) = (frob_prod_sym(obj_frob, &prim_z, var_dim), dot_prod(&point, &dual_y));
	gap = (dual_val - prim_val).abs() / (1.0 + dual_val.abs() + prim_val.abs());
	delta = pinf.max(dinf).max(gap);
	if delta < TOL &&
	    psd_block_check(&prim_z, &block_dims.iter().copied().
				scan(0, |start, dim| {
				    let block = (*start, dim);
				    *start += dim;
				    Some(block)
				}).chain(
				    symm_dims.iter().copied().
					scan(0, |start, dim| {
					    let block =	[(block_size_sum + *start, dim), (block_size_sum + *start + dim, dim)];
					    *start += 2 * dim;
					    Some(block)
					}).flatten()).collect()) {
	if block_size_sum == 0 {
	    result.push(prim_z);
	} else {
	    for d in symm_dims {
		result.push(vect_subt(&sym_matrix_diag_block(block_size_sum + symm_offset, d, &prim_z),
				      &sym_matrix_diag_block(block_size_sum + symm_offset + d, d, &prim_z)));
		symm_offset += 2 * d
	    }
	}
	println!("\nsdpad terminated with desired accuracy");
	return (result, prim_val);
    }
    if delta > best {
	it_stag += 1;
    } else {
	best = delta;
	it_stag = 0;
    }
    if ((it_stag > stag1 && delta < 1e-5) || (it_stag > stag2 && delta < 1e-4) || (it_stag > stag3 && delta < 1e-3)) &&
	psd_block_check(&prim_z,
			    &block_dims.iter().copied().
			    scan(0, |start, dim| {
				let block = (*start, dim);
				*start += dim;
				Some(block)
			    }).chain(
				symm_dims.iter().copied().
				    scan(0, |start, dim| {
					let block = [(block_size_sum + *start, dim), (block_size_sum + *start + dim, dim)];
					*start += 2 * dim;
					Some(block)
				    }).flatten()).collect()) {
	    if block_size_sum == 0 {
		result.push(prim_z);
	    } else {
		for d in symm_dims {
		    result.push(vect_subt(&sym_matrix_diag_block(block_size_sum + symm_offset, d, &prim_z),
					  &sym_matrix_diag_block(block_size_sum + symm_offset + d, d, &prim_z)));
		    symm_offset += 2 * d
		}
	    }
	    println!("\nsdpad terminated due to stagnation but with reasonable accuracy");
	    return (result, prim_val);
	}
    
	if count == TOTAL_STEPS {
	    if block_size_sum == 0 {
		result.push(prim_z);
	    } else {
		for d in symm_dims {
		    result.push(vect_subt(&sym_matrix_diag_block(block_size_sum + symm_offset, d, &prim_z),
					  &sym_matrix_diag_block(block_size_sum + symm_offset + d, d, &prim_z)));
		    symm_offset += 2 * d
		}
	    }
	    println!("\nsdpad terminated after {} iterations", TOTAL_STEPS);
	    println!("solution quality: pinf={:.3e} dinf={:.3e} gap={:.3e}", pinf, dinf, gap);
	    return (result, prim_val);
	}

	i_ratio = pinf / dinf;
	if !(i_ratio > pen_ratio1) {
	    it_pinf += 1;
	    it_dinf = 0;
	    if it_pinf >= pen_count {
		penalty = (pen_fact * penalty).max(pen_min);
		it_pinf = 0;
	    }
	} else if i_ratio > pen_ratio2 {
	    it_dinf += 1;
	    it_pinf = 0;
	    if it_dinf >= pen_count {
		penalty = (penalty / pen_fact).min(pen_max);
		it_dinf = 0;
	    }
	}
    }
}

// testing

#[cfg(test)]
#[path = "sdp_tests.rs"]
mod tests;
