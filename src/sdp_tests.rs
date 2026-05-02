// test harness for sdp solver

use crate::sdp::*;

use std::any::Any;

const TOL_TEST : f64 = 1e-5;

// testing if two sdp solutions are the same
fn sdp_equality(sdp: &SDP, sol1 : &(Vec<SymMatrix>, f64), sol2 : &(Option<Vec<SymMatrix>>, f64)) -> bool {
    if !float_equality(sol1.1, sol2.1, 1e-9) {
	println!("returned objective values are different");
	return false
    }
    
    let mut count = 0;
    
    // assuming that decision variable solutions are unique
    if let Some(dec_vars) = &sol2.0 {
	if sol1.0.len() != dec_vars.len() {
	    println!("number of returned decision variables must be the same");
	    return false
	}
	for (sm1, sm2) in sol1.0.iter().zip(dec_vars.iter()) {
	    if !vector_equality(sm1, sm2, TOL_TEST) {
		println!("different solutions for variable {}", count);
		println!("sol1={:#?} and sol2={:#?}", sm1, sm2);
		return false
	    }
	    count += 1;
	}
    }
    // if not, then just check feasibility of returned solution
    else {
	if sol1.0.len() != sdp.objective().len() {
	    println!("you have returned the wrong number of decision variables");
	    return false;
	}
	// check that each decision variable is psd
	for (sm1, obj) in sol1.0.iter().zip(sdp.objective().iter()) {
	    if !(negeigendecomp(&mut sm1.clone(), obj.len()).0.is_empty()) {
		println!("sol {} is not psd", count);
		return false
	    }
	    count += 1;
	}
	// check linear constraints
	let (maps_by_var, rhs) = &sdp.constraint();
	let mut lhs;
	if !maps_by_var.is_empty() {
            for j in 0..rhs.len() {
		lhs = 0.0;
		for (sm1, maps) in sol1.0.iter().zip(maps_by_var.iter()) {
                    lhs += dot_prod(&maps[j].concat(), sm1);
		}
		if (lhs - rhs[j]).abs() > 1e-6_f64.max(1e-6 * rhs[j].abs()) {
		    println!("linear constraint {} is not satisfied", j);
                    return false;
		}
            }
	}
	// check LMI constraints
	count = 0;
	for (maps, constant_mat) in sdp.lmi() {
            let block_dim = constant_mat.len();
	    let total = block_dim * block_dim;
            let mut lhs = vec![0.0; total];
	    for (sm1, map) in sol1.0.iter().zip(maps.iter()) {
		for i in 0..total {
		    lhs[i] += dot_prod(&map[i].concat(), sm1);
		}
	    }
	    let mut lhs_sym : SymMatrix = Vec::new();
	    for c in 0..block_dim {
		for r in 0..=c {
		    lhs_sym.push(lhs[c * block_dim + r] - constant_mat[c][r]);
		}
	    }
	    if !(negeigendecomp(&mut lhs_sym, block_dim).0.is_empty()) {
		println!("LMI {} is not satisfied", count);
		return false
	    }
	    count += 1;
	}
    }
    true
}

fn sdp_equality_err(sdp: &SDP,
		    result : &Result<(Vec<SymMatrix>, f64), Box<dyn Any + Send>>, reference : &(Option<Vec<SymMatrix>>, f64)) -> bool {
    match result {
	Err(e) => {
            if let Some(msg) = e.downcast_ref::<&str>() {
                println!("Caught panic: {}", msg);
            }
	    return false;
        }
	Ok(sol1) => sdp_equality(sdp, &sol1, reference)
    }
}

// start of tests

use crate::sdp_altdir::*;
use std::panic;

#[test]

fn test_sdp() {
    
    let (mut result, mut reference);
    //let mut result_pr;

    // a bunch of toy problems

    let sdp_test_1 = SDP::new(
	// X1 and X2 are 2x2
	// min trace(X1) - 2 * X1_12 + 2 * trace (X2)
	// s.t. trace(X1) + trace(X2) = 4, X1_22 = 1, X2_22 = 2
	//      [X1_11  X1_12; X1_12  1] >= 0, [X2_11  X2_12; X2_12  2] >= 0
	vec![sym_matrix_add(&sym_matrix_trace(2), &sym_matrix_scale(-2.0, &sym_matrix_proj(2, 0, 1))),
		       sym_matrix_scale(2.0, &sym_matrix_trace(2))],
	(vec![vec![sym_matrix_trace(2), sym_matrix_proj(2, 1, 1), sym_matrix_zero(2)],
			 vec![sym_matrix_trace(2), sym_matrix_zero(2), sym_matrix_proj(2, 1, 1)]], vec![4.0, 1.0, 2.0]),
	vec![(vec![mk_symmap(2, 4, vec![1.0, 0.0, 0.0,
						     0.0, 1.0, 0.0,
						     0.0, 1.0, 0.0,
						     0.0, 0.0, 0.0]),
				zero_symmap(2, 4)],
			   mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0])),
			  (vec![zero_symmap(2, 4),
				mk_symmap(2, 4, vec![1.0, 0.0, 0.0,
						     0.0, 1.0, 0.0,
						     0.0, 1.0, 0.0,
						     0.0, 0.0, 0.0])],
			   mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -2.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_1) });
    reference = (Some(vec![vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 2.0]]), 4.0);
    assert!(sdp_equality_err(&sdp_test_1, &result, &reference), "test 1 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 1: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_2 = SDP::new(
	// X1 and X2 are 2x2
	// min trace(X1) - 2 * X1_12 + 2 * trace(X2) - X2_12
	// s.t. trace(X1) + trace(X2) = 4, X1_22 = 1, X2_22 = 2
	//      [X1_11  X1_12; X1_12  1] >= 0, [X2_11  X2_12; X2_12  2] >= 0
	vec![
        sym_matrix_add(&sym_matrix_trace(2), &sym_matrix_scale(-2.0, &sym_matrix_proj(2, 0, 1))),
        sym_matrix_add(&sym_matrix_scale(2.0, &sym_matrix_trace(2)), &sym_matrix_scale(-1.0, &sym_matrix_proj(2, 0, 1)))],
	(vec![vec![sym_matrix_trace(2), sym_matrix_proj(2, 1, 1), sym_matrix_zero(2)],
			  vec![sym_matrix_trace(2), sym_matrix_zero(2), sym_matrix_proj(2, 1, 1)]],
		     vec![4.0, 1.0, 2.0]),
	vec![
            (vec![
                mk_symmap(2, 4, vec![
		    1.0, 0.0, 0.0,
		    0.0, 1.0, 0.0,
		    0.0, 1.0, 0.0,
		    0.0, 0.0, 0.0]),
                zero_symmap(2, 4)],
             mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0])),
            (vec![
                zero_symmap(2, 4),
                mk_symmap(2, 4, vec![
		    1.0, 0.0, 0.0,
		    0.0, 1.0, 0.0,
		    0.0, 1.0, 0.0,
		    0.0, 0.0, 0.0])],
             mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -2.0]))]
    );	
    result = panic::catch_unwind(|| { sdpad(&sdp_test_2) });
    reference = (Some(vec![vec![0.882675711869543, 0.939507164296746, 1.0],
			   vec![0.117324288130457, 0.484405383387852, 2.0]]),
		 3.75390244643155);
    assert!(sdp_equality_err(&sdp_test_2, &result, &reference), "test 2 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 2: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_3 = SDP::new(
	// X1 and X2 are 2x2
	// min trace(X1) - 2 * X1_12 + 2 * trace(X2) + 0.7 * X2_12
	// s.t. trace(X1) + trace(X2) = 4, X1_22 = 1, X2_22 = 2
	//      [X1_11  X1_12; X1_12  1] >= 0, [X2_11  X2_12; X2_12  2] >= 0
	vec![
            sym_matrix_add(&sym_matrix_trace(2), &sym_matrix_scale(-2.0, &sym_matrix_proj(2, 0, 1))),
            sym_matrix_add(&sym_matrix_scale(2.0, &sym_matrix_trace(2)), &sym_matrix_scale(0.7, &sym_matrix_proj(2, 0, 1)))],
	(
	    vec![
		vec![sym_matrix_trace(2), sym_matrix_proj(2, 1, 1), sym_matrix_zero(2)],
		vec![sym_matrix_trace(2), sym_matrix_zero(2), sym_matrix_proj(2, 1, 1)]],
	    vec![4.0, 1.0, 2.0]),
	vec![(
            vec![mk_symmap(2, 4,
			   vec![1.0, 0.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.0]),
		 zero_symmap(2, 4)],
            mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0])), (
            vec![zero_symmap(2, 4),
		 mk_symmap(2, 4,
			   vec![1.0, 0.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.0])],
            mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -2.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_3) });
    reference = (Some(vec![vec![0.94061055454, 0.96985079000, 1.0],
			   vec![0.05938944546, -0.34464313618, 2.0]]),
		 3.87843767087);
    assert!(sdp_equality_err(&sdp_test_3, &result, &reference), "test 3 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 3: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);
    
    let sdp_test_4 = SDP::new(
	// min trace(X)
	// s.t. X >= I
	// no linear constraints
	vec![sym_matrix_trace(2)],
	(vec![], vec![]),
	vec![(vec![ident_symmap(2)], mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_4) });
    reference = (Some(vec![vec![1.0, 0.0, 1.0]]), 2.0);
    assert!(sdp_equality_err(&sdp_test_4, &result, &reference), "test 4 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 4: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_5 = SDP::new(
	// X is 3x3
	// min trace(X) - X_13
	// s.t. trace(X) = 4, X_33 = 1
	//      X >= 0
	vec![sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_scale(-1.0, &sym_matrix_proj(3, 0, 2)))],
	(vec![vec![sym_matrix_trace(3), sym_matrix_proj(3, 2, 2)]], vec![4.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_5) });
    reference = (None, 2.267949192431123);
    assert!(sdp_equality_err(&sdp_test_5, &result, &reference), "test 5 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 5: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_6 = SDP::new(
	// X1 is 2x2, X2 is 3x3
	// min 2 trace(X1) + trace(X2) - X2_12
	// s.t. trace(X1) + trace(X2) = 5, (X1)_22 = 1, (X2)_33 = 1
	//      X1 >= 0, X2 >= 0
	vec![
	    sym_matrix_scale(2.0, &sym_matrix_trace(2)),
	    sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_scale(-1.0, &sym_matrix_proj(3, 0, 1)))],
	(
	    vec![
		vec![
		    sym_matrix_trace(2),
                    sym_matrix_proj(2, 1, 1),
                    sym_matrix_zero(2)],
		vec![
                    sym_matrix_trace(3),
                    sym_matrix_zero(3),
                    sym_matrix_proj(3, 2, 2)]],
	    vec![5.0, 1.0, 1.0]),
	vec![
	    (vec![ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
	    (vec![zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_6) });
    reference = (None, 4.5);
    assert!(sdp_equality_err(&sdp_test_6, &result, &reference), "test 6 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 6: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_7 = SDP::new(
	// X1, X2 are 2x2; X3 is 3x3
	// min trace(X1) + 2 trace(X2) + trace(X3) - X3_13
	// s.t. trace(X1) + trace(X2) + trace(X3) = 7
	//      (X1)_22 = 1, (X2)_22 = 2, (X3)_33 = 1
	//      X1 >= 0, X2 >= 0, X3 >= 0
	vec![
	    sym_matrix_trace(2),
	    sym_matrix_scale(2.0, &sym_matrix_trace(2)),
            sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_scale(-1.0, &sym_matrix_proj(3, 0, 2)))],
	(
            vec![
		vec![
                    sym_matrix_trace(2),
                    sym_matrix_proj(2, 1, 1),
                    sym_matrix_zero(2),
                    sym_matrix_zero(2),
		],
		vec![
                    sym_matrix_trace(2),
                    sym_matrix_zero(2),
                    sym_matrix_proj(2, 1, 1),
                    sym_matrix_zero(2),
		],
		vec![
                    sym_matrix_trace(3),
                    sym_matrix_zero(3),
                    sym_matrix_zero(3),
                    sym_matrix_proj(3, 2, 2),
		],
            ],
            vec![7.0, 1.0, 2.0, 1.0]),
	vec![
            (vec![ident_symmap(2), zero_symmap(2, 4), zero_symmap(3, 4)], zero_mat(2, 2)),
            (vec![zero_symmap(2, 4), ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
            (vec![zero_symmap(2, 9), zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_7) });
    reference = (None, 7.267949192431123);
    assert!(sdp_equality_err(&sdp_test_7, &result, &reference), "test 7 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 7: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_8 = SDP::new(
	// X is 3x3
	// min trace(X) + X_22
	// s.t. trace(X) = 6, X_11 = 2, X_33 = 1, X_12 = 0, X_23 = 0
	//      [X_11  X_13; X_13  X_33] >= I
	// X is also an unsplit PSD variable
	vec![sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_proj(3, 1, 1))],
	(
	    vec![
		vec![
                    sym_matrix_trace(3),
                    sym_matrix_proj(3, 0, 0),
                    sym_matrix_proj(3, 2, 2),
                    sym_matrix_proj(3, 0, 1),
                    sym_matrix_proj(3, 1, 2)]],
	    vec![6.0, 2.0, 1.0, 0.0, 0.0]),
	vec![(
            vec![mk_symmap(3, 4,
			   vec![
			       // (1,1) = X11
                               1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               // (1,2) = X13
                               0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                               // (2,1) = X13
                               0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                               // (2,2) = X33
                               0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
            mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_8) });
    reference = (Some(vec![vec![2.0, 0.0, 3.0, 0.0, 0.0, 1.0]]), 9.0);
    assert!(sdp_equality_err(&sdp_test_8, &result, &reference), "test 8 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 8: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_9 = SDP::new(
	// X is 3x3
	// min trace(X) - X_12
	// s.t. trace(X) = 4, X_11 = 1, X_33 = 1
	// X is an unsplit PSD variable
	vec![sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_scale(-1.0, &sym_matrix_proj(3, 0, 1)))],
	(vec![vec![sym_matrix_trace(3), sym_matrix_proj(3, 0, 0), sym_matrix_proj(3, 2, 2)]],
		     vec![4.0, 1.0, 1.0]),
	vec![],
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_9) });
    reference = (Some(vec![vec![1.0, 1.4142135623730951, 2.0, 0.0, 0.0, 1.0]]), 2.585786437626905);
    assert!(sdp_equality_err(&sdp_test_9, &result, &reference), "test 9 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 9: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_10 = SDP::new(
	// X is 2x2
	// min trace(X) - X_12
	// s.t. trace(X) = 3, X_22 = 1
	//      X >= 0
	vec![sym_matrix_add(&sym_matrix_trace(2), &sym_matrix_scale(-1.0, &sym_matrix_proj(2, 0, 1)))],
	(vec![vec![sym_matrix_trace(2), sym_matrix_proj(2, 1, 1)]], vec![3.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_10) });
    reference = (Some(vec![vec![2.0, 1.4142135623730951, 1.0]]), 1.585786437626905);
    assert!(sdp_equality_err(&sdp_test_10, &result, &reference), "test 10 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 10: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_11 = SDP::new(
	// X is 2x2
	// infeasible:
	// trace(X) = 1, X_11 = 1, X_22 = 1
	vec![sym_matrix_trace(2)],
	(vec![vec![sym_matrix_trace(2), sym_matrix_proj(2, 0, 0), sym_matrix_proj(2, 1, 1)]],
		     vec![1.0, 1.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_11) });
    assert!(result.is_err(), "test 11 failed");

    let sdp_test_12 = SDP::new(
	// X is 3x3
	// min trace(X) + X_22 - X_13
	// s.t. X_11 = 1, X_33 = 1
	//      X >= 0
	vec![sym_matrix_add(&sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_proj(3, 1, 1)),
				       &sym_matrix_scale(-1.0, &sym_matrix_proj(3, 0, 2)))],
	(vec![vec![sym_matrix_proj(3, 0, 0), sym_matrix_proj(3, 2, 2)]], vec![1.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_12) });
    reference = (Some(vec![vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]), 1.0);
    assert!(sdp_equality_err(&sdp_test_12, &result, &reference), "test 12 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 12: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_13 = SDP::new(
	// X1, X2, X3, X4 are 2x2
	// min trace(X1) + 2 * trace(X2) + 3 * trace(X3) + 4 * trace(X4)
	// s.t. trace(X1) + trace(X2) + trace(X3) + trace(X4) = 10
	//      X1_22 = 1, X2_22 = 2, X3_22 = 3, X4_22 = 4
	//      X1 >= 0, X2 >= 0, X3 >= 0, X4 >= 0
	vec![sym_matrix_trace(2),
			sym_matrix_scale(2.0, &sym_matrix_trace(2)),
			sym_matrix_scale(3.0, &sym_matrix_trace(2)),
			sym_matrix_scale(4.0, &sym_matrix_trace(2))],
	(
            vec![
		vec![sym_matrix_trace(2),
                     sym_matrix_proj(2, 1, 1),
                     sym_matrix_zero(2),
                     sym_matrix_zero(2),
                     sym_matrix_zero(2)],
		vec![sym_matrix_trace(2),
                     sym_matrix_zero(2),
                     sym_matrix_proj(2, 1, 1),
                     sym_matrix_zero(2),
                     sym_matrix_zero(2)],
		vec![sym_matrix_trace(2),
                     sym_matrix_zero(2),
                     sym_matrix_zero(2),
                     sym_matrix_proj(2, 1, 1),
                     sym_matrix_zero(2)],
		vec![sym_matrix_trace(2),
		     sym_matrix_zero(2),
                     sym_matrix_zero(2),
                     sym_matrix_zero(2),
                     sym_matrix_proj(2, 1, 1)]],
            vec![10.0, 1.0, 2.0, 3.0, 4.0]),
	vec![
	    (vec![ident_symmap(2), zero_symmap(2, 4), zero_symmap(2, 4), zero_symmap(2, 4)],
             zero_mat(2, 2)),
            (vec![zero_symmap(2, 4), ident_symmap(2), zero_symmap(2, 4), zero_symmap(2, 4)],
             zero_mat(2, 2)),
	    (vec![zero_symmap(2, 4), zero_symmap(2, 4), ident_symmap(2), zero_symmap(2, 4)],
	     zero_mat(2, 2)),
            (vec![zero_symmap(2, 4), zero_symmap(2, 4), zero_symmap(2, 4), ident_symmap(2)],
             zero_mat(2, 2))],
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_13) });
    reference = (Some(
	vec![vec![0.0, 0.0, 1.0],
             vec![0.0, 0.0, 2.0],
             vec![0.0, 0.0, 3.0],
             vec![0.0, 0.0, 4.0]]),
	30.0);
    assert!(sdp_equality_err(&sdp_test_13, &result, &reference), "test 13 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 13: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_14 = SDP::new(
	// X is 3x3
	// min X_22
	// s.t. X_11 = 1, X_33 = 1, X_13 = 1
	vec![sym_matrix_proj(3, 1, 1)],
	(vec![vec![sym_matrix_proj(3, 0, 0), sym_matrix_proj(3, 2, 2), sym_matrix_proj(3, 0, 2)]],
		     vec![1.0, 1.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_14) });
    reference = (Some(vec![vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]), 0.0);
    assert!(sdp_equality_err(&sdp_test_14, &result, &reference), "test 14 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 14: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_15 = SDP::new(
	// X is 2x2
	// min trace(X) - X_12
	// s.t. trace(X) = 2
	//      [X_11  X_12; X_12  1] >= 0, [1  X_12; X_12  X_22] >= 0
	vec![sym_matrix_add(&sym_matrix_trace(2), &sym_matrix_scale(-1.0, &sym_matrix_proj(2, 0, 1)))],
	(vec![vec![sym_matrix_trace(2)]], vec![2.0]),
	vec![
	    (vec![mk_symmap(2, 4, vec![1.0, 0.0, 0.0,
				       0.0, 1.0, 0.0,
				       0.0, 1.0, 0.0,
				       0.0, 0.0, 0.0])],
	     mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0])),
	    (vec![mk_symmap(2, 4, vec![0.0, 0.0, 0.0,
				       0.0, 1.0, 0.0,
				       0.0, 1.0, 0.0,
				       0.0, 0.0, 1.0])],
	     mk_sqmatrix(2, vec![-1.0, 0.0, 0.0, 0.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_15) });
    reference = (Some(vec![vec![1.0, 1.0, 1.0]]), 1.0);
    assert!(sdp_equality_err(&sdp_test_15, &result, &reference), "test 15 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 15: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_16 = SDP::new(
	// X is 2x2; min X11 + X22 - X12; X11 + X22 = 3, X22 = 1, X >= 0
	mk_symmap(2, 1, vec![1.0, -1.0, 1.0]),
	(vec![mk_symmap(2, 2, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0])],
		     vec![3.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_16) });
    reference = (Some(vec![vec![2.0, 1.4142135623730951, 1.0]]), 1.585786437626905);
    assert!(sdp_equality_err(&sdp_test_16, &result, &reference), "test 16 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 16: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_17 = SDP::new(
	// X is 3x3; min X11 + 2 * X22 + X33 - X13; tr(X) = 4, X33 = 1, X >= 0
	mk_symmap(3, 1, vec![1.0, 0.0, 2.0, -1.0, 0.0, 1.0]),
	(
	    vec![mk_symmap(3, 2, vec![
		1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
	    vec![4.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_17) });
    reference = (Some(vec![vec![3.0, 0.0, 0.0, 1.7320508075688772, 0.0, 1.0]]), 2.267949192431123);
    assert!(sdp_equality_err(&sdp_test_17, &result, &reference), "test 17 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 17: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_18 = SDP::new(
	// X is 2x2; min X11; X11 - X12 = 1, X22 = 1, [X11 + X22, X12; X12, 1] >= 0
	mk_symmap(2, 1, vec![1.0, 0.0, 0.0]),
	(vec![mk_symmap(2, 2,
			vec![1.0, -1.0, 0.0,
			     0.0, 0.0, 1.0])],
	 vec![1.0, 1.0]),
	vec![(
	    vec![mk_symmap(2, 4,
			   vec![1.0, 0.0, 1.0,
				0.0, 1.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.0])],
	    mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_18) });
    reference = (Some(vec![vec![0.0, -1.0, 1.0]]), 0.0);
    assert!(sdp_equality_err(&sdp_test_18, &result, &reference), "test 18 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 18: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_20 = SDP::new(
	// X is 3x3; min tr(X) - X12- X23; X11 = X22 = X33 = 1, X13 = 0
	mk_symmap(3, 1, vec![1.0, -1.0, 1.0, 0.0, -1.0, 1.0]),
	(vec![mk_symmap(3, 4,
			vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			     0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
			     0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
			     0.0, 0.0, 0.0, 1.0, 0.0, 0.0])],
	 vec![1.0, 1.0, 1.0, 0.0]),
	vec![(
	    vec![mk_symmap(3, 4,
			   vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
				0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0])],
	    mk_sqmatrix(2, vec![-1.0, 0.0, 0.0, -1.0])),
			   (vec![mk_symmap(3, 4, vec![
			       0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			       0.0, 0.0, 0.0, -1.0, 1.0, 0.0,
			       0.0, 0.0, 0.0, -1.0, 1.0, 0.0,
			       0.0, 0.0, 0.0, 0.0, 0.0, 0.0])],
			    mk_sqmatrix(2, vec![-1.0, 0.0, 0.0, -1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_20) });
    reference = (Some(vec![vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0]]), 1.0);
    assert!(sdp_equality_err(&sdp_test_20, &result, &reference), "test 20 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 20: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);
 
    let sdp_test_19 = SDP::new(
	// X1, X2 are 2x2; min (X1_11 - X1_12) + (X2_11 - X2_12); X1_22 = X2_22 = 1, same X11 and X12
	mk_symmap(2, 2, vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0]),
	(vec![mk_symmap(2, 4,
			vec![0.0, 0.0, 1.0,
			     0.0, 0.0, 0.0,
			     1.0, 0.0, 0.0,
			     0.0, 1.0, 0.0]),
	      mk_symmap(2, 4,
			vec![0.0, 0.0, 0.0,
			     0.0, 0.0, 1.0,
			     -1.0, 0.0, 0.0,
			     0.0, -1.0, 0.0])],
	 vec![1.0, 1.0, 0.0, 0.0]),
	vec![(
	    vec![mk_symmap(2, 4,
			   vec![1.0, 0.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.0]),
		 mk_symmap(2, 4,
			   vec![1.0, 0.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.0])],
	    mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -2.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_19) });
    reference = (Some(vec![vec![0.25, 0.5, 1.0], vec![0.25, 0.5, 1.0]]), -0.5);
    assert!(sdp_equality_err(&sdp_test_19, &result, &reference), "test 19 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 19: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_21 = SDP::new(
        // X1 is 2x2, X2 is 3x3
        // min (X1_11 + X1_22 - X1_12) + (X2_11 + 2 * X2_22 + X2_33 - X2_13)
        // s.t. trace(X1) = 3, X1_22 = 1, trace(X2) = 4, X2_33 = 1
        //      X1 >= 0, X2 >= 0
        vec![
            mk_symmap_real(2, &vec![1.0, -1.0, 1.0]),
            mk_symmap_real(3, &vec![1.0, 0.0, 2.0, -1.0, 0.0, 1.0])],
        (
            vec![
                mk_symmap(2, 4, vec![
                    1.0, 0.0, 1.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0]),
                mk_symmap(3, 4, vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
            vec![3.0, 1.0, 4.0, 1.0]),
        vec![
            (vec![ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
            (vec![zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_21) });
    reference = (
        Some(vec![
            vec![2.0, 1.4142135623730951, 1.0],
            vec![3.0, 0.0, 0.0, 1.7320508075688772, 0.0, 1.0]]),
        3.853735630058028);
    assert!(sdp_equality_err(&sdp_test_21, &result, &reference), "test 21 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 21: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_22 = SDP::new(
        // X1 is 2x2, X2 is 3x3
        // min X1_11 + trace(X2) - X2_12 - X2_23
        // s.t. X1_11 - X1_12 = 1, X1_22 = 1
        // X2_11 = X2_22 = X2_33 = 1, X2_13 = 0
        // X1 >= 0, X2 >= 0
        // [X1_11 + X1_22 X1_12; X1_12 1] >= 0
        // [X2_11 X2_12; X2_12 X2_22] >= I
        // [X2_22 X2_23; X2_23 X2_33] >= I
        vec![
            mk_symmap_real(2, &vec![1.0, 0.0, 0.0]),
            mk_symmap_real(3, &vec![1.0, -1.0, 1.0, 0.0, -1.0, 1.0])],
        (
            vec![
                mk_symmap(2, 6, vec![
                    1.0, -1.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0]),
                mk_symmap(3, 6, vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0])],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        vec![
            (vec![ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
            (vec![zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3)),
            (vec![
                mk_symmap(2, 4, vec![
                    1.0, 0.0, 1.0,
                    0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0]),
                zero_symmap(3, 4)],
             mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0])),
            (vec![
                zero_symmap(2, 4),
                mk_symmap(3, 4, vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0])],
             mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0])),
            (vec![
                zero_symmap(2, 4),
                mk_symmap(3, 4, vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
             mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_22) });
    reference = (
        Some(vec![
            vec![0.3819660111005423, -0.6180339888994575, 1.0000000000000002],
            vec![1.0000000000000002, -1.95006694600038e-13, 1.0, 0.0, -1.9497356183902432e-13, 1.0000000000000002]]),
        3.3819660111009333);
    assert!(sdp_equality_err(&sdp_test_22, &result, &reference), "test 22 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 22: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);
    
    let sdp_test_23 = SDP::new(
	// X1 is 2x2, X2 is 3x3
	// min (X1_11 + X1_22 - X1_12) + (trace(X2) + X2_22)
	// s.t. trace(X1) = 3, X1_22 = 1
	//      trace(X2) = 6, X2_11 = 2, X2_33 = 1, X2_12 = 0, X2_23 = 0
	//      X1 >= 0, X2 >= 0
	//      [X2_11  X2_13; X2_13  X2_33] >= I
	vec![
	    mk_symmap_real(2, &vec![1.0, -1.0, 1.0]),
	    mk_symmap_real(3, &vec![1.0, 0.0, 2.0, 0.0, 0.0, 1.0])],
	(
	    vec![
		mk_symmap(2, 7, vec![
		    1.0, 0.0, 1.0,
		    0.0, 0.0, 1.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0]),
		mk_symmap(3, 7, vec![
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
		    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
	    vec![3.0, 1.0, 6.0, 2.0, 1.0, 0.0, 0.0]),
	vec![
	    (vec![ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
	    (vec![zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3)),
	    (vec![
		zero_symmap(2, 4),
		mk_symmap(3, 4, vec![
		    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
	     mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_23) });
    reference = (
        Some(vec![
            vec![2.0, 1.4142135623730951, 1.0],
            vec![2.0, 0.0, 3.0, 0.0, 0.0, 1.0]]),
        10.585786437626904);
    assert!(sdp_equality_err(&sdp_test_23, &result, &reference), "test 23 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 23: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_24 = SDP::new(
	// X1 is 2x2, X2 is 3x3, X3 is 2x2
	// min X1_11 + X2_22 + (X3_11 + X3_22 - X3_12)
	// s.t. X1_11 - X1_12 = 1, X1_22 = 1
	//      X2_11 = 1, X2_33 = 1, X2_13 = 1
	//      trace(X3) = 3, X3_22 = 1
	//      X1 >= 0, X2 >= 0, X3 >= 0
	//      [X1_11 + X1_22  X1_12; X1_12  1] >= 0
	vec![
	    mk_symmap_real(2, &vec![1.0, 0.0, 0.0]),
	    mk_symmap_real(3, &vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
	    mk_symmap_real(2, &vec![1.0, -1.0, 1.0])],
	(
	    vec![
		mk_symmap(2, 7, vec![
		    1.0, -1.0, 0.0,
		    0.0, 0.0, 1.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0]),
		mk_symmap(3, 7, vec![
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
		mk_symmap(2, 7, vec![
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    1.0, 0.0, 1.0,
		    0.0, 0.0, 1.0])],
	    vec![1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0]),
	vec![
	    (vec![ident_symmap(2), zero_symmap(3, 4), zero_symmap(2, 4)], zero_mat(2, 2)),
	    (vec![zero_symmap(2, 9), ident_symmap(3), zero_symmap(2, 9)], zero_mat(3, 3)),
	    (vec![zero_symmap(2, 4), zero_symmap(3, 4), ident_symmap(2)], zero_mat(2, 2)),
	    (vec![
		mk_symmap(2, 4, vec![
		    1.0, 0.0, 1.0,
		    0.0, 1.0, 0.0,
		    0.0, 1.0, 0.0,
		    0.0, 0.0, 0.0]),
		zero_symmap(3, 4),
		zero_symmap(2, 4)],
	     mk_sqmatrix(2, vec![0.0, 0.0, 0.0, -1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_24) });
    reference = (
        Some(vec![
            vec![0.3819660112502016, -0.618033988749799, 1.0000000000000002],
            vec![1.0000000000000004, 0.0, 0.0, 0.9999999999999998, 0.0, 1.0000000000000004],
            vec![1.9999999999999996, 1.4142135622683163, 1.0000000000000002]]),
        1.9677524489818858);
    assert!(sdp_equality_err(&sdp_test_24, &result, &reference), "test 24 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 24: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_25 = SDP::new(
	// X1 is 2x2, X2 is 3x3
	// min (X1_11 + X1_22 - X1_12) + (trace(X2) - X2_12 - X2_23)
	// s.t. trace(X1) = 3, X1_22 = 1
	//      X2_11 = X2_22 = X2_33 = 1, X2_13 = 0
	//      X1 >= 0, X2 >= 0
	//      [X2_11  X2_12; X2_12  X2_22] >= I
	//      [X2_22  X2_23; X2_23  X2_33] >= I
	vec![
	    mk_symmap_real(2, &vec![1.0, -1.0, 1.0]),
	    mk_symmap_real(3, &vec![1.0, -1.0, 1.0, 0.0, -1.0, 1.0])],
	(
	    vec![
		mk_symmap(2, 6, vec![
		    1.0, 0.0, 1.0,
		    0.0, 0.0, 1.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0]),
		mk_symmap(3, 6, vec![
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		    0.0, 0.0, 0.0, 1.0, 0.0, 0.0])],
	    vec![3.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
	vec![
	    (vec![ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
	    (vec![zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3)),
	    (vec![
		zero_symmap(2, 4),
		mk_symmap(3, 4, vec![
		    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 1.0, 0.0, 0.0, 0.0])],
	     mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0])),
	    (vec![
		zero_symmap(2, 4),
		mk_symmap(3, 4, vec![
		    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		    0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
	     mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_25) });
    reference = (
        Some(vec![
            vec![1.9999999999999998, 1.414213562577729, 1.0000000000000002],
            vec![1.0000000000000002, -1.4034931783073302e-16, 1.0, 0.0, -1.427436936049385e-16, 1.0000000000000002]]),
        4.585786437422271);
    assert!(sdp_equality_err(&sdp_test_25, &result, &reference), "test 25 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 25: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);
    
    // some simple matrix completion problems (via nuclear norm minimization)
    
    let sdp_test_26 = SDP::new(
	// min ||M||_* s.t. M_11 = 1, M_12 = 1, M_21 = 1
	// lifted: min 1/2(tr(W1)+tr(W2)), [W1 M; M^T W2] >= 0
	vec![sym_matrix_scale(0.5, &sym_matrix_trace(4))],
	(
	    vec![vec![
		sym_matrix_proj(4, 0, 2),
		sym_matrix_proj(4, 0, 3),
		sym_matrix_proj(4, 1, 2)]],
	    vec![3.0, 1.0, 1.0]
	),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_26) });
    reference = (None, 10.0 / 3.0);
    assert!(sdp_equality_err(&sdp_test_26, &result, &reference), "test 26 failed");
    let (sol, _sol_obj) = result.as_ref().expect("test 26 failed");
    assert!(float_equality(sol[0][7], 1.0 / 3.0, TOL_TEST), "test 26: matrix completion failed with {}", sol[0][7]);
    //println!("\ntest 26: decision variable solutions= {:#?} with objective value= {}", sol, sol_obj);

    let sdp_test_27 = SDP::new(
	// 2x3 matrix completion via lifting
	// M = [1 2 d; 2 4 1]
	vec![sym_matrix_scale(0.5, &sym_matrix_trace(5))],
	(
	    vec![vec![
		sym_matrix_proj(5, 0, 2),
		sym_matrix_proj(5, 0, 3),
		sym_matrix_proj(5, 1, 2),
		sym_matrix_proj(5, 1, 3),
		sym_matrix_proj(5, 1, 4)]],
	    vec![1.0, 2.0, 2.0, 4.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_27) });
    reference = (None, 0.5 * 105.0_f64.sqrt());
    assert!(sdp_equality_err(&sdp_test_27, &result, &reference), "test 27 failed");
    let (sol, _sol_obj) = result.as_ref().expect("test 27 failed");
    assert!(float_equality(sol[0][10], 0.5, TOL_TEST), "test 27: matrix completion failed with {}", sol[0][10]);
    //println!("\ntest 27: decision variable solutions= {:#?} with objective value= {}", sol, sol_obj);

    let eps = 1e-4;
    let sdp_test_28 = SDP::new(
	// 2x2 matrix completion via lifting
	// M = [1  eps; eps  d]
	vec![sym_matrix_scale(0.5, &sym_matrix_trace(4))],
	(
	    vec![vec![
		sym_matrix_proj(4, 0, 2),
		sym_matrix_proj(4, 0, 3),
		sym_matrix_proj(4, 1, 2)]],
	    vec![1.0, eps, eps]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_28) });
    reference = (None, 1.0 + eps * eps);
    assert!(sdp_equality_err(&sdp_test_28, &result, &reference), "test 28 failed");
    let (sol, _sol_obj) = result.as_ref().expect("test 28 failed");
    assert!(float_equality(sol[0][7], eps * eps, TOL_TEST), "test 28: matrix completion failed with {}", sol[0][7]);
    //println!("\ntest 28: decision variable solutions= {:#?} with objective value= {}", sol, sol_obj);

    let sdp_test_29 = SDP::new(
	// min ||M||_* s.t. M_11 + M_22 = 1
	// lifted: min 1/2(tr(W1)+tr(W2)), X = [W1 M; M^T W2] >= 0
	vec![sym_matrix_scale(0.5, &sym_matrix_trace(4))],
	(vec![vec![sym_matrix_add(&sym_matrix_proj(4, 0, 2), &sym_matrix_proj(4, 1, 3))]], vec![1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_29) });
    reference = (None, 1.0);
    assert!(sdp_equality_err(&sdp_test_29, &result, &reference), "test 29 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 29: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let eps = 1e-3;
    let sdp_test_30 = SDP::new(
	// perturbed 2x2 matrix completion via lifting
	// M = [1 1; 1+eps d]
	vec![sym_matrix_scale(0.5, &sym_matrix_trace(4))],
	(
	    vec![vec![
		sym_matrix_proj(4, 0, 2),
		sym_matrix_proj(4, 0, 3),
		sym_matrix_proj(4, 1, 2)]],
	    vec![1.0, 1.0, 1.0 + eps]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_30) });
    reference = (None, 2.0 + eps);
    assert!(sdp_equality_err(&sdp_test_30, &result, &reference), "test 30 failed");
    let (sol, _sol_obj) = result.as_ref().expect("test 30 failed");
    assert!(float_equality(sol[0][7], 1.0, TOL_TEST), "test 30: matrix completion failed with {}", sol[0][7]);
    //println!("\ntest 30: decision variable solutions= {:#?} with objective value= {}", sol, sol_obj);

    // a few higher-dimensional problems

    let sdp_test_31 = SDP::new(
	// X is 4x4
	// min trace(X) - X_14
	// s.t. trace(X) = 6, X_11 = 2, X_22 = 1, X_33 = 1, X_44 = 2
	//      X_12 = X_13 = X_23 = X_24 = X_34 = 0
	//      [X_11  X_14; X_14  X_44] >= I
	vec![sym_matrix_add(&sym_matrix_trace(4), &sym_matrix_scale(-1.0, &sym_matrix_proj(4, 0, 3)))],
	(
	    vec![vec![
		sym_matrix_trace(4),
		sym_matrix_proj(4, 0, 0),
		sym_matrix_proj(4, 1, 1),
		sym_matrix_proj(4, 2, 2),
		sym_matrix_proj(4, 3, 3),
		sym_matrix_proj(4, 0, 1),
		sym_matrix_proj(4, 0, 2),
		sym_matrix_proj(4, 1, 2),
		sym_matrix_proj(4, 1, 3),
		sym_matrix_proj(4, 2, 3)]],
	    vec![6.0, 2.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
	vec![(
	    vec![mk_symmap(4, 4,
			   vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
	    mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_31) });
    reference = (
	Some(vec![vec![2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0]]),
	5.0);
    assert!(sdp_equality_err(&sdp_test_31, &result, &reference), "test 31 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 31: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_32 = SDP::new(
	// X1 and X2 are 4x4
	// min (trace(X1) - X1_14) + (2 * trace(X2) - 2 * X2_14)
	// s.t. trace(X1) = 5, X1_44 = 1
	//      trace(X2) = 6, X2_44 = 2
	//      X1 >= 0, X2 >= 0
	vec![
	    sym_matrix_add(&sym_matrix_trace(4), &sym_matrix_scale(-1.0, &sym_matrix_proj(4, 0, 3))),
	    sym_matrix_add(&sym_matrix_scale(2.0, &sym_matrix_trace(4)),
			   &sym_matrix_scale(-2.0, &sym_matrix_proj(4, 0, 3)))],
	(
	    vec![
		vec![sym_matrix_trace(4),
		     sym_matrix_proj(4, 3, 3),
		     sym_matrix_zero(4),
		     sym_matrix_zero(4)],
		vec![sym_matrix_zero(4),
		     sym_matrix_zero(4),
		     sym_matrix_trace(4),
		     sym_matrix_proj(4, 3, 3)]],
	    vec![5.0, 1.0, 6.0, 2.0]),
	vec![
	    (vec![ident_symmap(4), zero_symmap(4, 16)], zero_mat(4, 4)),
	    (vec![zero_symmap(4, 16), ident_symmap(4)], zero_mat(4, 4))]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_32) });
    reference = (
	Some(vec![
	    vec![4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
	    vec![4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.8284271247461903, 0.0, 0.0, 2.0]]),
	9.34314575050762);
    assert!(sdp_equality_err(&sdp_test_32, &result, &reference), "test 32 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 32: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_33 = SDP::new(
	// X is 4x4
	// min trace(X) - X_14
	// s.t. trace(X) = 5, X_44 = 1
	//      X >= 0
	vec![sym_matrix_add(&sym_matrix_trace(4), &sym_matrix_scale(-1.0, &sym_matrix_proj(4, 0, 3)))],
	(
	    vec![vec![
		sym_matrix_trace(4),
		sym_matrix_proj(4, 3, 3)]],
	    vec![5.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_33) });
    reference = (
	Some(vec![vec![4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0]]),
	3.0);
    assert!(sdp_equality_err(&sdp_test_33, &result, &reference), "test 33 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 33: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    // a couple of problems with redundant constraints

    let sdp_test_34 = SDP::new(
	// X is 2x2
	// min trace(X) - X_12
	// s.t. trace(X) = 3, X_22 = 1, 2 * X_22 = 2
	//      X >= 0
	// the last constraint is redundant
	vec![sym_matrix_add(&sym_matrix_trace(2), &sym_matrix_scale(-1.0, &sym_matrix_proj(2, 0, 1)))],
	(
	    vec![vec![
		sym_matrix_trace(2),
		sym_matrix_proj(2, 1, 1),
		sym_matrix_scale(2.0, &sym_matrix_proj(2, 1, 1))]],
	    vec![3.0, 1.0, 2.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_34) });
    reference = (Some(vec![vec![2.0, 1.4142135623730951, 1.0]]), 1.585786437626905);
    assert!(sdp_equality_err(&sdp_test_34, &result, &reference), "test 34 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 34: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_35 = SDP::new(
	// X is 3x3
	// min trace(X) - X_13
	// s.t. trace(X) = 4, X_33 = 1, X_33 = 1
	//      X >= 0
	// the last constraint is redundant
	vec![sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_scale(-1.0, &sym_matrix_proj(3, 0, 2)))],
	(
	    vec![vec![
		sym_matrix_trace(3),
		sym_matrix_proj(3, 2, 2),
		sym_matrix_proj(3, 2, 2)]],
	    vec![4.0, 1.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_35) });
    reference = (Some(vec![vec![3.0, 0.0, 0.0, 1.7320508075688772, 0.0, 1.0]]), 2.267949192431123);
    assert!(sdp_equality_err(&sdp_test_35, &result, &reference), "test 35 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 35: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    // a couple of single-variable problems with no given linear constraints

    let sdp_test_36 = SDP::new(
	// X is 2x2
	// min 2 * X_11 + X_22
	// s.t. X >= 0
	// no linear constraints
	vec![sym_matrix_add(&sym_matrix_scale(2.0, &sym_matrix_proj(2, 0, 0)),
				       &sym_matrix_proj(2, 1, 1))],
	(vec![], vec![]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_36) });
    reference = (Some(vec![vec![0.0, 0.0, 0.0]]), 0.0);
    assert!(sdp_equality_err(&sdp_test_36, &result, &reference), "test 36 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 36: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_37 = SDP::new(
	// X is 3x3
	// min trace(X) + X_22
	// s.t. X >= 0
	// no linear constraints
	vec![sym_matrix_add(&sym_matrix_trace(3), &sym_matrix_proj(3, 1, 1))],
	(vec![], vec![]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_37) });
    reference = (Some(vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 0.0);
    assert!(sdp_equality_err(&sdp_test_37, &result, &reference), "test 37 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 37: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    // a few simple problems for which the Slater conditions may fail

    let sdp_test_38 = SDP::new(
	// X is 2x2
	// min trace(X)
	// s.t. X_11 = 0, X_22 = 1
	//      X >= 0
	vec![sym_matrix_trace(2)],
	(vec![vec![sym_matrix_proj(2, 0, 0), sym_matrix_proj(2, 1, 1)]],
	 vec![0.0, 1.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_38) });
    reference = (Some(vec![vec![0.0, 0.0, 1.0]]), 1.0);
    assert!(sdp_equality_err(&sdp_test_38, &result, &reference), "test 38 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 38: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);
    
    let sdp_test_39 = SDP::new(
	// X is 2x2
	// min X_11
	// s.t. X >= 0
	// no linear constraints
	vec![sym_matrix_proj(2, 0, 0)],
	(vec![], vec![]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_39) });
    reference = (None, 0.0);
    assert!(sdp_equality_err(&sdp_test_39, &result, &reference), "test 39 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 39: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);

    let sdp_test_40 = SDP::new(
	// X is 2x2
	// min X_11
	// s.t. X_11 = 0
	//      X >= 0
	vec![sym_matrix_proj(2, 0, 0)],
	(vec![vec![sym_matrix_proj(2, 0, 0)]], vec![0.0]),
	vec![]
    );
    result = panic::catch_unwind(|| { sdpad(&sdp_test_40) });
    reference = (None, 0.0);
    assert!(sdp_equality_err(&sdp_test_40, &result, &reference), "test 40 failed");
    //result_pr = result.as_ref().unwrap();
    //println!("\ntest 40: decision variable solutions= {:#?} with objective value= {}", result_pr.0, result_pr.1);
}
