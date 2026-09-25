// runner for sdp solver

use convopt_rust::sdp::*;
use convopt_rust::sdp_altdir::*;

fn main() {
    // give your SDP problem here
    let sdp_problem = SDP::new(
	// fill in your SDP problem
        vec![sym_matrix_trace(1)],
	(vec![], vec![]),
	vec![]
    );
    // choose the variant of the Jacobi method you want to use for the solver
    let jacobi_variant = JacobiVariant::Cached;
    // decide whether you want to run Gaussian elimination on the dense constraint matrix
    let check_constraints : bool = true;
    let result = sdpad(&sdp_problem, check_constraints, &jacobi_variant);
    println!("decision variable solutions= {:#?} with objective value= {}", result.0, result.1)
}
