// runner for sdp solver

use convopt_rust::sdp::*;
use convopt_rust::sdp_altdir::*;

fn main() {
    let sdp_problem = SDP::new(
	// fill in your SDP problem
        vec![sym_matrix_trace(1)],
	(vec![], vec![]),
	vec![]
    );
    let result = sdpad(&sdp_problem);
    println!("\ndecision variable solutions= {:#?} with objective value= {}", result.0, result.1)
}
