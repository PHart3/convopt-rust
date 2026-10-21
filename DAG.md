L00 | Trusted exact-real model: rv: f64 -> real, sqrt_r axiom, exact-real axioms for vstd's f64 operator and comparison hooks, literal axioms, assume_specification contracts for std f64 methods, inline wrappers for unsupported primitives (neg, pos_infinity, abort); verified comparison functions |
deps: none |
targets: float_equality, scaled_near_zero, pivot_below_positive_threshold, is_significantly_negative, residual_is_small, relative_residual_is_small |
status: done

L01 | Vector views (Seq<real>) and basic vector and matrix operations |
deps: L00 |
targets: standard_basis, ident_mat, zero_mat, dot_prod, euclid_norm, euclid_distance, vect_add, vect_subt, scal_vect, negscal_vect, vect_add_scaled, matrix_add_diff, ident_mat_low_tri, vector_equality |
status: done

L02 | Packed symmetric view (packed index lemmas, symmetric entry view), column-major view, basic symmetric operations |
deps: L01 |
targets: max_diag_sym, frob_prod_sym, frob_norm_sym, frob_dist_sym, sym_mat_one_norm, ident_sym_mat, sym_mat_act, scale_off_diag, outer_prod_upper, outer_prod_single, outer_prod_single_scale, diag_of_sym_mat, sym_matrix_diag_block_map, sym_matrix_diag_block, sym_matrix_diagonal |
status: done

L03 | Spec-level matrix algebra: product, transpose, identity, orthogonality, eigenvalues and their similarity invariance, the order A >= -eps I |
deps: L02 |
targets: (spec/proof only: mmul, mtr, is_orthogonal, is_eigenvalue, psd_shifted, lemma_similarity_preserves_eigenvalues, lemma_mul_assoc, lemma_transpose_mul) |
status: done

L04 | Constraint operator A and adjoint A* with the off-diagonal factors 1/2: <A(X), y> = <X, A*(y)>_F; Gram matrix; Q D Q^T reconstruction |
deps: L03 |
targets: constraint_action, constraint_lin_comb, constraint_gram, diag_scale_gram |
status: done

L05 | Triangular and diagonal solvers: L x = b, L^T x = b, D x = b on the packed lower-triangular view, with division-safety contracts |
deps: L02 |
targets: forw_subst, back_subst, diag_solver |
status: done

L06 | LDLT and positive-definite solvers: L D L^T = A with unit-lower L and pivots d_i >= TOL * max(max_diag, 1); A x = b; upper-left block solve |
deps: L05, L03 |
targets: ldlt_decomp, pos_def_solver, pos_def_solver_upperleft |
status: done

L07 | PSD block check: success gives B >= -eps I for each block, eps = 1e-9 * max(max_diag(B), 1), from the shifted LDLT with positive pivots |
deps: L06 |
targets: psd_block_check |
status: done

L08 | Shared Jacobi rotation: for a pivot j < k, J^T J = J J^T = I, A' = J^T A J, A'_jk = 0, symmetry and eigenvalues preserved, Q' = Q J, returns |A_jk| |
deps: L03 |
targets: jacobi_rot |
status: done

L09 | Common eigensolver specification: invariant eigen_decomp_inv (Q orthogonal, A = Q B Q^T), preserved by every rotation; stopping predicate offdiag_below implies ||B - diag(B)||_F <= eps; shared contract eigensolver_ok |
deps: L08 |
targets: (spec/proof only: eigen_decomp_inv, eigen_decomp_approx, offdiag_below, lemma_eigen_inv_step, lemma_converged_approx, eigensolver_ok) |
status: done

L10 | Cached Jacobi: index_max_in_col returns a column argmax; maxima[m-1] is a maximal row of column m, kept correct by the update after each rotation; the pivot is a global off-diagonal maximum; eigensolver_ok |
deps: L09 |
targets: index_max_in_col, jacobi_eigen_cach |
status: done

L11 | Column-cyclic Jacobi: each sweep visits every pair 0 <= j < k < dim exactly once in column-cyclic order, each through jacobi_rot; eigensolver_ok |
deps: L09 |
targets: jacobi_eigen_cyc |
status: done

L12 | Spectral operations through eigensolver_ok: poseigendecomp keeps B_kk > 0, negeigendecomp keeps B_kk < -1e-8 * scale; the reconstruction equals Q diag(masked diag(B)) Q^T, PSD for the positive part |
deps: L10, L11, L04 |
targets: poseigendecomp, negeigendecomp |
status: done

L13 | Gaussian elimination: echelon invariants, partial-pivot selection, tolerance-based skip rule, solution set preserved, rank and consistency outputs |
deps: L02 |
targets: gauss_elim |
status: done

L14 | Constraint reduction: the first rank echelon rows, each divided by its Frobenius norm; input solutions solve the reduced system; reduced solutions satisfy each dropped row up to a residual bounded by a multiple of eps; inconsistency aborts |
deps: L13 |
targets: linear_remove_redundant_sym |
status: done

L15 | User-facing symmetric maps (MatrixSym), flattening, diagonal-block embeddings, helper constructors |
deps: L02 |
targets: matrixsym_flatten, matrix_diagonal_mult_sym, mat_sym_diagonal_mult_sym, mk_symmap_real, sym_matrix_trace, sym_matrix_scale, sym_matrix_zero, sym_matrix_proj, sym_matrix_add, mk_symmap, zero_symmap, ident_symmap |
status: done

L16 | Standard-form conversion: objective, equality rows, LMI rows (B_i = H_i(..) - M_i on the upper triangle), zero pattern, and reconstruction Z_i = B_{s+2i-1} - B_{s+2i}; both check_constraints branches |
deps: L15, L14 |
targets: sdp_to_standard |
status: done

L17 | Solver setup: Frobenius-weighted objective, Gram matrix, regularized Gram solve with relative-residual contract, PSD block partition, initial iterates; the dual solve is the A A* system of the full constraint operator |
deps: L16, L04, L06 |
targets: sdpad_setup, dual_solve, psd_block_list, initial_iterates |
status: done

L18 | Wen-Goldfarb-Yin iteration: (A A*) y = A(C - S) + mu (b - A(Z)); V = C - A*(y) - mu Z; S' = positive part of V via poseigendecomp; Z' = (1 - rho) Z + (rho / mu)(S' - V); penalty update |
deps: L17, L12, L07 |
targets: wgy_update, penalty_update |
status: done

L19 | Residuals and KKT quantities: pinf, dinf, objective values, gap, delta, feasibility check, stagnation and best-solution bookkeeping, solution read-out; loop outcome over a ghost trajectory from the initial iterates |
deps: L18 |
targets: kkt_quantities, read_solution, sdpad_loop |
status: done

L20 | Final audit: full-crate verification, cargo build and test, trusted-base audit, final counts |
deps: L19 |
targets: (all) |
status: done

L21 | Post-hoc proofs (ADDITIONAL_PROOFS.md): the gap identity <C_F, Z>_F - b . y = <Z, S>_F + <Z, C_F - A*(y) - S>_F + y . (A(Z) - b); soundness (a feasible standard-form point gives feasible user variables) and the converse (feasible user variables lift to a PSD block-diagonal standard-form point with A(Z) = b); block-structure lemmas for the semidefinite order |
deps: L20 |
targets: (spec/proof only: lemma_gap_identity, lemma_gap_complementarity, lemma_standard_feasible_sound, lemma_action_feasible_sound, lemma_user_feasible_lifts, lemma_user_feasible_lifts_single, lemma_psd_shifted_block, lemma_psd_shifted_two_blocks, lemma_psd_shifted_abs_sum) |
status: done
