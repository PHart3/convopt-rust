L00 | Verus scaffolding and trusted exact-real model: spec view rv: f64 -> real over Verus's native SMT reals, sqrt_r axiom, trusted axioms giving vstd's f64 operator/comparison hooks exact-real meaning, literal axioms, assume_specification contracts for std f64 methods (abs sqrt recip signum max min mul_add hypot is_finite), inline wrappers for unsupported primitives (neg, infinities, FRAC_1_SQRT_2, abort); verified comparison functions; create TRUSTED_BASE.md |
deps: none |
targets: float_equality, scaled_near_zero, pivot_below_positive_threshold, is_significantly_negative, residual_is_small, relative_residual_is_small |
status: done

L01 | Vector views (Seq<Real>) and basic vector/matrix constructors and operations |
deps: L00 |
targets: standard_basis, ident_mat, zero_mat, dot_prod, euclid_norm, euclid_distance, vect_add, vect_subt, scal_vect, negscal_vect, vect_add_scaled, matrix_add_diff, ident_mat_low_tri, vector_equality |
status: not started

L02 | Packed symmetric-matrix view (packed index lemmas, (i,j) -> Real spec view, symmetry), column-major Matrix view, and basic symmetric-matrix operations |
deps: L01 |
targets: max_diag_sym, frob_prod_sym, frob_norm_sym, frob_dist_sym, sym_mat_one_norm, ident_sym_mat, sym_mat_act, scale_off_diag, outer_prod_upper, outer_prod_single, outer_prod_single_scale, diag_of_sym_mat, sym_matrix_diag_block_map, sym_matrix_diag_block, sym_matrix_diagonal |
status: not started

L03 | Reusable spec-level matrix algebra: product, transpose, identity, orthogonality (Q^T Q = I and Q Q^T = I), associativity/transpose lemmas, Frobenius product lemmas, eigenvalue definition and similarity invariance, PSD/order definitions (A >= -eps I) |
deps: L02 |
targets: (spec/proof only: mat_mul, mat_transpose, is_orthogonal, is_eigenvalue, lemma_similarity_preserves_eigenvalues, lemma_mul_assoc, lemma_transpose_mul) |
status: not started

L04 | Constraint operator and adjoint: constraint_action spec, constraint_lin_comb adjoint identity <constraint_action(X), y> = <X, constraint_lin_comb(y)>_F with off-diagonal 1/2 factors, Gram matrix, Q D Q^T reconstruction |
deps: L03 |
targets: constraint_action, constraint_lin_comb, constraint_gram, diag_scale_gram |
status: not started

L05 | Triangular and diagonal solvers: packed lower-triangular view, forward/back substitution solve L x = b and L^T x = b, diagonal solve, with division-safety contracts |
deps: L02 |
targets: forw_subst, back_subst, diag_solver |
status: not started

L06 | LDLT factorization and positive-definite solvers: ldlt_decomp returns L D L^T = A with unit-lower L and all pivots d_i >= TOL * max(max_diag, 1); pos_def_solver solves A x = b; upper-left block solver |
deps: L05, L03 |
targets: ldlt_decomp, pos_def_solver, pos_def_solver_upperleft |
status: not started

L07 | PSD block check: successful psd_block_check establishes each diagonal block B satisfies B >= -eps I with eps = 1e-9 * max(max_diag(B), 1), via shifted LDLT with positive pivots (x^T L D L^T x = sum d_k y_k^2 >= 0) |
deps: L06 |
targets: psd_block_check |
status: not started

L08 | Shared Jacobi rotation core: for pivot (k, j) with j < k prove J^T J = I, J J^T = I, A' = J^T A J, annihilation of the (k, j) entry, symmetry preserved, eigenvalues preserved, eigenvector accumulation Q' = Q J, returned value is |old pivot entry| |
deps: L03 |
targets: jacobi_rot |
status: not started

L09 | Common eigensolver specification: predicate eigen_decomp_approx(A, Q, B, eps) with Q^T Q = I, A = Q B Q^T, B symmetric; max-off-diagonal stopping criterion lemma: max |B_ij| (i != j) < tol * max(1, max_diag) implies ||B - diag(B)||_F <= eps; rotation-sequence invariant lemma (orthogonality and similarity are preserved by each jacobi_rot) |
deps: L08 |
targets: (spec/proof only: eigen_decomp_approx, lemma_offdiag_bound, lemma_rotation_preserves_decomp) |
status: not started

L10 | Cached Jacobi: index_max_in_col returns argmax index in column; cache invariant (maxima[n-1] is column index of row-n maximum); cache update after rotation is correct; selected pivot is the cached global maximum; jacobi_eigen_cach satisfies eigen_decomp_approx with the implementation's tolerance |
deps: L09 |
targets: index_max_in_col, jacobi_eigen_cach |
status: not started

L11 | Column-cyclic Jacobi: each sweep visits every pair 0 <= j < k < dim exactly once in column-cyclic order; every rotation uses jacobi_rot; final max-off-diagonal scan computes the true maximum; jacobi_eigen_cyc satisfies eigen_decomp_approx |
deps: L09 |
targets: jacobi_eigen_cyc |
status: not started

L12 | PSD spectral operations via the common eigensolver interface: poseigendecomp selects eigenpairs with lambda > 0, negeigendecomp selects lambda < -1e-8 * scale; outputs are the selected columns of Q and diagonal of B; reconstruction diag_scale_gram(vals, vects) = sum lambda_i q_i q_i^T; positivity of selected values gives PSD reconstruction |
deps: L10, L11, L04 |
targets: poseigendecomp, negeigendecomp |
status: not started

L13 | Gaussian elimination: row-echelon invariants, partial-pivot selection correctness (chosen pivot has maximal magnitude in remaining rows of the column), tolerance-based skip rule, row operations preserve row space / solution set, rank and consistency outputs |
deps: L02 |
targets: gauss_elim |
status: not started

L14 | Constraint reduction: linear_remove_redundant_sym returns the first rank rows of the echelon form each scaled by its packed Frobenius norm, so the reduced augmented system has the same solution set as the input; inconsistency panic contract |
deps: L13 |
targets: linear_remove_redundant_sym |
status: not started

L15 | User-facing symmetric-map representation (MatrixSym) and composition with diagonal-block embedding: flattening, diagonal-block composition, and helper constructors |
deps: L02 |
targets: matrixsym_flatten, matrix_diagonal_mult_sym, mat_sym_diagonal_mult_sym, mk_symmap_real, sym_matrix_trace, sym_matrix_scale, sym_matrix_zero, sym_matrix_proj, sym_matrix_add, mk_symmap, zero_symmap, ident_symmap |
status: not started

L16 | Standard-form conversion: sdp_to_standard preserves objective (C(Z_1..Z_n) = <obj_sum, Z>), equality constraints, LMI constraints (block B_i = H_i(...) - M_i on upper triangle), zero-pattern constraints forcing block-diagonal form, and variable reconstruction Z_i = B_{s+2i-1} - B_{s+2i}; both check_constraints branches |
deps: L15, L14 |
targets: sdp_to_standard |
status: not started

L17 | Solver linear-system setup: Frobenius-weighted objective (scale_off_diag), Gram matrix of constraints, regularized Gram solve with relative-residual contract, psd_blocks partition, initial iterates |
deps: L16, L04, L06 |
targets: sdpad (setup section) |
status: not started

L18 | Wen-Goldfarb-Yin iteration: formalize the executable iteration spec (y from (A A^*) y = A(C - S) + penalty (b - A(Z)); W = C - A^*(y) - penalty Z; S = Pi_pos(W) via poseigendecomp thresholds; Z <- (1 - step) Z + (step / penalty)(S - W); penalty updates) and prove each executable iteration matches it |
deps: L17, L12, L07 |
targets: sdpad (iteration section) |
status: not started

L19 | Residuals and KKT quantities: primal residual pinf, dual residual dinf, objective values, duality gap, delta, feasibility check, stagnation/best-solution bookkeeping, final result reconstruction; correspondence with mathematical definitions and tolerance contracts |
deps: L18 |
targets: sdpad (termination section) |
status: not started

L20 | Final audit: full-crate verification, cargo build/test pass, TRUSTED_BASE.md audit of every external_body/assume, VERIFICATION_PROGRESS.md final counts |
deps: L19 |
targets: (all) |
status: not started
