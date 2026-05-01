# Summary

This repository provides a first-order semidefinite programming (SDP) solver with a flexible problem-input format. It is self-contained, implementing its own numerical linear algebra routines. The solver is based on the dual augmented Lagrangian method described in the following paper:

Z. Wen, D. Goldfarb, and W. Yin, “Alternating direction augmented Lagrangian methods for semidefinite programming,” *Mathematical Programming Computation*, 2, 203–230, 2010. [DOI: 10.1007/s12532-010-0017-1](https://doi.org/10.1007/s12532-010-0017-1).

For the paper's convergence guarantees to hold, the user should ensure that both the primal and the dual satisfy Slater's condition. In our implementation, however, if the solver converges on an input, then the returned solution will have small KKT residuals and therefore be optimal up to numerical error. So, in practice, one can simply run the solver on a general SDP problem to see whether it converges to a solution.

## Organization

- `src/lin_alg/`: matrix operations, including Gaussian elimination, `LDL^T` decomposition, and spectral decomposition.
- `src/sdp.rs`: the user-facing SDP struct, helper constructors, and conversion from the user formulation to standard form.
- `src/sdp_altdir.rs`: the SDP solver.
- `src/sdp_tests.rs`: the test harness and example SDP instances.
- `src/main.rs`: a small runner. Replace the toy SDP problem here with the problem you want to solve.

## Running the solver

Edit `src/main.rs` by replacing the toy SDP problem with your desired instance. Then run

```bash
cargo run
```

The test harness can be run with

```bash
cargo test
```

## SDP formulation

Conceptually, an SDP problem is specified by three pieces of data.

### 1. Linear matrix inequalities

A sequence of affine linear matrix inequalities is represented by the data

```text
(H_i, M_i), for i = 1, ..., s.
```

Each `H_i` is built from a list of `n` linear maps

```text
L(i)_1, ..., L(i)_n,
```

where each map has type

```text
L(i)_k : S^{r_k} -> Mat_{k_i}(R).
```

Here `S^{r_k}` is the space of real symmetric `r_k x r_k` matrices. The matrix `M_i` is a square `k_i x k_i` matrix. The pair represents the LMI

```text
H_i(S_1, ..., S_n) >= M_i.
```

Note that the matrices `H_i(S_1, ..., S_n)` and `M_i` are general matrices, not necessarily symmetric.

If the LMI list is empty, then the problem has a single decision variable.

### 2. Linear equality constraints

A single list of `n` linear maps

```text
A_1, ..., A_n
```

and a right-hand side vector `b` represent

```text
A_1(S_1) + ... + A_n(S_n) = b.
```

Each map has type

```text
A_i : S^{r_i} -> R^m.
```

If the LMI list is empty, this list must have at most one element.

### 3. Objective function

The objective is a list of linear maps

```text
C_1, ..., C_n
```

representing the scalar objective

```text
minimize C_1(S_1) + ... + C_n(S_n).
```

If the LMI list is empty, the objective list has exactly one element.

## Conversion to standard form

The user-specified SDP is converted into a standard-form SDP `(C, A, b)`.

The standard-form decision variable `Z` lives in

```text
S_+^{2(r_1 + ... + r_n) + (k_1 + ... + k_s)}.
```

The linear constraints force `Z` to have block-diagonal form

```text
[ B_1              ]
[     ...          ]
[          B_{2n+s}]
```

with

```text
B_i = H_i(B_{s+1} - B_{s+2}, ..., B_{2n+s-1} - B_{2n+s}) - M_i
```

on the upper triangle for each LMI block `i = 1, ..., s`. They also force

```text
L(B_{s+1} - B_{s+2}, ..., B_{2n+s-1} - B_{2n+s}) = b.
```

The implementation checks whether the constraint matrix `A`, excluding the off-block-diagonal zero constraints, has full rank. If it does not, redundant constraints are removed.

The standard-form objective is

```text
C_1(B_{s+1} - B_{s+2}) + ... + C_n(B_{2n+s-1} - B_{2n+s}).
```

The solver computes `Z` and returns the original decision variables

```text
Z_1 = B_{s+1} - B_{s+2}, ..., Z_n = B_{2n+s-1} - B_{2n+s}
```

along with the optimal objective value.

## Implementation details

A linear map

```text
T : S^q -> R^m
```

is represented as a `MatrixSym`, defined in `src/sdp.rs` as

```rust
type MatrixSym = Vec<Vec<Vector>>;
```

This is a list of `m` vectors, each of length `q(q + 1) / 2`. Each vector represents one packed column of the upper-triangular part of a symmetric matrix:

```text
[v(1)_1], [v(2)_1, v(2)_2], ..., [v(q)_1, ..., v(q)_q].
```

For example, a symmetric `2 x 2` matrix is represented in packed form as

```text
[X_11, X_12, X_22]
```

and a symmetric `3 x 3` matrix is represented as

```text
[X_11, X_12, X_22, X_13, X_23, X_33]
```

The following helper constructors are provided in `src/sdp.rs`:

```rust
mk_symmap(sym_dim, output_size, entries)
zero_symmap(sym_dim, output_size)
ident_symmap(dim)
```

The following helper constructors for symmetric-matrix functionals are provided in `src/lin_alg/sym_matrix.rs`:

```rust
sym_matrix_zero(dim)
sym_matrix_trace(dim)
sym_matrix_proj(dim, i, j)
sym_matrix_add(left, right)
sym_matrix_scale(scale, matrix)
mk_symmap_real(dim, entries)
```

The helper `mk_sqmatrix`, defined in `src/lin_alg/matrix.rs`, creates the square `Matrix` used as the constant matrix in an LMI.

A complete SDP problem is specified by

```rust
SDP::new(
    objective,   // MatrixSym
    constraint, // (Vec<MatrixSym>, Vector)
    lmi,        // Vec<(Vec<MatrixSym>, Matrix)>
)
```

## Example problems

### Example 1: two decision variables with equality constraints and PSD constraints

This example has two decision variables: `X1`, a `2 x 2` symmetric matrix, and `X2`, a `3 x 3` symmetric matrix.

```text
minimize (X1_11 + X1_22 - X1_12) + (X2_11 + 2 X2_22 + X2_33 - X2_13)
subject to trace(X1) = 3,
           X1_22 = 1,
           trace(X2) = 4,
           X2_33 = 1,
           X1 >= 0,
           X2 >= 0.
```

```rust
let sdp_test = SDP::new(
    vec![
        mk_symmap_real(2, &vec![1.0, -1.0, 1.0]),
        mk_symmap_real(3, &vec![1.0, 0.0, 2.0, -1.0, 0.0, 1.0]),
    ],
    (
        vec![
            mk_symmap(2, 4, vec![
                1.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ]),
            mk_symmap(3, 4, vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        ],
        vec![3.0, 1.0, 4.0, 1.0],
    ),
    vec![
        (vec![ident_symmap(2), zero_symmap(3, 4)], zero_mat(2, 2)),
        (vec![zero_symmap(2, 9), ident_symmap(3)], zero_mat(3, 3)),
    ],
);
```

### Example 2: no equality constraints

This example shows that the solver can handle a problem with no linear equality constraints.

```text
minimize trace(X)
subject to X >= I.
```

```rust
let sdp_test = SDP::new(
    vec![sym_matrix_trace(2)],
    (vec![], vec![]),
    vec![(
        vec![ident_symmap(2)],
        mk_sqmatrix(2, vec![1.0, 0.0, 0.0, 1.0]),
    )],
);
```

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE.txt) for details.
