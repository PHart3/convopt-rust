# Verified Semidefinite Programming in Rust

This branch contains a [Verus](https://github.com/verus-lang/verus) verification of the SDP solver on the `main` branch. That solver implements the dual augmented Lagrangian method of Wen, Goldfarb, and Yin, along with its own numerical linear algebra. See the `main` branch README for the problem format and usage.

The code still runs on `f64`. It now carries specifications, loop invariants, and proofs, and changed only where Verus required it or verification exposed a bug. Verus checks the solver's routines against their mathematical specifications, from the linear algebra up to the reported residuals.

## How it was built

We guided Claude (Fable 5.1) to generate the entire verification, overseeing the work, refining the design specifications, and curating the output. The process ran on these files:

- `CLAUDE.md`: the design specification, covering what to prove, the floating-point model, the trusted-base policy, and the workflow rules.
- `prompt.txt`, `run.sh`: the prompt for a single run and the script that launches it. Each run proves the next layers whose dependencies are done, checking each with Verus and `cargo`.
- `DAG.md`: the proof layers, their dependencies, and their status. Claude drafted it in its first run, and we reviewed it before any proofs began.
- `VERIFICATION_PROGRESS.md`: the state passed between runs, including layer status, verified-function counts, and open issues.
- `TRUSTED_BASE.md`: every trusted item, with its justification.
- `EXPOSED_BUGS.md`: bugs the verification uncovered in the original code, written for human review.
- `VERUS_AUDIT_GAPS.md`: Codex's audit of an early version. It found two places where the specifications claimed less than intended. Claude has since closed both, and closing the first exposed a bug in the dual solve.
- `ADDITIONAL_PROOFS.md`: five further statements relating the standard form and its residuals to feasibility of the user's problem, posed and proved by Claude after the main layers.

## What is proved

Primitive floating-point operations are modeled as exact real arithmetic. Under this model, Verus proves the following.

- **Conversion.** The solver's standard form encodes the user's problem faithfully. Feasible points of the user's problem lift to feasible points of the standard form with the same objective value.
- **Linear algebra.** The constraint operator and its adjoint, Gaussian elimination, the `LDL^T` factorization, the triangular solvers, and the PSD check all meet their specifications.
- **Eigensolvers.** Both Jacobi variants share one verified rotation and return `A = Q B Q^T` with `Q^T Q = I`. The cached variant selects the intended pivot, and the cyclic variant visits every pair exactly once per sweep.
- **Iteration.** Each step performs the specified Wen–Goldfarb–Yin update, and the returned solution comes from a single iterate of that trajectory.
- **Residuals.** The reported primal infeasibility, dual infeasibility, and duality gap are exactly the KKT residuals of that iterate.

A passed tolerance test `|x| < eps` establishes only that inequality, not `x = 0`. Likewise, a successful PSD check gives `A >= -eps I`, not `A >= 0`.

Convergence and floating-point error are outside the scope of this verification. The guarantees cover normal returns, not runs that abort on a failed numerical check.

## Trusted base

The exact-real model, a few contracts for standard library functions, and wrappers for two primitives that Verus rejects form the entire trusted base. They live in `src/trusted.rs` and are documented in `TRUSTED_BASE.md`. The code contains no `assume` or `admit`.

## Repository layout

- `src/lin_alg/`: the executable linear algebra (`matrix.rs`, `sym_matrix.rs`, `jacobi.rs`, `comparison.rs`), alongside `*_spec.rs` files of definitions and lemmas.
- `src/sdp.rs`, `src/sdp_altdir.rs`: the problem conversion and the solver, with their proofs.
- `src/sdp_spec.rs`, `src/sdpad_spec.rs`, `src/feas_spec.rs`: specifications of the standard form, the iteration, and the feasibility correspondence.
- `src/trusted.rs`, `src/real_spec.rs`: the trusted model and supporting lemmas about real arithmetic.
- `src/sdp_tests.rs`, `src/main.rs`: the runtime tests and example runner.

The Markdown files at the root are described under [How it was built](#how-it-was-built).

## Checking the proofs

With Verus 0.2026.08.23 installed, run from the repository root:

```bash
verus src/lib.rs --crate-type=lib --edition=2024 -V no-solver-version-check
```

The last full run reported 1046 verified items and 0 errors. The flag skips Verus's Z3 version check, which fails on the version string of the bundled Z3.

The solver still builds and runs as ordinary Rust:

```bash
cargo run
cargo test
```

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE.txt) for details.
