# Verus Verification Project

## Goal

Verify the SDP solver in `convopt-rust` (the current folder) using Verus,
including its core numerical linear algebra routines, while preserving
executable performance.

Work on the `verus` branch. Stage your changes for each run with `git add` 
but do not make commits. Respect the .gitignore.

The objective is to verify that the executable algorithms implement their
mathematical specifications. Do not attempt to prove convergence of the
SDP algorithm.

## 1. Floating-Point Model

Keep executable arithmetic as `f64`.

At the specification level, abstract primitive floating-point operations
as exact-real operations. Treat the correspondence between executable
arithmetic and these specifications as trusted assumptions, not properties
that Verus must prove.

All verification results are relative to these trusted abstractions.

Model tolerance-based comparisons using trusted specifications.

For example, a successful approximate-zero check with tolerance
epsilon establishes |x| < epsilon, not x == 0.

Do not attempt floating-point error analysis at any point in the project.

## 2. Dense Linear Algebra

Verify the numerical linear algebra routines used by the SDP solver.

Follow the existing implementation and data representations. Verify
that the existing linear algebra implementations satisfy their mathematical
specifications.

Develop reusable definitions and lemmas to both support higher-level
proofs and avoid duplicating low-level reasoning.

Do not introduce unrelated linear algebra functionality.

### Note on adjoint computation

The Frobenius inner product counts off-diagonal entries twice.
Consequently, the adjoint `constraint_lin_comb` is not simply the
transpose of the packed coefficient matrix.

Verify its defining adjoint identity:

    <constraint_action(X), y> = <X, constraint_lin_comb(y)>_F

Account for the off-diagonal factors of 1/2 in the implementation.

## 3. Shared Scalar Jacobi Core

Verify one common Jacobi rotation implementation.

For a selected pair p < q, prove:

- J^T J = I;
- A' = J^T A J;
- the selected off-diagonal entry is annihilated;
- symmetry is preserved;
- eigenvalues are preserved (using the similarity invariance of eigenvalues).

Verify accumulation of the eigenvector matrix Q.

Both eigensolvers must share this verified rotation core.

## 4. Cached Scalar Jacobi

Verify the cached Jacobi implementation.

Prove:

- the cache stores the column indices of the row maxima;
- cache updates after rotations are correct;
- the selected pivot is the intended cached maximum.

## 5. Column-Cyclic Scalar Jacobi

Verify the column-cyclic implementation.

For each sweep, prove that every pair

    0 <= p < q < n

is visited exactly once, in column-cyclic order.

Prove that every rotation performed uses the shared verified
Jacobi rotation.

## 6. Eigensolver Specification

For symmetric input A, both eigensolvers must establish:

    Q^T Q = I
    A = Q B Q^T

where B is the transformed symmetric matrix.

On successful termination, verify the implementation's
maximum-off-diagonal stopping criterion and establish the
corresponding approximate eigendecomposition bound.

This bound is an exact mathematical property under the trusted 
floating-point abstraction.

Do not assume that tolerance-based termination produces an exactly
diagonal matrix.

## 7. PSD Spectral Operations

Using the common eigensolver interface, verify:

- PSD detection;
- positive and negative spectral parts;
- PSD projection.

The exact mathematical PSD projection is

    Pi_+(A) = U diag(max(lambda_i, 0)) U^T,

where U and lambda form an exact eigendecomposition of A.

The executable eigensolver instead produces Q and B satisfying

    A = Q B Q^T
    ||B - diag(B)||_F <= epsilon.

Specify and verify the spectral operations actually implemented, including
their tolerance-based eigenvalue selection. In particular,
`nonnegeigendecomp` may retain slightly negative eigenvalues; its
reconstruction must not be assumed positive semidefinite without an
additional mathematical condition.

Prove that the executable reconstruction matches its specification under
the exact-real arithmetic abstraction. Do not require a bound on its
error relative to the exact PSD projection of A.

Do not assume that approximate eigenpairs constitute an exact
eigendecomposition of A.

For `psd_block_check`, verify that a successful check establishes

    A >= -epsilon I,

where epsilon is the implementation's scale-dependent tolerance.

Verify this guarantee from the shifted LDLT factorization and
its pivot conditions. Do not treat a successful check as proof
that A is exactly positive semidefinite.

## 8. SDP Preprocessing

Verify conversion from the public problem representation to the
internal standard-form SDP representation.

Prove preservation of:

- the objective;
- equality constraints;
- semidefinite constraints;
- variable reconstruction.

Also verify constraint reduction and the linear-system setup used
by the solver.

For the equality constraints, verify Gaussian elimination by establishing 
its echelon-form invariants, pivot-selection correctness, and the 
relationship between the output and the input under the implemented 
tolerance-based elimination rules.

Do not require numerical rank to equal exact algebraic rank.

## 9. SDP Iteration

Formalize the mathematical Wen-Goldfarb-Yin iteration implemented
by the solver.

For each executable iteration, prove that the Rust operations
implement their mathematical specifications under the trusted
floating-point abstraction.

Cover:

- equality-system solve;
- dual update;
- PSD projection;
- primal update;
- penalty-related updates.

Specify the executable update using the spectral operation actually
computed by the solver, including its eigenvalue-selection thresholds.
Prove that each update matches this specification under the floating-point
abstraction. Do not require an error bound comparing the update against
the ideal iteration using exact PSD projection.

Do not attempt to prove accumulation bounds across iterations,
global convergence, or convergence rates.

## 10. Residuals and KKT Quantities

Verify computations of:

- primal residual;
- dual residual;
- objective values;
- duality gap;
- feasibility checks;
- complementarity quantities;
- KKT certificate components.

Prove correspondence with their mathematical definitions.

Account for tolerance contracts and approximation bounds where
applicable.

Verification of computed KKT quantities does not require proving
that the algorithm converges to a KKT point.

## 11. Trusted Base

Keep the trusted base small, explicit, and auditable.

Permitted trusted components include:

- exact-real specifications of primitive `f64` operations;
- tolerance and threshold contracts;
- documented contracts for external or unsupported operations
  that cannot reasonably be verified in Verus.

Every use of `external_body`, `assume`, or an equivalent trusted
mechanism must be documented and justified in a clear, concise way.
Place such documentation in TRUSTED_BASE.md, a file that should
be kept concise and up-to-date.

Do not introduce trusted assumptions merely to bypass difficult proofs.

Everything outside the trusted base should be proved.

## 12. Implementation Requirements

### Incremental Verification

Proceed one verification layer at a time, respecting proof dependencies.

Keep the project compiling and existing tests passing.

Prefer small, independently verifiable changes.

Track progress in `VERIFICATION_PROGRESS.md`: layer status (not started / in progress / done),
verified-function count, constructs rewritten for Verus compatibility/verification, and open 
issues. Update it at the end of each session, replacing stale content and maintaining a clear 
and concise writing style. Read this file at session start, creating it if it does not exist 
yet. At the end of each session, remove stale/obsolete content from the file.

If you encounter a genuine error in the original Rust code as a result of trying to verify 
it, briefly explain the error, how Verus exposed it, and how it was fixed in `EXPOSED_BUGS.md`. 
This file is just for human review, so make it highly readable and do not consult it during
your runs.

### Performance Preservation

Preserve executable performance.

Do not replace efficient implementations with slower alternatives
merely to simplify verification.

Use specification-level mathematical views of existing data structures,
along with ghost state, proof functions, and loop invariants as needed.

Avoid runtime verification overhead.

### Verification Architecture

Reuse mathematical specifications and proofs wherever possible.

Both Jacobi eigensolvers must share the verified rotation core
and the same eigensolver specification.

Keep implementation-specific reasoning localized.

Follow existing code architecture unless a change is necessary
for verification.

### Verus Compatibility

- Treat the installed Verus version as authoritative, rather 
  than relying on potentially outdated knowledge or documentation.
- Before declaring a feature unsupported, consult the documentation 
  and test a minimal example against the installed version.
- Do not infer unsupported functionality from a failed verification 
  attempt; first rule out incorrect syntax, missing specifications, 
  and proof failures.
- If a limitation is confirmed, find a reasonable workaround to the 
  Rust code. Otherwise, retain the original Rust code as much as
  possible.

### Workflow

- Always include `-V no-solver-version-check` when invoking Verus.
- The `vstd` source folder is available as a read-only library 
  in `~/verus/source/target-verus/release`.
- Beyond `vstd::prelude::*`, import in each file the specific 
  `vstd` modules you want to use for that file.
- Run all Verus commands from the project root.
- During iteration, use module/function filters (`--verify-module`,
  `--verify-function`; see `verus --help`). Expand scope when changing
  shared definitions, specifications, or dependencies.
- After completing each proof layer, run full verification:
  `verus src/lib.rs --crate-type=lib --edition=2024`
  Fix any failures before proceeding. Report the verified-function count
  in `VERIFICATION_PROGRESS.md` and investigate unexpected decreases.
- Use `--time` to identify slow proofs. Prefer splitting lemmas, adding
  intermediate assertions, and improving triggers over increasing `--rlimit`.
  Raise `--rlimit` only when justified.
- Never weaken specifications, remove meaningful assertions, or introduce
  `assume`/`admit` merely to pass verification.

### Proof Clarity

Keep Verus specifications and proofs idomatic, transparent, concise, and 
conceptually instructive. Prefer clear invariants and reusable lemmas 
over opaque, oversized proofs or unnecessary verification machinery.

Code comments describe the current behavior and specification of the code;
do not record change history or bug fixes in source comments.

### Scope

Verify the existing SDP solver and the numerical routines it uses.

Do not introduce unrelated functionality, redesign numerical
algorithms, or attempt a global convergence proof.

Existing runtime testing of the executable Rust code should be preserved
during this verification project. Don’t break `cargo test`.

### Network Access

Verus is already installed locally.

Use network access as needed to obtain dependencies and to
consult high-quality documentation or academic references.

Do not download or execute untrusted scripts.

### Filesystem Scope

Make source-code, configuration, and other intentional file
modifications only within the current project directory.

Normal external writes by development tools, such as dependency
caches and temporary files, are permitted.

## References

- `README.md`: Project overview and implementation guidance.
- `sdp-rep.pdf`: SDP representation and standard-form conversion.
- `sdpad.pdf`: Wen–Goldfarb–Yin SDP algorithm (do not search internet for this same article).

Use the Rust implementation as the authoritative source for
the algorithms and representations being verified.

Consult the mathematical references as needed, accounting for
differences in notation and data representation.
