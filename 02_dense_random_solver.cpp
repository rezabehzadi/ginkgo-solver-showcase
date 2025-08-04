#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <random>
#include <chrono>  // For timing the solver

int main()
{
    // Create a reference executor (runs on CPU, single-threaded)
    auto exec = gko::ReferenceExecutor::create();
    const int n = 500;  // Size of the linear system (n x n matrix)

    // Random number generator setup for reproducibility
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    // Create matrix A and vectors b, x_cg, x_gmres
    auto A_unique = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, n});
    auto b = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1});
    auto x_cg = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1});
    auto x_gmres = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1});
    x_cg->fill(0.0);    // Initialize solution vectors with zeros
    x_gmres->fill(0.0);

    // Fill matrix A and vector b with random values
    for (int i = 0; i < n; ++i) {
        b->at(i, 0) = dist(gen);
        for (int j = 0; j < n; ++j) {
            A_unique->at(i, j) = dist(gen);
        }
    }

    // Convert unique_ptr matrix A to shared_ptr<const LinOp> as required by solver
    std::shared_ptr<const gko::LinOp> A = std::shared_ptr<const gko::LinOp>(A_unique.release());

    // Build the Conjugate Gradient (CG) solver with stopping criteria
    auto cg_solver = gko::solver::Cg<>::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(100u).on(exec), // Max 100 iterations
                             gko::stop::ResidualNorm<>::build()                            // Stop when residual norm is small enough
                                 .with_baseline(gko::stop::mode::rhs_norm)
                                 .with_reduction_factor(1e-8)
                                 .on(exec))
                         .on(exec);

    // Build the GMRES solver with stopping criteria and Krylov dimension
    auto gmres_solver = gko::solver::Gmres<>::build()
                            .with_criteria(
                                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                                gko::stop::ResidualNorm<>::build()
                                    .with_baseline(gko::stop::mode::rhs_norm)
                                    .with_reduction_factor(1e-8)
                                    .on(exec))
                            .with_krylov_dim(30)  // Krylov subspace dimension for GMRES
                            .on(exec);

    // Generate solver instances for matrix A
    auto cg = cg_solver->generate(A);
    auto gmres = gmres_solver->generate(A);

    // Time and apply CG solver to solve Ax = b
    auto start_cg = std::chrono::high_resolution_clock::now();
    cg->apply(b, x_cg);
    auto end_cg = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cg = end_cg - start_cg;

    // Time and apply GMRES solver to solve Ax = b
    auto start_gmres = std::chrono::high_resolution_clock::now();
    gmres->apply(b, x_gmres);
    auto end_gmres = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gmres = end_gmres - start_gmres;

    // Function to compute relative residual ||b - A*x|| / ||b||
    auto compute_relative_residual = [&](const gko::matrix::Dense<double>* x) {
        auto r = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1});
        A->apply(x, r);  // Compute r = A*x

        // Create scalar -1 to compute residual vector r = A*x - b
        auto neg_one = gko::matrix::Dense<double>::create(exec, gko::dim<2>{1, 1});
        neg_one->at(0, 0) = -1.0;

        r->add_scaled(neg_one, b);  // r = A*x - b

        // Compute norm of residual vector
        auto res_norm = gko::matrix::Dense<double>::create(exec, gko::dim<2>{1, 1});
        r->compute_norm2(res_norm);

        // Compute norm of vector b
        auto b_norm = gko::matrix::Dense<double>::create(exec, gko::dim<2>{1, 1});
        b->compute_norm2(b_norm);

        // Return relative residual norm
        return res_norm->at(0, 0) / b_norm->at(0, 0);
    };

    // Calculate relative residuals for CG and GMRES solutions
    double rel_res_cg = compute_relative_residual(x_cg.get());
    double rel_res_gmres = compute_relative_residual(x_gmres.get());

    // Print relative residuals and solver times
    std::cout << "CG relative residual: " << rel_res_cg << std::endl;
    std::cout << "CG solve time (seconds): " << elapsed_cg.count() << std::endl;

    std::cout << "GMRES relative residual: " << rel_res_gmres << std::endl;
    std::cout << "GMRES solve time (seconds): " << elapsed_gmres.count() << std::endl;

    return 0;
}
