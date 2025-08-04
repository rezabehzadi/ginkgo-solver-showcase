#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    using value_type = double;
    using index_type = int;

    auto exec = gko::ReferenceExecutor::create();

    // Load matrix from file (Matrix Market format)
    auto A = gko::read<gko::matrix::Csr<value_type, index_type>>(std::ifstream("data/A.mtx"), exec);

    // Create RHS and initial guess
    auto b = gko::matrix::Dense<value_type>::create(exec, gko::dim<2>{A->get_size()[0], 1});
    auto x = gko::matrix::Dense<value_type>::create(exec, gko::dim<2>{A->get_size()[1], 1});
    b->fill(1.0);
    x->fill(0.0);

    // Solver: CG or GMRES or BiCGSTAB, etc.
    auto solver = gko::solver::Cg<value_type>::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(100u).on(exec),
            gko::stop::ResidualNormReduction<value_type>::build().with_reduction_factor(1e-6).on(exec))
        .on(exec);

    auto solver_instance = solver->generate(A);
    solver_instance->apply(b, x);

    std::cout << "Solution computed.\n";
}
