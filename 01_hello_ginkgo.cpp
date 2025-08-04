#include <ginkgo/ginkgo.hpp>
#include <iostream>

int main()
{
    std::cout << "Checking available Ginkgo executors:\n";

    auto ref_exec = gko::ReferenceExecutor::create();
    std::cout << "- ReferenceExecutor: available\n";

    try
    {
        auto omp_exec = gko::OmpExecutor::create();
        std::cout << "- OmpExecutor: available\n";
    }
    catch (...)
    {
        std::cout << "- OmpExecutor: NOT available\n";
    }

    if (gko::CudaExecutor::get_num_devices() > 0)
    {
        try
        {
            auto cuda_exec = gko::CudaExecutor::create(0, ref_exec);
            std::cout << "- CudaExecutor: available\n";
        }
        catch (...)
        {
            std::cout << "- CudaExecutor: NOT available\n";
        }
    }
    else
    {
        std::cout << "- CudaExecutor: NO CUDA devices found\n";
    }

#ifdef GKO_HAVE_HIP
    if (gko::HipExecutor::get_num_devices() > 0)
    {
        try
        {
            auto hip_exec = gko::HipExecutor::create(0, ref_exec);
            std::cout << "- HipExecutor: available\n";
        }
        catch (...)
        {
            std::cout << "- HipExecutor: NOT available\n";
        }
    }
    else
    {
        std::cout << "- HipExecutor: NO HIP devices found\n";
    }
#else
    std::cout << "- HipExecutor: NOT compiled in\n";
#endif

#ifdef GKO_HAVE_DPCPP
    try
    {
        auto dpcpp_exec = gko::DpcppExecutor::create(0, ref_exec);
        std::cout << "- DpcppExecutor: available\n";
    }
    catch (...)
    {
        std::cout << "- DpcppExecutor: NOT available\n";
    }
#else
    std::cout << "- DpcppExecutor: NOT compiled in\n";
#endif

    return 0;
}
