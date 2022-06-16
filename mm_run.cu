#include <iostream>
#include "mm.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
constexpr int M = 1 << 13, N = 1 << 13, K = 1 << 13;

int main() {
    cudaSetDevice(1);
    util::Handler hlr;
    const uint64_t seed = 100;
    const float flo = 2.0f * M * N * K;
    const auto alpha = 1.0f, beta = 0.0f;
    const auto OPA = CUBLAS_OP_N, OPB = CUBLAS_OP_N, OPC = CUBLAS_OP_N;
    const auto lda = M, ldb = K, ldc = M;
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dA({M, K}),
            dB({K, N}), gtC({M, N});
    cutlass::reference::device::TensorFillRandomGaussian(dA.device_view(), seed, 10.0f, 10.0f);
    cutlass::reference::device::TensorFillRandomGaussian(dB.device_view(), seed, 10.0f, 10.0f);


//    dA.sync_host();
//    dB.sync_host();
//    {
//        mm::naive_mm(M, N, K, dA.host_data(), lda, dB.host_data(), ldb,
//                     gtC.host_data(), ldc,alpha, beta);
//    }
//    {
//        util::timer("warm up", flo, [&] {
//            cublasSgemm_v2(hlr.h, OPA, OPB, M, N, K, &alpha,
//                           dA.device_data(), lda, dB.device_data(), ldb, &beta,
//                           dC.device_data(), ldc);
//        });
//    }
    {
        util::timer("gemm_v1", flo, [&] {
            dim3 block(32, 32);
            dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
            mm::gemm_v1<<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
                gtC.device_data(), ldc, alpha, beta);
        });
        gtC.sync_host();
//        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
    }

//    {
//        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
//        util::timer("cublas_sgemm", flo, [&] {
//            cublasSgemm_v2(hlr.h, OPA, OPB, M, N, K, &alpha,
//                           dA.device_data(), lda, dB.device_data(), ldb, &beta,
//                           dC.device_data(), ldc);
//        });
//        dC.sync_host();
//        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
//    }
//
//    {
//        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
//        util::timer("cutlass_sgemm", flo, [&] {
//            mm::cutlass_gemm(M, N, K, dA.device_data(), lda, dB.device_data(), ldb, dC.device_data(),
//                             ldc, alpha, beta);
//        });
//        dC.sync_host();
//        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
//    }
//
//    {
//        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
//        util::timer("gemm_v2", flo, [&] {
//            dim3 block(32, 32);
//            dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
//            mm::gemm_v2<<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
//                dC.device_data(), ldc, alpha, beta);
//        });
//        dC.sync_host();
//        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
//    }
//    {
//        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
//
//        util::timer("gemm_v3", flo, [&] {
//            dim3 block(32, 32);
//            dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
//            mm::gemm_v3<<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
//                dC.device_data(), ldc, alpha, beta);
//        });
//        dC.sync_host();
//        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
//    }
    {
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
        util::timer("gemm_v4", flo, [&] {
            dim3 block(32, 32);
            dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
            mm::gemm_v4<<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
                dC.device_data(), ldc, alpha, beta);
        });
        dC.sync_host();
        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
    }
    {
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
        util::timer("gemm_v5", flo, [&] {
            dim3 block(256);
            dim3 grid((M - 1) / 32 + 1, (N - 1) / 32 + 1);
            mm::gemm_v5<float, float4><<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
                dC.device_data(), ldc, alpha, beta);
        });
        dC.sync_host();
        util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
    }

    return 0;
}