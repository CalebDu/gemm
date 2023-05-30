#include "include/mm.cuh"
#include "include/utils.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <iostream>
// #include <cutlass/matrix_shape.h>
#include <fmt/core.h>
#include <numeric>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
constexpr int M = 1 << 13, N = 1 << 13, K = 1 << 13;
// float peak_flops;
// float cublas_flops;

int main() {

  fmt::print("GEMM implement with CUDA\n");
  util::initGPU(0);
  util::Handler hlr;
  const uint64_t seed = 100;
  const float flo = 2.0f * M * N * K;
  const auto alpha = 1.0f, beta = 0.0f;
  const auto OPA = CUBLAS_OP_N, OPB = CUBLAS_OP_N;
  const auto lda = M, ldb = K, ldc = M;
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dA({M, K}),
      dB({K, N}), gtC({M, N});
  cutlass::reference::device::TensorFillRandomGaussian(dA.device_view(), seed,
                                                       5.0f, 1.0f);
  cutlass::reference::device::TensorFillRandomGaussian(dB.device_view(), seed,
                                                       5.0f, 1.0f);
  util::displayWithFrame(
      fmt::format("A shape ({0} x{2})\n"
                  "B shape ({2} x {1})\n"
                  "C = alpha*(AB) + beta*C shape ({0} x {1})\n"
                  "FLO: 2*M*N*K = {3:.5e}\n",
                  M, N, K, flo));

  //    std::printf("A shape (%d x %d)\n"
  //                "B shape (%d x %d) \n"
  //                "C = alpha*(AB) + beta*C shape (%d x %d)\n", M, K, K, N, M,
  //                N);
  //    std::printf("flo: %e\n\n", flo);
  //    dA.sync_host();
  //    dB.sync_host();
  //    {
  //        mm::naive_mm(M, N, K, dA.host_data(), lda, dB.host_data(), ldb,
  //                     gtC.host_data(), ldc,alpha, beta);
  //    }
  //    {
  //        util::timer("warm up", flo, [&] {
  //            cublasSgemm_v2(hlr.h, OPA, OPB, M, N, K, &alpha,
  //                           dA.device_data(), lda, dB.device_data(), ldb,
  //                           &beta, dC.device_data(), ldc);
  //        });
  //    }

  mm::cutlass_gemm(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
                   gtC.device_data(), ldc, alpha, beta);
  gtC.sync_host();

  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("cublas_sgemm", flo, [&] {
      cublasSgemm_v2(hlr.h, OPA, OPB, M, N, K, &alpha, dA.device_data(), lda,
                     dB.device_data(), ldb, &beta, dC.device_data(), ldc);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }

  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("cutlass_sgemm", flo, [&] {
      mm::cutlass_gemm(M, N, K, dA.device_data(), lda, dB.device_data(), ldb,
                       dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }
  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("gemm_v1", flo, [&] {
      dim3 block(32, 32);
      dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
      mm::gemm_v1<<<grid, block>>>(M, N, K, dA.device_data(), lda,
                                   dB.device_data(), ldb, dC.device_data(), ldc,
                                   alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }
  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("gemm_v2", flo, [&] {
      dim3 block(32, 32);
      dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
      mm::gemm_v2_32x32<<<grid, block>>>(M, N, K, dA.device_data(), lda,
                                         dB.device_data(), ldb,
                                         dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }
  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});

    util::timer("gemm_v3", flo, [&] {
      dim3 block(32, 32);
      dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
      mm::gemm_v3_32x32<<<grid, block>>>(M, N, K, dA.device_data(), lda,
                                         dB.device_data(), ldb,
                                         dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }
  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("gemm_v4", flo, [&] {
      dim3 block(32, 32);
      dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
      mm::gemm_v4_32x32<<<grid, block>>>(M, N, K, dA.device_data(), lda,
                                         dB.device_data(), ldb,
                                         dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }
  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("gemm_v5_32x32", flo, [&] {
      dim3 block(256);
      constexpr int Tile = 32;
      dim3 grid((M - 1) / Tile + 1, (N - 1) / Tile + 1);
      mm::gemm_v5_32x32<float, float4>
          <<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(),
                            ldb, dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }
  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("gemm_v6_64x64", flo, [&] {
      dim3 block(256);
      constexpr int Tile = 64;
      dim3 grid((M - 1) / Tile + 1, (N - 1) / Tile + 1);
      mm::gemm_v6_64x64<float, float4>
          <<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(),
                            ldb, dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }

  {
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> dC({M, N});
    util::timer("gemm_v7_128x128", flo, [&] {
      dim3 block(256);
      constexpr int Tile = 128;
      dim3 grid((M - 1) / Tile + 1, (N - 1) / Tile + 1);
      mm::gemm_v7_128x128<float, float4>
          <<<grid, block>>>(M, N, K, dA.device_data(), lda, dB.device_data(),
                            ldb, dC.device_data(), ldc, alpha, beta);
    });
    dC.sync_host();
    util::checkAnswer(dC.host_data(), gtC.host_data(), M * N);
  }

  return 0;
}