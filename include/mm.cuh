//
// created by caleb on 2022/5/24.
//

#ifndef GEMM_MM_cUH
#define GEMM_MM_cUH

#include <functional>
#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <chrono>
// column major general matrix multiply
namespace mm {
#define  IDX2c(i, j, ld) ((i) + (j)*(ld))

    template<typename scalar_t>
    cudaError_t cutlass_gemm(const int m, const int n, const int k,
                             scalar_t *__restrict__ a, const int lda,
                             scalar_t *__restrict__ b, const int ldb,
                             scalar_t *__restrict__ c, const int ldc,
                             const scalar_t alpha, const scalar_t beta) {

        using columnMajor = cutlass::layout::ColumnMajor;
        using cutlassGemm = cutlass::gemm::device::Gemm<scalar_t,
                columnMajor, scalar_t, columnMajor, scalar_t, columnMajor>;
        cutlassGemm op;
        typename cutlassGemm::arguments args(
                {m, n, k},
                {a, lda},
                {b, ldb},
                {c, ldc},
                {c, ldc},
                {alpha, beta}
        );
        cutlass::Status sta = op(args);
        return sta == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
    }

    template<typename scalar_t>
    void naive_mm(const int m, const int n, const int k,
                  scalar_t *__restrict__ a, const int lda,
                  scalar_t *__restrict__ b, const int ldb,
                  scalar_t *__restrict__ c, const int ldc,
                  const scalar_t alpha, const scalar_t beta) {
        for (int ki = 0; ki < k; ki++) {
            for (int ji = 0; ji < n; ji++) {
                for (int ii = 0; ii < m; ii++) {
//                    c[ii * m + ji] += a[ii * k + ki] * b[ki * m + ji];
                    c[IDX2c(ii, ji, ldc)] += a[IDX2c(ii, ki, lda)] * b[IDX2c(ki, ji, ldb)];
                }
            }
        }
    }

    template<typename scalar_t>
    __global__
    void gemm_v1(const int m, const int n, const int k,
                 scalar_t *__restrict__ a, const int lda,
                 scalar_t *__restrict__ b, const int ldb,
                 scalar_t *__restrict__ c, const int ldc,
                 const scalar_t alpha, const scalar_t beta) {
        scalar_t sum = 0;
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;
        // check boundary
        if (x >= m || y >= n)
            return;
        for (int i = 0; i < k; i++) {
            sum += a[IDX2c(x, i, lda)] * b[IDX2c(i, y, ldb)];
        }
        c[IDX2c(x, y, ldc)] = alpha * sum + beta * c[IDX2c(x, y, ldc)];
    }

    template<typename scalar_t>
    __global__
    __launch_bounds__(1024)
    void gemm_v2(const int m, const int n, const int k,
                 scalar_t *__restrict__ a, const int lda,
                 scalar_t *__restrict__ b, const int ldb,
                 scalar_t *__restrict__ c, const int ldc,
                 const scalar_t alpha, const scalar_t beta) {
        auto bidx = blockIdx.x, bidy = blockIdx.y,
                tidx = threadIdx.x, tidy = threadIdx.y;
//        auto x = bidx * blockDim.x + tidx, y = bidy * blockDim.y + tidy;
        auto xRange = (((int) bidx + 1) << 5) - m, yRange = (((int) bidy + 1) << 5) - n;
        if (xRange > 0) {
            a -= xRange;
            c -= xRange;
        }
        if (yRange > 0) {
            b -= yRange * ldb;
            c -= yRange * ldc;
        }
        a += bidx << 5;
        b += (bidy << 5) * ldb;
        c += (bidx << 5) + (bidy << 5) * ldc;
        scalar_t accumulate = static_cast<scalar_t>(0);
        for (int i = 0; i < k; i++) {
            accumulate += a[IDX2c(tidx, i, lda)] * b[IDX2c(i, tidy, ldb)];
        }
        c[IDX2c(tidx, tidy, ldc)] = alpha * accumulate + beta * c[IDX2c(tidx, tidy, ldc)];
    }

    template<typename scalar_t>
    __global__
    void gemm_v3(const int m, const int n, const int k,
                 scalar_t *__restrict__ a, const int lda,
                 scalar_t *__restrict__ b, const int ldb,
                 scalar_t *__restrict__ c, const int ldc,
                 const scalar_t alpha, const scalar_t beta) {
        auto bidx = blockIdx.x, bidy = blockIdx.y,
                tidx = threadIdx.x, tidy = threadIdx.y;
//        auto x = bidx * blockDim.x + tidx, y = bidy * blockDim.y + tidy;
        auto xRange = (((int) bidx + 1) << 5) - m, yRange = (((int) bidy + 1) << 5) - n;
        if (xRange > 0) {
            a -= xRange;
            c -= xRange;
        }
        if (yRange > 0) {
            b -= yRange * ldb;
            c -= yRange * ldc;
        }
        a += bidx << 5;
        b += (bidy << 5) * ldb;
        c += (bidx << 5) + (bidy << 5) * ldc;
        scalar_t accumulate = 0.0f;
        __shared__ scalar_t smemA[32][32], smemB[32][32];
        for (int i = 0; i < k; i += 32) {
            smemA[tidx][tidy] = a[IDX2c(tidx, i + tidy, lda)];
            smemB[tidx][tidy] = b[IDX2c(i + tidx, tidy, ldb)];
            __syncthreads();
            for (int j = 0; j < 32; j++) {
                accumulate += smemA[tidx][j] * smemB[j][tidy];
            }
            __syncthreads();
        }
        c[IDX2c(tidx, tidy, ldc)] = alpha * accumulate + beta * c[IDX2c(tidx, tidy, ldc)];
    }

    template<typename scalar_t>
    __global__
    void gemm_v4(const int m, const int n, const int k,
                 scalar_t *__restrict__ a, const int lda,
                 scalar_t *__restrict__ b, const int ldb,
                 scalar_t *__restrict__ c, const int ldc,
                 const scalar_t alpha, const scalar_t beta) {
        auto bidx = blockIdx.x, bidy = blockIdx.y,
                tidx = threadIdx.x, tidy = threadIdx.y;
//        auto x = bidx * blockDim.x + tidx, y = bidy * blockDim.y + tidy;
        auto xRange = (((int) bidx + 1) << 5) - m, yRange = (((int) bidy + 1) << 5) - n;
        if (xRange > 0) {
            a -= xRange;
            c -= xRange;
        }
        if (yRange > 0) {
            b -= yRange * ldb;
            c -= yRange * ldc;
        }
        a += bidx << 5;
        b += (bidy << 5) * ldb;
        c += (bidx << 5) + (bidy << 5) * ldc;
        scalar_t accumulate = 0.0f;
        __shared__ scalar_t smemA[32][32 | 1], smemB[32][32 | 1];
        for (int i = 0; i < k; i += 32) {
            smemA[tidx][tidy] = a[IDX2c(tidx, i + tidy, lda)];
            smemB[tidx][tidy] = b[IDX2c(i + tidx, tidy, ldb)];
            __syncthreads();
            for (int j = 0; j < 32; j++) {
                accumulate += smemA[tidx][j] * smemB[j][tidy];
            }
            __syncthreads();
        }
        c[IDX2c(tidx, tidy, ldc)] = alpha * accumulate + beta * c[IDX2c(tidx, tidy, ldc)];
    }

    template<typename scalar_t, typename scalarN_t>
    __global__
    __launch_bounds__(256)
    void gemm_v5(const int m, const int n, const int k,
                 scalar_t *__restrict__ a, const int lda,
                 scalar_t *__restrict__ b, const int ldb,
                 scalar_t *__restrict__ c, const int ldc,
                 const scalar_t alpha, const scalar_t beta) {
        const int N = sizeof(scalarN_t) / sizeof(scalar_t), idx = threadIdx.x,
                x_range = ((int) (blockIdx.x + 1) << 5) - m,
                y_range = ((int) (blockIdx.y + 1) << 5) - n;
        if (x_range > 0) {
            a -= x_range;
            c -= x_range;
        }
        if (y_range > 0) {
            b -= y_range * ldb;
            c -= y_range * ldc;
        }
        a += blockIdx.x << 5;
        b += (blockIdx.y << 5) * ldb;
        c += (blockIdx.x << 5) + (blockIdx.y << 5) * ldc;
        scalarN_t ansc;
        memset(&ansc, 0, sizeof(ansc));

        __shared__ scalar_t smemBuffer[2048];
        scalar_t *smemA = smemBuffer;
        scalar_t *smemB = smemBuffer + 1024;

        scalarN_t loada = *(scalarN_t *) &(a[((idx * N) & 31) + ((idx * N) >> 5) * lda]),
                loadb = *(scalarN_t *) &(b[((idx * N) & 31) + ((idx * N) >> 5) * ldb]);

        for (int l = 0; l < k; l += 32) {
            ((scalarN_t *) smemA)[idx] = loada;
            for (int i = 0; i < N; ++i)
                smemB[((((idx * N) & 31) + i) << 5) + ((idx * N) >> 5)] =
                        ((scalar_t *) &loadb)[i];
            __syncthreads();

            if (l + 32 < k) {
                a += lda << 5;
                b += 32;
                loada =
                        *(scalarN_t *) &(a[((idx * N) & 31) + ((idx * N) >> 5) * lda]);
                loadb =
                        *(scalarN_t *) &(b[((idx * N) & 31) + ((idx * N) >> 5) * ldb]);
            }

            for (int j = 0; j < 32; ++j) {
                scalarN_t tmpA = *(scalarN_t *) &smemA[((idx * N) & 31) + (j << 5)];
                scalar_t tmpB = smemB[(j << 5) + ((idx * N) >> 5)];
                for (int i = 0; i < N; ++i)
                    ((scalar_t *) &ansc)[i] += ((scalar_t *) &tmpA)[i] * tmpB;
            }
            __syncthreads();

        }
        {
            scalarN_t *devC =
                    (scalarN_t *) &(c[((idx * N) & 31) + ((idx * N) >> 5) * ldc]),
                    resC = *devC;
            for (int i = 0; i < N; ++i)
                ((scalar_t *) &resC)[i] = alpha * ((scalar_t *) &ansc)[i] + beta * ((scalar_t *) &resC)[i];
            *devC = resC;
        }
    }

}

namespace util {
    struct Handler {
        cublasHandle_t h;

        Handler() {
            cublasCreate_v2(&h);
        }

        ~Handler() {
            cublasDestroy_v2(h);
        }
    };

    void timer(const std::string &tag, float flo, const std::function<void()> &gemm, int repeat = 10) {
        double total = 0.0f;
        for (int _ = 0; _ < repeat; _++) {
            float time = 0.0f;
            cudaEvent_t begin, end;
            cudaEventCreate(&begin);
            cudaEventCreate(&end);
            cudaEventRecord(begin);
            gemm();
            cudaEventRecord(end);
            cudaEventSynchronize(begin);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&time, begin, end);
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            total += time;
        }
        auto meanscalar_time = total / repeat;
        auto flops = flo * 1e3 / meanscalar_time;

        std::printf("%s: %f ms, %e flo, %e flops\n", tag.c_str(), meanscalar_time, flo, flops);
    }

    template<typename scalar_t>
    void checkAnswer(scalar_t *ans, scalar_t *gt, uint64_t len) {
        double error = 0;
        for (int i = 0; i < len; i++) {
            auto a = gt[i], b = ans[i];
            error += abs(a - b);
        }
        error = error / len;
        printf("error: %lf\n", error);
    }


}


#endif //GEMM_MM_cUH
