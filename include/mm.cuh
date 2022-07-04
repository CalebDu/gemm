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
#include <fmt/core.h>


// column major general matrix multiply

namespace mm {
#define  IDX2C(i, j, ld) ((i) + (j)*(ld))

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
        typename cutlassGemm::Arguments args(
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
                    c[IDX2C(ii, ji, ldc)] += a[IDX2C(ii, ki, lda)] * b[IDX2C(ki, ji, ldb)];
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
            sum += a[IDX2C(x, i, lda)] * b[IDX2C(i, y, ldb)];
        }
        c[IDX2C(x, y, ldc)] = alpha * sum + beta * c[IDX2C(x, y, ldc)];
    }

    template<typename scalar_t>
    __global__
    __launch_bounds__(1024)
    void gemm_v2_32x32(const int m, const int n, const int k,
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
            accumulate += a[IDX2C(tidx, i, lda)] * b[IDX2C(i, tidy, ldb)];
        }
        c[IDX2C(tidx, tidy, ldc)] = alpha * accumulate + beta * c[IDX2C(tidx, tidy, ldc)];
    }

    template<typename scalar_t>
    __global__
    void gemm_v3_32x32(const int m, const int n, const int k,
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
            smemA[tidx][tidy] = a[IDX2C(tidx, i + tidy, lda)];
            smemB[tidx][tidy] = b[IDX2C(i + tidx, tidy, ldb)];
            __syncthreads();
            for (int j = 0; j < 32; j++) {
                accumulate += smemA[tidx][j] * smemB[j][tidy];
            }
            __syncthreads();
        }
        c[IDX2C(tidx, tidy, ldc)] = alpha * accumulate + beta * c[IDX2C(tidx, tidy, ldc)];
    }

    template<typename scalar_t>
    __global__
    void gemm_v4_32x32(const int m, const int n, const int k,
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
            smemA[tidx][tidy] = a[IDX2C(tidx, i + tidy, lda)];
            smemB[tidx][tidy] = b[IDX2C(i + tidx, tidy, ldb)];
            __syncthreads();
            for (int j = 0; j < 32; j++) {
                accumulate += smemA[tidx][j] * smemB[j][tidy];
            }
            __syncthreads();
        }
        c[IDX2C(tidx, tidy, ldc)] = alpha * accumulate + beta * c[IDX2C(tidx, tidy, ldc)];
    }

    template<typename scalar_t, typename scalarN_t>
    __global__
    __launch_bounds__(256)
    void gemm_v5_32x32(const int m, const int n, const int k,
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

    template<typename scalar_t, typename scalarN_t>
    __global__
    __launch_bounds__(256)
    void gemm_v6_64x64(const int m, const int n, const int k,
                       scalar_t *__restrict__ a, const int lda,
                       scalar_t *__restrict__ b, const int ldb,
                       scalar_t *__restrict__ c, const int ldc,
                       const scalar_t alpha, const scalar_t beta) {
        const auto N = sizeof(scalarN_t) / sizeof(scalar_t);
        auto bidx = blockIdx.x, bidy = blockIdx.y,
                tidx = threadIdx.x;
        auto Ntidx = N * tidx;
        auto xRange = (((int) bidx + 1) << 6) - m,
                yRange = (((int) bidy + 1) << 6) - n;
        if (xRange > 0) {
            a -= xRange;
            c -= xRange;
        }
        if (yRange > 0) {
            b -= yRange * ldb;
            c -= yRange * ldc;
        }
        a += bidx << 6;
        b += (bidy << 6) * ldb;
        c += (bidx << 6) + (bidy << 6) * ldc;
        scalarN_t accumulate[N];
        memset(accumulate, 0, sizeof(accumulate));

        __shared__ scalar_t buff[2048];
        scalar_t *smemA = buff, *smemB = buff + 1024;

        scalarN_t loadA = *(scalarN_t *) &(a[(Ntidx & 63) + (Ntidx >> 6) * lda]),
                loadB = *(scalarN_t *) &(b[(Ntidx & 15) + (Ntidx >> 4) * ldb]);
        for (int i = 0; i < k; i += (1 << 4)) {
            ((scalarN_t *) smemA)[tidx] = loadA;
            for (int x = 0; x < N; x++) {
                smemB[(((Ntidx & 15) + x) << 6) + (Ntidx >> 4)] = ((scalar_t *) &loadB)[x];
            }
            __syncthreads();
            if (i + (1 << 4) < k) {
                a += lda << 4;
                b += (1 << 4);
                loadA = *(scalarN_t *) &(a[(Ntidx & 63) + (Ntidx >> 6) * lda]);
                loadB = *(scalarN_t *) &(b[(Ntidx & 15) + (Ntidx >> 4) * ldb]);
            }
            for (int j = 0; j < 16; j++) {
                scalarN_t tmpA = *(scalarN_t *) &(smemA[(j << 6) + (Ntidx & 63)]),
                        tmpB = *(scalarN_t *) &(smemB[(j << 6) + (tidx >> 4) * N]);
                for (int x = 0; x < N; x++) {
                    for (int y = 0; y < N; y++) {
                        ((scalar_t *) &accumulate[x])[y] +=
                                ((scalar_t *) &tmpA)[y] * ((scalar_t *) &tmpB)[x];
                    }
                }
            }
            __syncthreads();
        }
        for (int x = 0; x < N; x++) {
            scalarN_t *devC = (scalarN_t *) &(c[(tidx & 15) * 4 + ((tidx >> 4) * 4 + x) * ldc]),
                    loadC = *devC;
            for (int y = 0; y < N; y++) {
                ((scalar_t *) &loadC)[y] = alpha * ((scalar_t *) &accumulate[x])[y] +
                                           beta * ((scalar_t *) &loadC)[y];
            }
            *devC = loadC;
        }

    }


    template<typename scalar_t, typename scalarN_t>
    __global__
    __launch_bounds__(256)
    void gemm_v7_128x128(const int m, const int n, const int k,
                         scalar_t *__restrict__ a, const int lda,
                         scalar_t *__restrict__ b, const int ldb,
                         scalar_t *__restrict__ c, const int ldc,
                         const scalar_t alpha, const scalar_t beta) {
        constexpr auto N = sizeof(scalarN_t) / sizeof(scalar_t);
        auto tidx = threadIdx.x, bidx = blockIdx.x,
                bidy = blockIdx.y;
        auto xRange = (((int) bidx + 1) << 7) - m, yRange = (((int) bidy + 1) << 7) - n;
        auto x_a = tidx & 31, y_a = tidx >> 5,
                x_b = tidx & 1, y_b = tidx >> 1,
                x_c = tidx & 15, y_c = tidx >> 4;
        if (xRange > 0) {
            a -= xRange;
            c -= xRange;
        }
        if (yRange > 0) {
            b -= yRange * ldb;
            c -= yRange * ldc;
        }
        a += bidx << 7;
        b += (bidy << 7) * ldb;
        c += (bidx << 7) + (x_c << 2) + ((bidy << 7) + (y_c << 2)) * ldc;
        scalarN_t accumulate[2][2][N];
        memset(accumulate, 0, sizeof(accumulate));

        __shared__ scalar_t buff[2048];
        scalar_t *smemA = buff, *smemB = buff + 1024;

        scalarN_t loadA = ((scalarN_t *) a)[x_a + y_a * (lda >> 2)],
                loadB = ((scalarN_t *) b)[x_b + y_b * (ldb >> 2)];
        for (int ki = 0; ki < k; ki += 8) {
            ((scalarN_t *) smemA)[tidx] = loadA;
            for (int x = 0; x < N; x++) {
                smemB[(((x_b << 2) + x) << 7) + (y_b)] = ((scalar_t *) &loadB)[x];
            }
            __syncthreads();
            if (ki + 8 < k) {
                a += lda << 3;
                b += 8;
                loadA = ((scalarN_t *) a)[x_a + y_a * (lda >> 2)];
                loadB = ((scalarN_t *) b)[x_b + y_b * (ldb >> 2)];
            }

#pragma unroll(4)
            for (int l = 0; l < 8; l++) {
                scalarN_t tmpA[2], tmpB[2];
                for (int i = 0; i < 2; i++) {
                    tmpA[i] = ((scalarN_t *) &(smemA[(l << 7) + (x_c << 2)]))[i << 4];
                }
                for (int i = 0; i < 2; i++) {
                    tmpB[i] = ((scalarN_t *) &(smemB[(l << 7) + (y_c << 2)]))[i << 4];
                }
                for (int x = 0; x < 2; x++) {
                    for (int y = 0; y < 2; y++) {
                        for (int i = 0; i < N; i++) {
                            for (int j = 0; j < N; j++) {
                                ((scalar_t *) &accumulate[x][y][i])[j] += ((scalar_t *) &tmpA[x])[j] *
                                                                          ((scalar_t *) &tmpB[y])[i];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                for (int i = 0; i < N; i++) {
                    scalarN_t *devC = &((scalarN_t *) &c[ldc * (y * 64 + i)])[x << 4],
                            loadC = *devC;
                    for (int j = 0; j < N; j++) {
                        ((scalar_t *) &loadC)[j] = alpha * ((scalar_t *) &accumulate[x][y][i])[j] +
                                                   beta * ((scalar_t *) &loadC)[j];
                    }
                    *devC = loadC;
                }
            }
        }
    }
}

//namespace util {
//    int cc2cores(int major, int minor) {
//        typedef struct {
//            int SM;
//            int Cores;
//        } sSMtoCores;
//
//        sSMtoCores nGpuArchCoresPerSM[] =
//                {
//                        {0x30, 192},
//                        {0x32, 192},
//                        {0x35, 192},
//                        {0x37, 192},
//                        {0x50, 128},
//                        {0x52, 128},
//                        {0x53, 128},
//                        {0x60, 64},
//                        {0x61, 128},
//                        {0x62, 128},
//                        {0x70, 64},
//                        {0x72, 64},
//                        {0x75, 64},
//                        {0x80, 64},
//                        {0x86, 128},
//                        {-1,   -1}
//                };
//
//        int index = 0;
//
//        while (nGpuArchCoresPerSM[index].SM != -1) {
//            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
//                return nGpuArchCoresPerSM[index].Cores;
//            }
//
//            index++;
//        }
//
////        printf(
////                "MapSMtoCores for SM %d.%d is undefined."
////                "  Default to use %d Cores/SM\n",
////                major, minor, nGpuArchCoresPerSM[index - 1].Cores);
//        return nGpuArchCoresPerSM[index - 1].Cores;
//    }
//
//    void displayWithFrame(const std::string &info, int width = 50) {
//        fmt::print("┌{0:─^{1}}┐\n", "", width);
//        std::istringstream ss(info);
//        std::string tmp;
//        while (std::getline(ss, tmp, '\n')) {
//            fmt::print("│{0:<{1}}│\n", tmp, width);
//        }
//        fmt::print("└{0:─^{1}}┘\n", "", width);
//    }
//
//    void initGPU(int gpuId = 1) {
//        cudaSetDevice(gpuId);
//        cudaDeviceProp info;
//        cudaGetDeviceProperties(&info, gpuId);
//        auto clock = info.clockRate * 1e3f;
//        auto sm = info.multiProcessorCount;
//        auto cores = cc2cores(info.major, info.minor);
//        auto fp32_flops = cores * sm * clock * 2;
//        peak_flops = fp32_flops;
//        auto info_str = fmt::format("GPU name: {} \n"
//                                    "Compute capability: {}.{}\n"
//                                    "GPU SMs: {}\n"
//                                    "GPU CUDA cores: {}\n"
//                                    "GPU SM clock rate: {:.4f} GLOPS\n"
//                                    "FP32 theoretical FLOPS: {:.5e}\n",
//                                    info.name,
//                                    info.major, info.minor,
//                                    sm,
//                                    cores * sm,
//                                    clock * 1e-9f,
//                                    fp32_flops);
//        displayWithFrame(info_str);
//
//    }
//
//    struct Handler {
//        cublasHandle_t h;
//
//        Handler() {
//            cublasCreate_v2(&h);
//        }
//
//        ~Handler() {
//            cublasDestroy_v2(h);
//        }
//    };
//    template<typename scalar_t>
//    void checkAnswer(scalar_t *ans, scalar_t *gt, uint64_t len) {
//        double error = 0;
//        for (int i = 0; i < len; i++) {
//            auto a = gt[i], b = ans[i];
//            error += abs(a - b);
//        }
//        error = error / len;
//        fmt::print("computation error: {:.5f}\n", error);
////        printf("computation error: %lf\n", error);
//    }
//    void timer(const std::string &tag, float flo, const std::function<void()> &gemm, int repeat = 10) {
//        double total = 0.0f;
//        for (int _ = 0; _ < repeat; _++) {
//            float time = 0.0f;
//            cudaEvent_t begin, end;
//            cudaEventCreate(&begin);
//            cudaEventCreate(&end);
//            cudaEventRecord(begin);
//            gemm();
//            cudaEventRecord(end);
//            cudaEventSynchronize(begin);
//            cudaEventSynchronize(end);
//            cudaEventElapsedTime(&time, begin, end);
//            cudaEventDestroy(begin);
//            cudaEventDestroy(end);
//            total += time;
//        }
//        auto mean = total / repeat;
//        auto flops = flo * 1e3 / mean;
//        if(tag=="cublas_sgemm") {
//            cublas_flops = flops;
//        }
//        fmt::print("{}: {:.5f} ms,"
//                   " {:.5e} flops,"
//                   " {:.3f}% performance in theoretic flops,"
//                   " {:.3f}% performance in cublas flops\n",
//                   tag, mean, flops,flops/peak_flops*100,
//                   flops/cublas_flops*100);
//
////        std::printf("%s: %f ms, %e flops\n", tag.c_str(), mean, flops);
//
//    }



//}


#endif //GEMM_MM_cUH
