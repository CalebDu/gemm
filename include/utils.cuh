//
// Created by Caleb on 2022/6/19.
//

#ifndef GEMM_UTILS_CUH
#define GEMM_UTILS_CUH

#include <fmt/core.h>
#include <sstream>
#include <cublas_v2.h>
#include <functional>

float peak_flops;
float cublas_flops;
namespace util {
    int cc2cores(int major, int minor) {
        typedef struct {
            int SM;
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
                {
                        {0x30, 192},
                        {0x32, 192},
                        {0x35, 192},
                        {0x37, 192},
                        {0x50, 128},
                        {0x52, 128},
                        {0x53, 128},
                        {0x60, 64},
                        {0x61, 128},
                        {0x62, 128},
                        {0x70, 64},
                        {0x72, 64},
                        {0x75, 64},
                        {0x80, 64},
                        {0x86, 128},
                        {-1,   -1}
                };

        int index = 0;

        while (nGpuArchCoresPerSM[index].SM != -1) {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
                return nGpuArchCoresPerSM[index].Cores;
            }

            index++;
        }

//        printf(
//                "MapSMtoCores for SM %d.%d is undefined."
//                "  Default to use %d Cores/SM\n",
//                major, minor, nGpuArchCoresPerSM[index - 1].Cores);
        return nGpuArchCoresPerSM[index - 1].Cores;
    }

    void displayWithFrame(const std::string &info, int width = 50) {
        fmt::print("┌{0:─^{1}}┐\n", "", width);
        std::istringstream ss(info);
        std::string tmp;
        while (std::getline(ss, tmp, '\n')) {
            fmt::print("│{0:<{1}}│\n", tmp, width);
        }
        fmt::print("└{0:─^{1}}┘\n", "", width);
    }

    void initGPU(int gpuId = 1) {
        cudaSetDevice(gpuId);
        cudaDeviceProp info;
        cudaGetDeviceProperties(&info, gpuId);
        auto clock = info.clockRate * 1e3f;
        auto sm = info.multiProcessorCount;
        auto cores = cc2cores(info.major, info.minor);
        auto fp32_flops = cores * sm * clock * 2;
        peak_flops = fp32_flops;
        auto info_str = fmt::format("GPU name: {} \n"
                                    "Compute capability: {}.{}\n"
                                    "GPU SMs: {}\n"
                                    "GPU CUDA cores: {}\n"
                                    "GPU SM clock rate: {:.4f} GLOPS\n"
                                    "FP32 theoretical FLOPS: {:.5e}\n",
                                    info.name,
                                    info.major, info.minor,
                                    sm,
                                    cores * sm,
                                    clock * 1e-9f,
                                    fp32_flops);
        displayWithFrame(info_str);

    }

    struct Handler {
        cublasHandle_t h;

        Handler() {
            cublasCreate_v2(&h);
        }

        ~Handler() {
            cublasDestroy_v2(h);
        }
    };

    template<typename scalar_t>
    void checkAnswer(scalar_t *ans, scalar_t *gt, uint64_t len) {
        double error = 0;
        for (int i = 0; i < len; i++) {
            auto a = gt[i], b = ans[i];
            error += abs(a - b);
        }
        error = error / len;
        fmt::print("computation error: {:.5f}\n", error);
//        printf("computation error: %lf\n", error);
    }

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
        auto mean = total / repeat;
        auto flops = flo * 1e3 / mean;
        if (tag == "cublas_sgemm") {
            cublas_flops = flops;
        }
        fmt::print("{}: {:.5f} ms,"
                   " {:.5e} flops,"
                   " {:.3f}% performance in theoretic flops,"
                   " {:.3f}% performance in cublas flops\n",
                   tag, mean, flops, flops / peak_flops * 100,
                   flops / cublas_flops * 100);
//        std::printf("%s: %f ms, %e flops\n", tag.c_str(), mean, flops);
    }


}
#endif //GEMM_UTILS_CUH
