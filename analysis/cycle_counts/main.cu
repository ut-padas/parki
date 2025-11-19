#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Include rational function evaluation code for erf(x)/x. */
#include "rational_code_xmax6.cu"

/* Measure the error between two vectors d_v1 and d_v2, put the
 * maximum elementwise error in d_out[0]. Stupid non-parallel
 * reduction, but the vectors are not supposed to be that big.
 */
__global__ void kernel_max_error(thrust::device_vector<double> & d_v1,
                                 thrust::device_vector<double> & d_v2,
                                 size_t num_val,
                                 thrust::device_vector<double> & d_out)
{
    double err;
    double max_err = -1;
    for (size_t i=0; i<num_val; i++) {
        err = fabs(d_v1[i] - d_v2[i]);
        max_err = fmax(max_err, err);
    }
    d_out[0] = max_err;
}

/* Macros for calling an operation multiple times. */
#define CALL_1(CALL) CALL;
#define CALL_2(CALL) CALL_1(CALL) CALL_1(CALL)
#define CALL_4(CALL) CALL_2(CALL) CALL_2(CALL)
#define CALL_8(CALL) CALL_4(CALL) CALL_4(CALL)
#define CALL_16(CALL) CALL_8(CALL) CALL_8(CALL)
#define CALL_32(CALL) CALL_16(CALL) CALL_16(CALL)
#define CALL_64(CALL) CALL_32(CALL) CALL_32(CALL)
#define CALL_128(CALL) CALL_64(CALL) CALL_64(CALL)
#define CALL_256(CALL) CALL_128(CALL) CALL_128(CALL)
#define CALL_512(CALL) CALL_256(CALL) CALL_256(CALL)
#define CALL_1024(CALL) CALL_512(CALL) CALL_512(CALL)
#define CALL_2048(CALL) CALL_1024(CALL) CALL_1024(CALL)

/* Macro for a test kernel. */
#define TEST(NAME, CALL) \
    __global__ void NAME(thrust::device_vector<double> & d_t, \
                         thrust::device_vector<double> & d_out, \
                         size_t num_val) \
    { \
        size_t idx = threadIdx.x + blockIdx.x*blockDim.x; \
        if (idx >= num_val) return; \
        double t0 = d_t[idx]; \
        double t = t0; \
        double agg = CALL; \
        double out_part; \
        CALL_2048(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_1024(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_64(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_32(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_16(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_8(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_4(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_2(t += 3.125e-4; out_part = CALL; agg += out_part) \
        CALL_1(t += 3.125e-4; out_part = CALL; agg += out_part) \
        double out = agg * 3.125e-4; /* average */ \
        d_out[idx] = out; \
    }

/* Define the tests. */
TEST(kernel_test_fma, fma(1.53, t, 0.231));
TEST(kernel_test_add, 1.53 + t);
TEST(kernel_test_mul, 1.53 * t);
TEST(kernel_test_div, 1.53 / t);
TEST(kernel_test_sqrt, sqrt(t));
TEST(kernel_test_rsqrt, rsqrt(t));
TEST(kernel_test_expn, exp(-t));
TEST(kernel_test_erf, erf(t));
TEST(kernel_test_erfc, erfc(t));
TEST(kernel_test_sinh, sinh(t));
TEST(kernel_test_i0, cyl_bessel_i0(t));
TEST(kernel_test_j0, j0(t));
TEST(kernel_test_j1, j1(t));
TEST(kernel_test_erfx, erf(t)/t);
TEST(kernel_test_rat_M3, rational_erfoverx_M3(t));
TEST(kernel_test_rat_M4, rational_erfoverx_M4(t));
TEST(kernel_test_rat_M5, rational_erfoverx_M5(t));
TEST(kernel_test_rat_M6, rational_erfoverx_M6(t));
TEST(kernel_test_rat_M7, rational_erfoverx_M7(t));
TEST(kernel_test_rat_M8, rational_erfoverx_M8(t));
TEST(kernel_test_rat_M10, rational_erfoverx_M10(t));
TEST(kernel_test_rat_M12, rational_erfoverx_M12(t));
TEST(kernel_test_rat_M13, rational_erfoverx_M13(t));
TEST(kernel_test_rat_M14, rational_erfoverx_M14(t));

/* Macros for performing a test. */
#define DO_TEST_1(NAME, REPS) \
    for (int r=0; r<REPS; r++) { \
        cudaEventRecord(start); \
        NAME<<<num_blocks, block_size>>>(d_t, d_out1, num_val); \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&milli, start, stop); \
        std::cout << #NAME << " exec time (ms): " << milli \
                  << ", values/sec: " << num_val/(1e-3*milli) << std::endl; \
    }

#define ERFX0 1.1283791670955125739
#define DO_TEST_2(NAME, REPS) \
    for (int r=0; r<REPS; r++) { \
        cudaEventRecord(start); \
        NAME<<<num_blocks, block_size>>>(d_t, d_out2, num_val); \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&milli, start, stop); \
        std::cout << #NAME << " exec time (ms): " << milli \
                  << ", values/sec: " << num_val/(1e-3*milli) << std::endl; \
    } \
    kernel_max_error<<<1, 1>>>(d_out1, d_out2, num_val, d_max_err); \
    h_max_err = d_max_err; \
    std::cout << "  Maximum error: " << h_max_err[0] / ERFX0 << std::endl;

int main(int argc, char* argv[])
{
    int reps = 1;
    for (int i=1; i<argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [-h|--help] [-r|--reps <reps>]" << std::endl
                      << std::endl
                      << "  -h, --help            show this help message and exit" << std::endl
                      << "  -r, --reps <reps>     set number of repeats (default: 1)" << std::endl;
            return 0;
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--reps") == 0) {
            if (i == argc-1) {
                std::cerr << "Error: option " << argv[i] << " requires an argument"
                          << " <reps>" << std::endl;
                return 2;
            }
            reps = atoi(argv[i+1]);
            if (reps <= 0) {
                std::cerr << "Error: argument <reps> must be a positive integer" << std::endl;
                return 2;
            }
        }
    }

    // Initialize CUDA timing
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    size_t num_val = 1048576;
    size_t block_size = 128;
    size_t num_blocks = num_val / block_size;
    if (num_val % block_size != 0) {
        num_blocks++;
    }
    std::cout << "" << num_blocks << " blocks of " << block_size
              << " threads each = " << num_blocks*block_size
              << " threads in total, for " << num_val << " elements."
              << std::endl;
    thrust::device_vector<double> d_out1(num_val);
    thrust::device_vector<double> d_out2(num_val);
    thrust::device_vector<double> d_max_err(1);

    // Allocate memory on host
    thrust::host_vector<double> h_t(num_val);
    thrust::host_vector<double> h_max_err(1);

    // Generate uniform random data between t_min and t_max
    double t_min = 1e-3;
    double t_max = 6 - 1;
    thrust::default_random_engine gen(1923ULL);
    thrust::uniform_real_distribution<double> distr(t_min, t_max);
    cudaEventRecord(start);
    thrust::generate(h_t.begin(), h_t.end(), [&] { return distr(gen); });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Random number generation time (ms): " << milli
              << ", values/sec: " << num_val/(1e-3*milli) << std::endl;

    // Transfer random numbers to device
    cudaEventRecord(start);
    thrust::device_vector<double> d_t = h_t;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Transfer from host to device (ms): " << milli
              << ", values/sec: " << num_val/(1e-3*milli) << std::endl;

    // Perform tests on built-in functions
    DO_TEST_1(kernel_test_fma, reps);
    DO_TEST_1(kernel_test_add, reps);
    DO_TEST_1(kernel_test_mul, reps);
    DO_TEST_1(kernel_test_div, reps);
    DO_TEST_1(kernel_test_sqrt, reps);
    DO_TEST_1(kernel_test_rsqrt, reps);
    DO_TEST_1(kernel_test_expn, reps);
    DO_TEST_1(kernel_test_erf, reps);
    DO_TEST_1(kernel_test_erfc, reps);
    DO_TEST_1(kernel_test_sinh, reps);
    DO_TEST_1(kernel_test_i0, reps);
    DO_TEST_1(kernel_test_j0, reps);
    DO_TEST_1(kernel_test_j1, reps);
    DO_TEST_1(kernel_test_erfx, reps);
    // Perform tests on rational function approximation of erf(x)/x
    DO_TEST_2(kernel_test_rat_M3, reps);
    DO_TEST_2(kernel_test_rat_M4, reps);
    DO_TEST_2(kernel_test_rat_M5, reps);
    DO_TEST_2(kernel_test_rat_M6, reps);
    DO_TEST_2(kernel_test_rat_M7, reps);
    DO_TEST_2(kernel_test_rat_M8, reps);
    DO_TEST_2(kernel_test_rat_M10, reps);
    DO_TEST_2(kernel_test_rat_M12, reps);
    DO_TEST_2(kernel_test_rat_M13, reps);
    DO_TEST_2(kernel_test_rat_M14, reps);

    return 0;
}
