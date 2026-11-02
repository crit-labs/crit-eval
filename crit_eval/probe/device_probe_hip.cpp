// crit eval device probe for ROCm (HIP): the measurements of device_probe.cu on an AMD GPU, built with hipcc by
// `crit eval probe --backend hip` and printing one JSON object on stdout in the same shape.
//
// Memory, copy bandwidth, the host-to-device and device-to-host curves and the host's memory and CPU rates are
// measured exactly as device_probe.cu measures them (the shared code is the same apart from the runtime's names).
// GEMM: hipBLASLt with FP32 accumulation at the same prefill-like shapes, best rate per format: BF16, and FP8 E4M3
// where the architecture has FP8 matrix instructions (the FNUZ encoding on gfx94x, the OCP one on gfx95x and gfx12xx).
// No raw matrix-instruction rates yet (mma_tflops is empty): MFMA and WMMA differ per architecture, so the GEMM
// rate is the ceiling. unsupported_formats lists the rate keys this architecture has no matrix units for, which
// crit_eval.limits.ceiling.best_rates prices at the BF16 rate: NVFP4 always (AMD's FP4 is MXFP4, with E8M0 scales per
// 32), FP8 before gfx94x and on RDNA before gfx12xx.
//
// Not yet run on hardware: written against the HIP and hipBLASLt APIs of ROCm 6.4.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <atomic>
#include <immintrin.h>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

__global__ void fill_random(uint8_t* data, size_t bytes, uint32_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < bytes; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = uint32_t(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        data[i] = uint8_t(h);
    }
}

void fill(void* data, size_t bytes, uint32_t seed) { fill_random<<<1024, 256>>>(static_cast<uint8_t*>(data), bytes, seed); }

double copy_bandwidth_gbs(size_t bytes, int batches, int copies) {
    void *src = nullptr, *dst = nullptr;
    if (hipMalloc(&src, bytes) != hipSuccess || hipMalloc(&dst, bytes) != hipSuccess) return 0.0;
    hipMemset(src, 1, bytes);
    hipMemset(dst, 0, bytes);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);  // warmup
    double total = 0.0;
    for (int b = 0; b < batches; ++b) {
        hipEventRecord(start);
        for (int i = 0; i < copies; ++i) hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        total += 2.0 * bytes * copies / (ms * 1e-3) / 1e9;
    }
    hipFree(src);
    hipFree(dst);
    return total / batches;
}

__global__ void read_mapped(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t words) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

// GB/s moving `bytes` between host and device: kind 0 pinned copy, 1 pageable copy, 2 kernel read of mapped
// pinned memory (host to device only); `to_host` copies the other way. 0 when an allocation, a copy or a launch
// fails, or a batch times as zero; the error is cleared so later measurements do not see it.
double transfer_gbs(int kind, size_t bytes, int batches, int copies, bool to_host = false) {
    void* host = nullptr;
    void* device = nullptr;
    if (hipMalloc(&device, bytes) != hipSuccess) { hipGetLastError(); return 0.0; }
    const bool pageable = kind == 1;
    if (pageable) host = std::malloc(bytes);
    else if (hipHostMalloc(&host, bytes, kind == 2 ? hipHostMallocMapped : hipHostMallocDefault) != hipSuccess) host = nullptr;
    if (!host) { hipGetLastError(); hipFree(device); return 0.0; }
    std::memset(host, 1, bytes);
    void* alias = nullptr;
    if (kind == 2 && hipHostGetDevicePointer(&alias, host, 0) != hipSuccess) alias = nullptr;
    bool failed = false;
    auto move = [&] {
        if (kind == 2) read_mapped<<<1024, 256>>>(static_cast<const uint4*>(alias), static_cast<uint4*>(device), bytes / sizeof(uint4));
        else if ((to_host ? hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost)
                          : hipMemcpyAsync(device, host, bytes, hipMemcpyHostToDevice)) != hipSuccess) failed = true;
        if (hipPeekAtLastError() != hipSuccess) failed = true;
    };
    double total = 0.0;
    if (kind != 2 || alias) {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        move();  // warmup
        hipDeviceSynchronize();
        for (int b = 0; b < batches; ++b) {
            hipEventRecord(start);
            for (int i = 0; i < copies; ++i) move();
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            float ms = 0.f;
            if (hipEventElapsedTime(&ms, start, stop) != hipSuccess || !(ms > 0.f)) failed = true;
            if (failed) break;
            total += double(bytes) * copies / (ms * 1e-3) / 1e9;
        }
        hipEventDestroy(start);
        hipEventDestroy(stop);
    } else {
        failed = true;
    }
    if (failed) { hipGetLastError(); total = 0.0; }
    if (pageable) std::free(host);
    else hipHostFree(host);
    hipFree(device);
    return total / batches;
}

using Clock = std::chrono::steady_clock;

// Runs work(thread index) on `threads` threads started together; seconds from the start to the last one done.
template <class Work>
double timed_threads(int threads, Work work) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            work(t);
        });
    while (ready.load() < threads) std::this_thread::yield();
    const auto start = Clock::now();
    go.store(true);
    for (auto& thread : pool) thread.join();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

volatile uint64_t g_sink = 0;   // keeps reads and sums from being optimized away

// Host memory GB/s over `bytes`-sized buffers: kind 0 read, 1 write, 2 copy (read plus written bytes), each
// thread on its own slice, best of `passes`.
double host_memory_gbs(int kind, size_t bytes, int threads, int passes) {
    std::vector<uint64_t> a(bytes / 8, 1), b(kind == 2 ? bytes / 8 : 0, 0);
    const size_t words = a.size(), slice = words / threads;
    double best = 0.0;
    for (int pass = 0; pass < passes; ++pass) {
        const double seconds = timed_threads(threads, [&](int t) {
            uint64_t* begin = a.data() + t * slice;
            const size_t n = t + 1 == threads ? words - t * slice : slice;
            if (kind == 0) {
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t i = 0; i + 4 <= n; i += 4) {
                    s0 += begin[i];
                    s1 += begin[i + 1];
                    s2 += begin[i + 2];
                    s3 += begin[i + 3];
                }
                g_sink += s0 + s1 + s2 + s3;
            } else if (kind == 1) {
                std::memset(begin, pass + 2, n * 8);
            } else {
                std::memcpy(b.data() + t * slice, begin, n * 8);
            }
        });
        const double moved = double(bytes) * (kind == 2 ? 2 : 1);
        if (seconds > 0) best = std::max(best, moved / seconds / 1e9);
    }
    return best;
}

// One thread's FP32 fused multiply-adds: kChains independent vector chains, so every FMA unit stays busy, in the
// widest vectors this build targets (AVX-512, else AVX2 with FMA; plain floats elsewhere). Returns FLOP done.
constexpr int kChains = 12;
constexpr long kSteps = long(1) << 23;
#if defined(__AVX512F__)
constexpr const char* kIsa = "avx512";
double fma_chains(int seed) {
    __m512 acc[kChains];
    for (int j = 0; j < kChains; ++j) acc[j] = _mm512_set1_ps(1.0f + 1e-3f * (j + seed));
    const __m512 m = _mm512_set1_ps(0.9999999f), c = _mm512_set1_ps(1e-7f);
    for (long i = 0; i < kSteps; ++i) {   // written out, so the chains stay in registers under any compiler
        acc[0] = _mm512_fmadd_ps(acc[0], m, c); acc[1] = _mm512_fmadd_ps(acc[1], m, c); acc[2] = _mm512_fmadd_ps(acc[2], m, c);
        acc[3] = _mm512_fmadd_ps(acc[3], m, c); acc[4] = _mm512_fmadd_ps(acc[4], m, c); acc[5] = _mm512_fmadd_ps(acc[5], m, c);
        acc[6] = _mm512_fmadd_ps(acc[6], m, c); acc[7] = _mm512_fmadd_ps(acc[7], m, c); acc[8] = _mm512_fmadd_ps(acc[8], m, c);
        acc[9] = _mm512_fmadd_ps(acc[9], m, c); acc[10] = _mm512_fmadd_ps(acc[10], m, c); acc[11] = _mm512_fmadd_ps(acc[11], m, c);
    }
    float sum = 0.0f;
    for (int j = 0; j < kChains; ++j) sum += _mm512_reduce_add_ps(acc[j]);
    g_sink += uint64_t(sum);
    return 2.0 * 16 * kChains * kSteps;
}
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
constexpr const char* kIsa = "avx2";
double fma_chains(int seed) {
    __m256 acc[kChains];
    for (int j = 0; j < kChains; ++j) acc[j] = _mm256_set1_ps(1.0f + 1e-3f * (j + seed));
    const __m256 m = _mm256_set1_ps(0.9999999f), c = _mm256_set1_ps(1e-7f);
    for (long i = 0; i < kSteps; ++i) {   // written out, so the chains stay in registers under any compiler
        acc[0] = _mm256_fmadd_ps(acc[0], m, c); acc[1] = _mm256_fmadd_ps(acc[1], m, c); acc[2] = _mm256_fmadd_ps(acc[2], m, c);
        acc[3] = _mm256_fmadd_ps(acc[3], m, c); acc[4] = _mm256_fmadd_ps(acc[4], m, c); acc[5] = _mm256_fmadd_ps(acc[5], m, c);
        acc[6] = _mm256_fmadd_ps(acc[6], m, c); acc[7] = _mm256_fmadd_ps(acc[7], m, c); acc[8] = _mm256_fmadd_ps(acc[8], m, c);
        acc[9] = _mm256_fmadd_ps(acc[9], m, c); acc[10] = _mm256_fmadd_ps(acc[10], m, c); acc[11] = _mm256_fmadd_ps(acc[11], m, c);
    }
    alignas(32) float lanes[8];
    float sum = 0.0f;
    for (int j = 0; j < kChains; ++j) {
        _mm256_store_ps(lanes, acc[j]);
        for (float lane : lanes) sum += lane;
    }
    g_sink += uint64_t(sum);
    return 2.0 * 8 * kChains * kSteps;
}
#else
constexpr const char* kIsa = "scalar";
double fma_chains(int seed) {
    float acc[kChains * 8];
    for (int j = 0; j < kChains * 8; ++j) acc[j] = 1.0f + 1e-3f * (j + seed);
    for (long i = 0; i < kSteps / 8; ++i)
        for (int j = 0; j < kChains * 8; ++j) acc[j] = acc[j] * 0.9999999f + 1e-7f;
    float sum = 0.0f;
    for (float a : acc) sum += a;
    g_sink += uint64_t(sum);
    return 2.0 * kChains * 8 * (kSteps / 8);
}
#endif

// CPU FP32 GFLOPS of fma_chains on `threads` threads, best of `passes`.
double cpu_gflops(int threads, int passes) {
    double best = 0.0;
    for (int pass = 0; pass < passes; ++pass) {
        std::atomic<double> flop{0.0};
        const double seconds = timed_threads(threads, [&](int t) {
            const double done = fma_chains(t);
            double seen = flop.load();
            while (!flop.compare_exchange_weak(seen, seen + done)) {}
        });
        if (seconds > 0) best = std::max(best, flop.load() / seconds / 1e9);
    }
    return best;
}

// FP8 matrix instructions on this architecture: (has FP8, FNUZ encoding). gfx94x (MI300) uses E4M3FNUZ; gfx95x
// (MI350) and RDNA4 (gfx12xx) the OCP E4M3 that checkpoints store.
std::pair<bool, bool> fp8_support(const std::string& arch) {
    if (arch.rfind("gfx94", 0) == 0) return {true, true};
    if (arch.rfind("gfx95", 0) == 0 || arch.rfind("gfx12", 0) == 0) return {true, false};
    return {false, false};
}

// TFLOPS of D[N x M] = W[N x K] * X[K x M] (W row-major, op T), or 0 when unsupported.
double gemm_tflops(hipblasLtHandle_t lt, hipDataType type, int M, int N, int K) {
    const size_t element_bits = (type == HIP_R_8F_E4M3 || type == HIP_R_8F_E4M3_FNUZ) ? 8 : 16;
    const size_t wbytes = size_t(N) * K * element_bits / 8, xbytes = size_t(M) * K * element_bits / 8;
    const size_t ws_bytes = 64ull << 20;
    void *w = nullptr, *x = nullptr, *d = nullptr, *ws = nullptr;
    double result = 0.0;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulHeuristicResult_t heuristics[8];
    int found = 0;
    const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    const float one = 1.f, zero = 0.f;
    hipEvent_t start, stop;
    if (hipMalloc(&w, wbytes) || hipMalloc(&x, xbytes) || hipMalloc(&d, size_t(M) * N * 2) || hipMalloc(&ws, ws_bytes))
        goto done;
    fill(w, wbytes, 1);
    fill(x, xbytes, 2);
    if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F)) goto done;
    hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    if (hipblasLtMatrixLayoutCreate(&a, type, K, N, K) || hipblasLtMatrixLayoutCreate(&b, type, K, M, K) ||
        hipblasLtMatrixLayoutCreate(&c, HIP_R_16BF, N, M, N))
        goto done;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes);
    if (hipblasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, 8, heuristics, &found) || !found) goto done;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    for (int h = 0; h < found; ++h) {  // best of the offered algorithms
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
            ok = !hipblasLtMatmul(lt, op, &one, w, a, x, b, &zero, d, c, d, c, &heuristics[h].algo, ws, ws_bytes, 0);
        if (!ok) continue;
        const int reps = 20;
        hipEventRecord(start);
        for (int i = 0; i < reps; ++i)
            hipblasLtMatmul(lt, op, &one, w, a, x, b, &zero, d, c, d, c, &heuristics[h].algo, ws, ws_bytes, 0);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        const double tflops = 2.0 * M * N * K / (ms / reps * 1e-3) / 1e12;
        if (hipGetLastError() == hipSuccess && tflops > result) result = tflops;
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
done:
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (c) hipblasLtMatrixLayoutDestroy(c);
    if (op) hipblasLtMatmulDescDestroy(op);
    for (void* p : {w, x, d, ws})
        if (p) hipFree(p);
    hipGetLastError();
    return result;
}

}  // namespace

int main() {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        std::printf("{\"error\": \"no HIP device\"}\n");
        return 1;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        std::printf("{\"error\": \"hipMemGetInfo failed\"}\n");
        return 1;
    }
    // gcnArchName carries target features after the name (gfx942:sramecc+:xnack-); the architecture is the name.
    std::string arch = prop.gcnArchName;
    arch = arch.substr(0, arch.find(':'));
    const double bandwidth = copy_bandwidth_gbs(size_t(2) << 30, 5, 4);
    const size_t sizes[] = {size_t(64) << 10, size_t(1) << 20, size_t(16) << 20, size_t(256) << 20};
    const char* kinds[] = {"pinned", "pageable", "mapped_read"};
    std::string host_to_device;
    for (int kind = 0; kind < 3; ++kind) {
        std::string curve;
        for (const size_t size : sizes) {
            const int copies = int(std::max<size_t>(1, (size_t(512) << 20) / size));   // about 512 MiB per batch
            const double gbs = transfer_gbs(kind, size, 3, copies);
            char item[64];
            std::snprintf(item, sizeof item, "%s\"%zu\": %.1f", curve.empty() ? "" : ", ", size, gbs);
            curve += item;
        }
        host_to_device += std::string(host_to_device.empty() ? "" : ", ") + "\"" + kinds[kind] + "\": {" + curve + "}";
    }
    std::string device_to_host;
    for (int kind = 0; kind < 2; ++kind) {
        std::string curve;
        for (const size_t size : sizes) {
            const int copies = int(std::max<size_t>(1, (size_t(512) << 20) / size));
            const double gbs = transfer_gbs(kind, size, 3, copies, true);
            char item[64];
            std::snprintf(item, sizeof item, "%s\"%zu\": %.1f", curve.empty() ? "" : ", ", size, gbs);
            curve += item;
        }
        device_to_host += std::string(device_to_host.empty() ? "" : ", ") + "\"" + kinds[kind] + "\": {" + curve + "}";
    }
    const int threads = int(std::max(1u, std::thread::hardware_concurrency()));
    const size_t host_bytes = size_t(1) << 30;
    char host[512];
    std::snprintf(host, sizeof host,
                  "\"host_threads\": %d, \"host_memory_gbs\": {\"read\": %.1f, \"write\": %.1f, \"copy\": %.1f, "
                  "\"read_one_thread\": %.1f}, \"cpu_gflops\": {\"fp32\": %.1f, \"fp32_one_thread\": %.1f}, \"cpu_isa\": \"%s\"",
                  threads, host_memory_gbs(0, host_bytes, threads, 5), host_memory_gbs(1, host_bytes, threads, 5),
                  host_memory_gbs(2, host_bytes, threads, 5), host_memory_gbs(0, host_bytes, 1, 5),
                  cpu_gflops(threads, 3), cpu_gflops(1, 3), kIsa);
    hipblasLtHandle_t lt;
    if (hipblasLtCreate(&lt)) {
        std::printf("{\"error\": \"hipblasLtCreate failed\"}\n");
        return 1;
    }
    // The shapes device_probe.cu times: M prompt rows against wide and square projection matrices.
    const int shapes[][3] = {{2048, 17408, 5120}, {4096, 17408, 5120}, {4096, 5120, 17408}, {4096, 8192, 8192}};
    const std::pair<bool, bool> fp8 = fp8_support(arch);
    struct Format { const char* name; hipDataType type; };
    std::vector<Format> formats = {{"bf16", HIP_R_16BF}};
    if (fp8.first) formats.push_back({"fp8", fp8.second ? HIP_R_8F_E4M3_FNUZ : HIP_R_8F_E4M3});
    std::string tflops;
    for (const Format& f : formats) {
        double best = 0.0;
        for (const auto& s : shapes) {
            const double t = gemm_tflops(lt, f.type, s[0], s[1], s[2]);
            if (t > best) best = t;
        }
        if (best > 0.0) {
            char item[96];
            std::snprintf(item, sizeof item, "%s\"%s\": %.1f", tflops.empty() ? "" : ", ", f.name, best);
            tflops += item;
        }
    }
    hipblasLtDestroy(lt);
    const char* unsupported = fp8.first ? "\"nvfp4\", \"fp4_fp8\"" : "\"fp8\", \"nvfp4\", \"fp4_fp8\"";
    const char* encoding = fp8.first ? (fp8.second ? "e4m3fnuz" : "e4m3fn") : "none";
    int runtime = 0;
    hipRuntimeGetVersion(&runtime);
    std::printf("{\"backend\": \"hip\", \"device\": \"%s\", \"arch\": \"%s\", \"runtime_version\": %d, "
                "\"fp8_encoding\": \"%s\", \"unsupported_formats\": [%s], "
                "\"memory_total_bytes\": %llu, \"memory_free_bytes\": %llu, "
                "\"bandwidth_gbs\": %.1f, \"host_to_device_gbs\": {%s}, \"device_to_host_gbs\": {%s}, %s, "
                "\"tflops\": {%s}, \"mma_tflops\": {}}\n",
                prop.name, arch.c_str(), runtime, encoding, unsupported,
                static_cast<unsigned long long>(total_bytes), static_cast<unsigned long long>(free_bytes), bandwidth,
                host_to_device.c_str(), device_to_host.c_str(), host, tflops.c_str());
    return 0;
}
