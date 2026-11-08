// crit eval device probe: free device memory, achievable copy bandwidth and dense GEMM rates per
// weight format. Built and run by `crit eval probe`; prints one JSON object on stdout.
//
// Memory: free and total device memory (cudaMemGetInfo) once the probe's CUDA context exists and
// before it allocates anything, which is what an engine process can use; Eval's max context starts
// from the free bytes.
//
// Bandwidth: device-to-device copies of a 2 GiB buffer, counting bytes read plus bytes written,
// mean over batches. Host to device, at each transfer size of a curve (the record names the sizes):
// cudaMemcpyAsync from pinned and from pageable host memory, and a kernel reading mapped pinned
// memory (device-readable host memory, as host-resident weights are read); bytes moved per second,
// mean over batches. Pageable copies stage through the driver, so they show what an unpinned host
// buffer costs on this platform. GEMM: cuBLASLt with FP32 accumulation at prefill-like shapes, best rate per
// format: BF16, FP8 E4M3 and NVFP4 (E2M1 with 16-element E4M3 block scales). Operands hold
// pseudo-random bytes (constant operands draw less power and run faster than real weights);
// block scales are a fixed small power of two. A format cuBLASLt cannot run is omitted.
//
// Device to host: the same curve for cudaMemcpyAsync into pinned and into pageable host memory, as an engine
// moves KV or activations off the device. Host, for engines that keep weights or experts in host memory or
// compute on the CPU: host memory read, write and copy GB/s over 1 GiB buffers (a copy counts bytes read plus
// bytes written), on every hardware thread and on one, best of five passes; and CPU FP32 throughput as GFLOPS of
// independent fused multiply-add chains in the widest vectors the build targets (AVX-512 or AVX2, recorded as
// cpu_isa), on every hardware thread and on one. These are the host's ceilings, not what a particular kernel reaches.
//
// Raw tensor-core rates (mma_tflops): mma.sync throughput with operands in registers and every SM busy, per input and
// accumulator format: BF16 and FP16 into FP32, FP16 into FP16 (twice the FP32-accumulate rate on some devices), E4M3
// into FP32 (sm_89+), and on sm_120a E2M1 with E4M3 block scales (NVFP4) and E2M1 times E4M3 (kind::f8f6f4, the
// rate of FP4 weights with FP8 activations). A GEMM moves operands through memory, so it reaches less than these,
// but more than cuBLASLt where cuBLASLt's kernels are not the device's best (BF16 on an RTX 5090: 185 vs 252). The
// probe is built for the device's own architecture (`crit eval probe` passes it); a kind the build cannot run is
// omitted.
#include <cublasLt.h>
#include <cuda_runtime.h>

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
    if (cudaMalloc(&src, bytes) != cudaSuccess || cudaMalloc(&dst, bytes) != cudaSuccess) return 0.0;
    cudaMemset(src, 1, bytes);
    cudaMemset(dst, 0, bytes);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);  // warmup
    double total = 0.0;
    for (int b = 0; b < batches; ++b) {
        cudaEventRecord(start);
        for (int i = 0; i < copies; ++i) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        total += 2.0 * bytes * copies / (ms * 1e-3) / 1e9;
    }
    cudaFree(src);
    cudaFree(dst);
    return total / batches;
}

// GB/s copying `bytes` from device `from` to device `to` (cudaMemcpyPeerAsync, peer access enabled where the pair
// supports it), timed on `to`; 0 when an allocation or a copy fails. The same device twice times a device-to-device
// copy's one direction.
double peer_bandwidth_gbs(int from, int to, size_t bytes, int batches, int copies, int* access) {
    *access = 0;
    if (from != to) {
        cudaDeviceCanAccessPeer(access, to, from);
        if (*access) {
            cudaSetDevice(to);
            if (cudaDeviceEnablePeerAccess(from, 0) == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    } else {
        *access = 1;
    }
    void *src = nullptr, *dst = nullptr;
    cudaSetDevice(from);
    if (cudaMalloc(&src, bytes) != cudaSuccess) { cudaGetLastError(); return 0.0; }
    cudaMemset(src, 1, bytes);
    cudaSetDevice(to);
    if (cudaMalloc(&dst, bytes) != cudaSuccess) { cudaGetLastError(); cudaSetDevice(from); cudaFree(src); return 0.0; }
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemcpyPeerAsync(dst, to, src, from, bytes, stream);  // warmup
    cudaStreamSynchronize(stream);
    double total = 0.0;
    for (int b = 0; b < batches; ++b) {
        cudaEventRecord(start, stream);
        for (int i = 0; i < copies; ++i) cudaMemcpyPeerAsync(dst, to, src, from, bytes, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms > 0.f ? double(bytes) * copies / (ms * 1e-3) / 1e9 : 0.0;
    }
    const bool failed = cudaGetLastError() != cudaSuccess;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(dst);
    cudaSetDevice(from);
    cudaFree(src);
    return failed ? 0.0 : total / batches;
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
    if (cudaMalloc(&device, bytes) != cudaSuccess) { cudaGetLastError(); return 0.0; }
    const bool pageable = kind == 1;
    if (pageable) host = std::malloc(bytes);
    else if (cudaHostAlloc(&host, bytes, kind == 2 ? cudaHostAllocMapped : cudaHostAllocDefault) != cudaSuccess) host = nullptr;
    if (!host) { cudaGetLastError(); cudaFree(device); return 0.0; }
    std::memset(host, 1, bytes);
    void* alias = nullptr;
    if (kind == 2 && cudaHostGetDevicePointer(&alias, host, 0) != cudaSuccess) alias = nullptr;
    bool failed = false;
    auto move = [&] {
        if (kind == 2) read_mapped<<<1024, 256>>>(static_cast<const uint4*>(alias), static_cast<uint4*>(device), bytes / sizeof(uint4));
        else if ((to_host ? cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost)
                          : cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) failed = true;
        if (cudaPeekAtLastError() != cudaSuccess) failed = true;
    };
    double total = 0.0;
    if (kind != 2 || alias) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        move();  // warmup
        cudaDeviceSynchronize();
        for (int b = 0; b < batches; ++b) {
            cudaEventRecord(start);
            for (int i = 0; i < copies; ++i) move();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess || !(ms > 0.f)) failed = true;
            if (failed) break;
            total += double(bytes) * copies / (ms * 1e-3) / 1e9;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        failed = true;
    }
    if (failed) { cudaGetLastError(); total = 0.0; }
    if (pageable) std::free(host);
    else cudaFreeHost(host);
    cudaFree(device);
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

// The CPU expert lane: 4-bit expert weights against int8 activations as crit's integer block rows compute them
// (host-expert-arithmetic host_int.h, interleaved layout: one vector lane per 16-block, exact integer block sums, one
// FP32 step per block), on vpdpbusd (AVX-512 VNNI), vpmaddubsw (AVX2) or scalar code, whichever this build targets.
// GB/s of weight bytes (codes and block scales) for `tokens` tokens a matrix, each thread streaming its own share of a
// buffer far larger than the caches. What a CPU tier computing an expert's rows reaches, which the host memory read
// rate alone overstates for FP32 rows and several tokens.
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
constexpr const char* kIntIsa = "avx512_vnni";
#elif defined(__AVX2__)
constexpr const char* kIntIsa = "avx2";
#else
constexpr const char* kIntIsa = "scalar";
#endif
constexpr size_t kGroupBytes = 144;   // 16 blocks: 128 bytes of codes, 16 scale bytes

void expert_group(const uint8_t* g, const uint8_t* acts, const float* scales, int tokens, float* acc) {
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
    __m512i w[4];
    for (int c = 0; c < 4; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g + 32 * c));
        const __m256i lo = _mm256_and_si256(q, _mm256_set1_epi8(15));
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), _mm256_set1_epi8(15));
        w[c] = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
    }
    alignas(64) float ws[16];
    for (int i = 0; i < 16; ++i) ws[i] = 0.01f * float(g[128 + i] & 127);
    const __m512 wv = _mm512_load_ps(ws);
    for (int t = 0; t < tokens; ++t) {
        const uint8_t* x = acts + 256 * t;
        const __m512i zero = _mm512_setzero_si512();
        const __m512i sum = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x), w[0]),
                             _mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x + 64), w[1])),
            _mm512_add_epi32(_mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x + 128), w[2]),
                             _mm512_dpbusd_epi32(zero, _mm512_loadu_si512(x + 192), w[3])));
        __m512 a = _mm512_loadu_ps(acc + 16 * t);
        a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), _mm512_mul_ps(wv, _mm512_loadu_ps(scales + 16 * t)), a);
        _mm512_storeu_ps(acc + 16 * t, a);
    }
#elif defined(__AVX2__)
    const __m128i lut128 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m256i lut = _mm256_setr_m128i(lut128, lut128), ones = _mm256_set1_epi16(1);
    __m256i w[4][2];
    for (int c = 0; c < 4; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g + 32 * c));
        w[c][0] = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, _mm256_set1_epi8(15)));
        w[c][1] = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), _mm256_set1_epi8(15)));
    }
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < 2; ++h) {
            __m256i sum = _mm256_setzero_si256();
            for (int c = 0; c < 4; ++c)
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acts + 256 * t + 64 * c + 32 * h)), w[c][h]), ones));
            __m256 a = _mm256_loadu_ps(acc + 16 * t + 8 * h);
            a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), _mm256_loadu_ps(scales + 16 * t + 8 * h), a);
            _mm256_storeu_ps(acc + 16 * t + 8 * h, a);
        }
#else
    for (int t = 0; t < tokens; ++t)
        for (int b = 0; b < 16; ++b) {
            int sum = 0;
            for (int e = 0; e < 16; ++e) {
                const uint8_t byte = g[32 * (e / 4) + 4 * (b % 8) + e % 4];
                const int code = b < 8 ? byte & 15 : byte >> 4;
                sum += (code & 8 ? -(code & 7) : (code & 7)) * int(acts[256 * t + 64 * (e / 4) + 4 * b + e % 4]);
            }
            acc[16 * t + b] += float(sum) * scales[16 * t + b];
        }
#endif
}

// GB/s of weights the lane reads for `tokens` tokens on `threads` threads over `bytes` of experts, best of `passes`.
double host_expert_gbs(int tokens, int threads, size_t bytes, int passes) {
    const size_t groups = bytes / kGroupBytes;
    std::vector<uint8_t> weights(groups * kGroupBytes);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = uint8_t((i * 2654435761u) >> 13);
    double best = 0.0;
    for (int pass = 0; pass < passes; ++pass) {
        const double seconds = timed_threads(threads, [&](int t) {
            std::vector<uint8_t> acts(size_t(256) * 8, 129);
            std::vector<float> scales(16 * 8, 0.01f), acc(16 * 8, 0.0f);
            const size_t begin = groups * t / threads, end = groups * (t + 1) / threads;
            for (size_t gi = begin; gi < end; ++gi)
                expert_group(weights.data() + gi * kGroupBytes, acts.data(), scales.data(), tokens, acc.data());
            g_sink += uint64_t(acc[0]);
        });
        if (seconds > 0) best = std::max(best, double(groups * kGroupBytes) / seconds / 1e9);
    }
    return best;
}

// The lane at 3 tokens while expert-sized pinned copies (2.76 MB) run over the host link: both read host memory, so
// each runs below its rate alone. Fills cpu and link GB/s measured over the same window.
void host_expert_with_link(int threads, size_t bytes, double* cpu, double* link) {
    *cpu = *link = 0.0;
    const size_t piece = 2764800, pieces = 64;
    void* host = nullptr;
    void* device = nullptr;
    if (cudaHostAlloc(&host, piece * pieces, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); return; }
    if (cudaMalloc(&device, piece * 8) != cudaSuccess) { cudaGetLastError(); cudaFreeHost(host); return; }
    std::memset(host, 1, piece * pieces);
    std::atomic<bool> stop{false};
    std::atomic<double> moved{0.0};
    std::thread copier([&] {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        for (size_t i = 0; !stop.load(); ++i) {
            cudaMemcpyAsync(static_cast<uint8_t*>(device) + piece * (i % 8), static_cast<uint8_t*>(host) + piece * (i % pieces),
                            piece, cudaMemcpyHostToDevice, stream);
            if (i % 8 == 7) {
                cudaStreamSynchronize(stream);
                double seen = moved.load();
                while (!moved.compare_exchange_weak(seen, seen + double(piece) * 8)) {}
            }
        }
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));   // the copies run before the lane starts
    const double before = moved.load();
    const auto start = Clock::now();
    *cpu = host_expert_gbs(3, threads, bytes, 1);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    *link = (moved.load() - before) / seconds / 1e9;
    stop.store(true);
    copier.join();
    cudaFree(device);
    cudaFreeHost(host);
}

// Raw tensor-core rates: mma.sync throughput per input and accumulator format, every SM busy, operands in registers.
// Kinds: 0 bf16->f32, 1 f16->f32, 2 f16->f16, 3 e4m3->f32, 4 e2m1 x e2m1 block-scaled (nvfp4) -> f32,
// 5 e2m1 x e4m3 (kind::f8f6f4) -> f32.
constexpr int kMmaKinds = 6;
constexpr const char* kMmaNames[kMmaKinds] = {"bf16", "fp16", "fp16_acc", "fp8", "nvfp4", "fp4_fp8"};
constexpr int kMmaK[kMmaKinds] = {16, 16, 16, 32, 64, 32};
constexpr int kMmaChains = 8;   // independent accumulators per thread, enough to keep the tensor cores fed

// The kinds this build's device code runs: FP8 from sm_89, the FP4 kinds on sm_120a/sm_121a only.
__global__ void mma_kinds(int* out) {
    int kinds = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    kinds |= 0x7;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    kinds |= 0x8;
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    kinds |= 0x30;
#endif
    *out = kinds;
}

template <int KIND>
__global__ void mma_rate(float* out, int iters) {
    uint32_t a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x3c3c3c3cu, a2 = a0 ^ 0x5a5a5a5au, a3 = a0 ^ 0x69696969u;
    uint32_t b0 = a0 ^ 0x33333333u, b1 = a0 ^ 0x55555555u;
    float c[kMmaChains][4] = {};
    uint32_t h[kMmaChains][2] = {};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kMmaChains; ++j) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (KIND == 0)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            if constexpr (KIND == 1)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            if constexpr (KIND == 2)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                             : "+r"(h[j][0]), "+r"(h[j][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
            if constexpr (KIND == 3)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
            if constexpr (KIND == 4) {
                const uint32_t scale = 0x38383838u;   // E4M3 1.0 in every byte
                const uint16_t zero = 0;
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale), "h"(zero), "h"(zero));
            }
            if constexpr (KIND == 5)
                asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
        }
    }
    float s = 0.f;
    for (int j = 0; j < kMmaChains; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3] + float(h[j][0] & 0xff) + float(h[j][1] & 0xff);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// TFLOPS of one kind (m16n8kK per instruction, 2 FLOPs per multiply-add), best of three launches; 0 when the build
// cannot run it.
double mma_tflops(int kind, int sms) {
    const int blocks = sms * 4, threads = 256, iters = 2048;
    float* out = nullptr;
    if (cudaMalloc(&out, size_t(blocks) * threads * sizeof(float)) != cudaSuccess) { cudaGetLastError(); return 0.0; }
    auto launch = [&] {
        switch (kind) {
            case 0: mma_rate<0><<<blocks, threads>>>(out, iters); break;
            case 1: mma_rate<1><<<blocks, threads>>>(out, iters); break;
            case 2: mma_rate<2><<<blocks, threads>>>(out, iters); break;
            case 3: mma_rate<3><<<blocks, threads>>>(out, iters); break;
            case 4: mma_rate<4><<<blocks, threads>>>(out, iters); break;
            default: mma_rate<5><<<blocks, threads>>>(out, iters); break;
        }
    };
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    launch();   // warmup
    double best = 0.0;
    for (int rep = 0; rep < 3; ++rep) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess || !(ms > 0.f)) break;
        const double flops = 2.0 * blocks * (threads / 32) * double(iters) * kMmaChains * 16 * 8 * kMmaK[kind];
        best = std::max(best, flops / (ms * 1e-3) / 1e12);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(out);
    if (cudaGetLastError() != cudaSuccess) best = 0.0;
    return best;
}

// TFLOPS of D[N x M] = W[N x K] * X[K x M] (W row-major, op T), or 0 when unsupported.
double gemm_tflops(cublasLtHandle_t lt, cudaDataType_t type, bool block_scaled, int M, int N, int K) {
    const bool fp4 = type == CUDA_R_4F_E2M1;
    const size_t element_bits = fp4 ? 4 : (type == CUDA_R_8F_E4M3 ? 8 : 16);
    const size_t wbytes = size_t(N) * K * element_bits / 8, xbytes = size_t(M) * K * element_bits / 8;
    // Block-scale tensors are padded to 128 rows x 4 scale columns.
    const size_t sw_bytes = size_t((N + 127) / 128 * 128) * ((K / 16 + 3) / 4 * 4);
    const size_t sx_bytes = size_t((M + 127) / 128 * 128) * ((K / 16 + 3) / 4 * 4);
    const size_t ws_bytes = 64ull << 20;
    void *w = nullptr, *x = nullptr, *d = nullptr, *ws = nullptr, *sw = nullptr, *sx = nullptr;
    double result = 0.0;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristics[8];
    int found = 0;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    const float one = 1.f, zero = 0.f;
    cudaEvent_t start, stop;
    if (cudaMalloc(&w, wbytes) || cudaMalloc(&x, xbytes) || cudaMalloc(&d, size_t(M) * N * 2) || cudaMalloc(&ws, ws_bytes))
        goto done;
    fill(w, wbytes, 1);
    fill(x, xbytes, 2);
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) goto done;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    if (block_scaled) {
        const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        if (cudaMalloc(&sw, sw_bytes) || cudaMalloc(&sx, sx_bytes)) goto done;
        cudaMemset(sw, 0x38, sw_bytes);
        cudaMemset(sx, 0x38, sx_bytes);
        if (cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode) ||
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode) ||
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw, sizeof sw) ||
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof sx))
            goto done;
    }
    if (cublasLtMatrixLayoutCreate(&a, type, K, N, K) || cublasLtMatrixLayoutCreate(&b, type, K, M, K) ||
        cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, M, N))
        goto done;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes);
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, 8, heuristics, &found) || !found) goto done;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int h = 0; h < found; ++h) {  // best of the offered algorithms
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
            ok = !cublasLtMatmul(lt, op, &one, w, a, x, b, &zero, d, c, d, c, &heuristics[h].algo, ws, ws_bytes, 0);
        if (!ok) continue;
        const int reps = 20;
        cudaEventRecord(start);
        for (int i = 0; i < reps; ++i)
            cublasLtMatmul(lt, op, &one, w, a, x, b, &zero, d, c, d, c, &heuristics[h].algo, ws, ws_bytes, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        const double tflops = 2.0 * M * N * K / (ms / reps * 1e-3) / 1e12;
        if (cudaGetLastError() == cudaSuccess && tflops > result) result = tflops;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (op) cublasLtMatmulDescDestroy(op);
    for (void* p : {w, x, d, ws, sw, sx})
        if (p) cudaFree(p);
    cudaGetLastError();
    return result;
}

}  // namespace

// device_probe [--device N | --count | --peer FROM TO]: one device's figures (device 0 by default), the device count,
// or the copy rate from one device to another.
int main(int argc, char** argv) {
    int device = 0;
    if (argc == 2 && std::strcmp(argv[1], "--count") == 0) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
        std::printf("{\"devices\": %d}\n", count);
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "--peer") == 0) {
        const int from = std::atoi(argv[2]), to = std::atoi(argv[3]);
        int access = 0;
        const double gbs = peer_bandwidth_gbs(from, to, size_t(256) << 20, 5, 4, &access);
        std::printf("{\"from\": %d, \"to\": %d, \"gbs\": %.1f, \"peer_access\": %s}\n", from, to, gbs,
                    access ? "true" : "false");
        return gbs > 0.0 ? 0 : 1;
    }
    if (argc == 3 && std::strcmp(argv[1], "--device") == 0) device = std::atoi(argv[2]);
    if (cudaSetDevice(device) != cudaSuccess) {
        std::printf("{\"error\": \"no CUDA device %d\"}\n", device);
        return 1;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::printf("{\"error\": \"no CUDA device\"}\n");
        return 1;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        std::printf("{\"error\": \"cudaMemGetInfo failed\"}\n");
        return 1;
    }
    const double bandwidth = copy_bandwidth_gbs(size_t(2) << 30, 5, 4);
    // Host-to-device curve: sizes of a single expert or layer slice up to a large bulk transfer.
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
    // The expert lane on one thread per physical core (half the hardware threads where they pair), and beside the
    // link with one core left to issue the copies.
    const int cores = std::max(1, threads > 1 ? threads / 2 : 1);
    const size_t lane_bytes = size_t(512) << 20;
    double with_cpu = 0.0, with_link = 0.0;
    host_expert_with_link(std::max(1, cores - 1), lane_bytes, &with_cpu, &with_link);
    char host[1024];
    std::snprintf(host, sizeof host,
                  "\"host_threads\": %d, \"host_memory_gbs\": {\"read\": %.1f, \"write\": %.1f, \"copy\": %.1f, "
                  "\"read_one_thread\": %.1f}, \"cpu_gflops\": {\"fp32\": %.1f, \"fp32_one_thread\": %.1f}, \"cpu_isa\": \"%s\", "
                  "\"host_expert_gbs\": {\"tokens_1\": %.1f, \"tokens_3\": %.1f, \"tokens_8\": %.1f, \"threads\": %d, "
                  "\"with_link_cpu\": %.1f, \"with_link_link\": %.1f}, \"host_expert_isa\": \"%s\"",
                  threads, host_memory_gbs(0, host_bytes, threads, 5), host_memory_gbs(1, host_bytes, threads, 5),
                  host_memory_gbs(2, host_bytes, threads, 5), host_memory_gbs(0, host_bytes, 1, 5),
                  cpu_gflops(threads, 3), cpu_gflops(1, 3), kIsa,
                  host_expert_gbs(1, cores, lane_bytes, 3), host_expert_gbs(3, cores, lane_bytes, 3),
                  host_expert_gbs(8, cores, lane_bytes, 3), cores, with_cpu, with_link, kIntIsa);
    cublasLtHandle_t lt;
    if (cublasLtCreate(&lt)) {
        std::printf("{\"error\": \"cublasLtCreate failed\"}\n");
        return 1;
    }
    // Prefill-like shapes: M prompt rows against wide and square projection matrices.
    const int shapes[][3] = {{2048, 17408, 5120}, {4096, 17408, 5120}, {4096, 5120, 17408}, {4096, 8192, 8192}};
    struct Format { const char* name; cudaDataType_t type; bool scaled; };
    const Format formats[] = {{"bf16", CUDA_R_16BF, false}, {"fp8", CUDA_R_8F_E4M3, false}, {"nvfp4", CUDA_R_4F_E2M1, true}};
    std::string tflops;
    for (const Format& f : formats) {
        double best = 0.0;
        for (const auto& s : shapes) {
            const double t = gemm_tflops(lt, f.type, f.scaled, s[0], s[1], s[2]);
            if (t > best) best = t;
        }
        if (best > 0.0) {
            char item[96];
            std::snprintf(item, sizeof item, "%s\"%s\": %.1f", tflops.empty() ? "" : ", ", f.name, best);
            tflops += item;
        }
    }
    cublasLtDestroy(lt);
    std::string mma;
    int* mma_out = nullptr;
    int mma_supported = 0;
    if (cudaMalloc(&mma_out, sizeof(int)) == cudaSuccess) {
        mma_kinds<<<1, 1>>>(mma_out);
        if (cudaMemcpy(&mma_supported, mma_out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) mma_supported = 0;
        cudaFree(mma_out);
    }
    cudaGetLastError();
    for (int kind = 0; kind < kMmaKinds; ++kind) {
        const double t = mma_supported & (1 << kind) ? mma_tflops(kind, prop.multiProcessorCount) : 0.0;
        if (t > 0.0) {
            char item[96];
            std::snprintf(item, sizeof item, "%s\"%s\": %.1f", mma.empty() ? "" : ", ", kMmaNames[kind], t);
            mma += item;
        }
    }
    int runtime = 0;
    cudaRuntimeGetVersion(&runtime);
    std::printf("{\"device\": \"%s\", \"compute_capability\": \"%d.%d\", \"cuda_runtime\": %d, "
                "\"memory_total_bytes\": %llu, \"memory_free_bytes\": %llu, "
                "\"bandwidth_gbs\": %.1f, \"host_to_device_gbs\": {%s}, \"device_to_host_gbs\": {%s}, %s, "
                "\"tflops\": {%s}, \"mma_tflops\": {%s}}\n",
                prop.name, prop.major, prop.minor, runtime, static_cast<unsigned long long>(total_bytes),
                static_cast<unsigned long long>(free_bytes), bandwidth, host_to_device.c_str(), device_to_host.c_str(),
                host, tflops.c_str(), mma.c_str());
    return 0;
}
