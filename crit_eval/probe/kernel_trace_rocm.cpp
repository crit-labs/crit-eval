// Kernel tracer for `crit eval kernels --backend hip`: a rocprofiler-sdk tool library (ROCP_TOOL_LIBRARIES) that
// records every kernel dispatch the engine process runs, HIP graph nodes included, without changes to the engine. It
// writes the trace format of kernel_trace.cpp, one file per process to $CRIT_KTRACE_DIR/trace-<pid>.txt at exit:
//   #clock <rocprofiler_ns> <realtime_ns>   pairs rocprofiler's clock with CLOCK_REALTIME (at start and at exit)
//   #device <cu_count> <name>
//   <start_ns> <end_ns> <workgroups> <workgroup_threads> <name>
// A dispatch's grid is counted in workgroups (HSA grid sizes are work-items), as CUDA grids count blocks; compute
// units stand where SMs stand for the launch width. Names are the code objects' mangled kernel symbols. Linux only.
//
// Not yet run on hardware: written against the rocprofiler-sdk API of ROCm 6.4.
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unistd.h>

namespace {

FILE* g_out = nullptr;
std::mutex g_mu;
std::unordered_map<uint64_t, std::string> g_names;   // kernel_id -> symbol, from code-object registration
rocprofiler_context_id_t g_context{};
rocprofiler_buffer_id_t g_buffer{};
int g_cus = 0;
std::string g_device = "?";

void clock_line() {
    rocprofiler_timestamp_t now = 0;
    rocprofiler_get_timestamp(&now);
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    const unsigned long long real = static_cast<unsigned long long>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
    fprintf(g_out, "#clock %llu %llu\n", static_cast<unsigned long long>(now), real);
}

unsigned long long groups(uint32_t items, uint32_t size) { return size ? (items + size - 1) / size : items; }

void code_objects(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*) {
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
        return;
    const auto* symbol = static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(record.payload);
    std::lock_guard<std::mutex> lock(g_mu);
    g_names[symbol->kernel_id] = symbol->kernel_name ? symbol->kernel_name : "?";
}

void dispatches(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers, size_t count,
                void*, uint64_t) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (size_t i = 0; i < count; ++i) {
        const rocprofiler_record_header_t* header = headers[i];
        if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING || header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            continue;
        const auto* r = static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
        const auto& info = r->dispatch_info;
        const auto name = g_names.find(info.kernel_id);
        if (g_out)
            fprintf(g_out, "%llu %llu %llu %u %s\n", static_cast<unsigned long long>(r->start_timestamp),
                    static_cast<unsigned long long>(r->end_timestamp),
                    groups(info.grid_size.x, info.workgroup_size.x) * groups(info.grid_size.y, info.workgroup_size.y) *
                        groups(info.grid_size.z, info.workgroup_size.z),
                    info.workgroup_size.x * info.workgroup_size.y * info.workgroup_size.z,
                    name == g_names.end() ? "?" : name->second.c_str());
    }
}

rocprofiler_status_t first_gpu(rocprofiler_agent_version_t, const void** agents, size_t count, void*) {
    for (size_t i = 0; i < count; ++i) {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if (agent->type == ROCPROFILER_AGENT_TYPE_GPU) {
            g_cus = static_cast<int>(agent->cu_count);
            g_device = agent->product_name ? agent->product_name : (agent->name ? agent->name : "?");
            break;
        }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int tool_init(rocprofiler_client_finalize_t, void*) {
    const char* dir = getenv("CRIT_KTRACE_DIR");
    if (!dir) return 0;
    const std::string path = std::string(dir) + "/trace-" + std::to_string(getpid()) + ".txt";
    g_out = fopen(path.c_str(), "w");
    if (!g_out) return 0;
    clock_line();
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, first_gpu, sizeof(rocprofiler_agent_t), nullptr);
    if (rocprofiler_create_context(&g_context) != ROCPROFILER_STATUS_SUCCESS) return -1;
    rocprofiler_configure_callback_tracing_service(g_context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                                   code_objects, nullptr);
    rocprofiler_create_buffer(g_context, 8u << 20, 7u << 20, ROCPROFILER_BUFFER_POLICY_LOSSLESS, dispatches, nullptr,
                              &g_buffer);
    rocprofiler_configure_buffer_tracing_service(g_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_buffer);
    return rocprofiler_start_context(g_context) == ROCPROFILER_STATUS_SUCCESS ? 0 : -1;
}

void tool_fini(void*) {
    rocprofiler_flush_buffer(g_buffer);
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_out) return;
    clock_line();
    fprintf(g_out, "#device %d %s\n", g_cus, g_device.c_str());
    fclose(g_out);
    g_out = nullptr;
}

}  // namespace

// rocprofiler's clock now, for phase marks on the clock the dispatch records are on.
extern "C" unsigned long long crit_ktrace_timestamp() {
    rocprofiler_timestamp_t now = 0;
    rocprofiler_get_timestamp(&now);
    return static_cast<unsigned long long>(now);
}

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t, const char*, uint32_t,
                                                                      rocprofiler_client_id_t* id) {
    id->name = "crit-kernel-trace";
    static rocprofiler_tool_configure_result_t result = {sizeof(rocprofiler_tool_configure_result_t), &tool_init,
                                                         &tool_fini, nullptr};
    return &result;
}
