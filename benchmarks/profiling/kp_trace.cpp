// kp_trace: minimal Kokkos Tools library that records a timeline of the
// solver for profiling without a vendor profiler (no ROCm/CUDA tools needed).
//
//   g++ -O2 -std=c++17 -shared -fPIC -o libkp_trace.so kp_trace.cpp
//   KOKKOS_TOOLS_LIBS=$PWD/libkp_trace.so KP_TRACE_OUT=out/trace vleo_solver ...
//
// Records, on the host clock (steady_clock, ns):
//   region   Kokkos profiling regions (vleo phases and stages), with nesting depth
//   kernel   parallel_for / reduce / scan by label
//   copy     deep_copy (destination <- source space, bytes)
//   fence    explicit fences (the host waiting for the device)
// and writes <KP_TRACE_OUT>.csv and <KP_TRACE_OUT>.json (Chrome trace format,
// open in https://ui.perfetto.dev or chrome://tracing).
//
// KP_TRACE_FENCE=1 (default) asks Kokkos to fence after every kernel, so a
// kernel interval spans launch to completion on the GPU (GPU time + a few µs
// launch/sync latency) and the gaps between kernels are host-only time.
// KP_TRACE_FENCE=0 keeps the asynchronous execution: kernel intervals are then
// only the host-side launches and the waiting shows up in the next fence or
// copy.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Event {
  char kind;  // r region, k kernel, c copy, f fence
  std::string name;
  std::int64_t t0, t1;
  int depth;
  std::uint64_t bytes;
};

std::vector<Event> events;
std::vector<std::size_t> region_stack;  // indices into events
std::vector<std::size_t> open_kernels;  // by kernel id
std::size_t open_copy = SIZE_MAX;
std::vector<std::size_t> open_fences;
std::int64_t t_start = 0;

std::int64_t now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::size_t open(char kind, const char* name, std::uint64_t bytes = 0) {
  events.push_back({kind, name ? name : "?", now(), -1, static_cast<int>(region_stack.size()), bytes});
  return events.size() - 1;
}

void close(std::size_t i) {
  if (i < events.size()) events[i].t1 = now();
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) o += c;
  }
  return o;
}

void begin_kernel(const char* name, std::uint64_t* id) {
  const std::size_t i = open('k', name);
  open_kernels.push_back(i);
  *id = open_kernels.size() - 1;
}
void end_kernel(std::uint64_t id) {
  if (id < open_kernels.size()) close(open_kernels[id]);
}

}  // namespace

extern "C" {

struct Kokkos_Tools_ToolSettings {
  bool requires_global_fencing;
  bool padding[255];
};
struct Kokkos_Profiling_SpaceHandle {
  char name[64];
};

void kokkosp_request_tool_settings(const std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  const char* f = std::getenv("KP_TRACE_FENCE");
  settings->requires_global_fencing = !(f && std::strcmp(f, "0") == 0);
}

void kokkosp_init_library(const int, const std::uint64_t, const std::uint32_t, void*) {
  events.reserve(1 << 20);
  open_kernels.reserve(1 << 20);
  t_start = now();
}

void kokkosp_begin_parallel_for(const char* name, std::uint32_t, std::uint64_t* id) { begin_kernel(name, id); }
void kokkosp_end_parallel_for(std::uint64_t id) { end_kernel(id); }
void kokkosp_begin_parallel_reduce(const char* name, std::uint32_t, std::uint64_t* id) { begin_kernel(name, id); }
void kokkosp_end_parallel_reduce(std::uint64_t id) { end_kernel(id); }
void kokkosp_begin_parallel_scan(const char* name, std::uint32_t, std::uint64_t* id) { begin_kernel(name, id); }
void kokkosp_end_parallel_scan(std::uint64_t id) { end_kernel(id); }

void kokkosp_push_profile_region(const char* name) { region_stack.push_back(open('r', name)); }
void kokkosp_pop_profile_region() {
  if (region_stack.empty()) return;
  close(region_stack.back());
  region_stack.pop_back();
}

void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle dst, const char* dst_name, const void*,
                             Kokkos_Profiling_SpaceHandle src, const char* src_name, const void*, std::uint64_t size) {
  const std::string n = std::string(dst.name) + "<-" + src.name + " " + (dst_name ? dst_name : "") + "<-" +
                        (src_name ? src_name : "");
  open_copy = open('c', n.c_str(), size);
}
void kokkosp_end_deep_copy() {
  close(open_copy);
  open_copy = SIZE_MAX;
}

void kokkosp_begin_fence(const char* name, std::uint32_t, std::uint64_t* id) {
  open_fences.push_back(open('f', name));
  *id = open_fences.size() - 1;
}
void kokkosp_end_fence(std::uint64_t id) {
  if (id < open_fences.size()) close(open_fences[id]);
}

void kokkosp_finalize_library() {
  const char* out = std::getenv("KP_TRACE_OUT");
  const std::string base = out ? out : "kp_trace";
  std::FILE* c = std::fopen((base + ".csv").c_str(), "w");
  std::FILE* j = std::fopen((base + ".json").c_str(), "w");
  if (!c || !j) {
    std::fprintf(stderr, "kp_trace: cannot write %s.csv/.json\n", base.c_str());
    return;
  }
  std::fprintf(c, "kind,start_ns,end_ns,depth,bytes,name\n");
  std::fprintf(j, "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n");
  // Perfetto rows: 1 host regions, 2 kernels, 3 copies, 4 fences
  std::fprintf(j, "{\"ph\":\"M\",\"pid\":1,\"tid\":1,\"name\":\"thread_name\",\"args\":{\"name\":\"host: regions\"}},\n"
                  "{\"ph\":\"M\",\"pid\":1,\"tid\":2,\"name\":\"thread_name\",\"args\":{\"name\":\"kernels\"}},\n"
                  "{\"ph\":\"M\",\"pid\":1,\"tid\":3,\"name\":\"thread_name\",\"args\":{\"name\":\"deep_copy\"}},\n"
                  "{\"ph\":\"M\",\"pid\":1,\"tid\":4,\"name\":\"thread_name\",\"args\":{\"name\":\"fences\"}}");
  std::size_t written = 0;
  for (const Event& e : events) {
    if (e.t1 < 0) continue;
    const char* kind = e.kind == 'r' ? "region" : e.kind == 'k' ? "kernel" : e.kind == 'c' ? "copy" : "fence";
    // the tool's own global fences are part of the kernel intervals
    if (e.kind == 'f' && e.name.find("Kokkos Profile Tool Fence") != std::string::npos) continue;
    std::string name = e.name;
    for (char& ch : name)
      if (ch == ',' || ch == '\n') ch = ';';
    std::fprintf(c, "%s,%lld,%lld,%d,%llu,%s\n", kind, static_cast<long long>(e.t0 - t_start),
                 static_cast<long long>(e.t1 - t_start), e.depth, static_cast<unsigned long long>(e.bytes),
                 name.c_str());
    const int tid = e.kind == 'r' ? 1 : e.kind == 'k' ? 2 : e.kind == 'c' ? 3 : 4;
    std::fprintf(j, ",\n{\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"cat\":\"%s\",\"name\":\"%s\",\"ts\":%.3f,\"dur\":%.3f%s}",
                 tid, kind, json_escape(e.name).c_str(), (e.t0 - t_start) / 1e3, (e.t1 - e.t0) / 1e3,
                 e.kind == 'c' ? (",\"args\":{\"bytes\":" + std::to_string(e.bytes) + "}").c_str() : "");
    ++written;
  }
  std::fprintf(j, "\n]}\n");
  std::fclose(c);
  std::fclose(j);
  std::fprintf(stderr, "kp_trace: %zu events -> %s.csv, %s.json\n", written, base.c_str(), base.c_str());
}

}  // extern "C"
