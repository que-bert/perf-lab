// wmma_peak: RDNA4 / RADV cooperative-matrix ceilings (plan P0.1 + P0.2).
//
//   wmma_peak --device 1 --type {f16,f16acc16,s8,fp8,all} [--iters N] [--sg 32|64]
//   wmma_peak --device 1 --bw                                  # DRAM read + LDS read GB/s
//   wmma_peak --device 1 --gemm {f16,s8} --m M --n N --k K --tile {128x128,256x128}
//             [--bk 32|64] [--sg 32|64] [--pad U] [--sweep] [--verify]
//
// Shaders are GLSL next to this file, compiled on demand with glslc into a cache dir
// (WMMA_SHADER_DIR / WMMA_CACHE override). All timing uses GPU timestamps.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static std::string g_shader_dir, g_cache;

struct Ctx {
    VkInstance inst{}; VkPhysicalDevice pd{}; VkDevice dev{}; VkQueue q{}; uint32_t qf = 0;
    VkCommandPool pool{}; VkQueryPool qp{}; float ts_period = 1; bool fp8 = false;
    VkPhysicalDeviceMemoryProperties memp{};
};

struct Buf { VkBuffer b{}; VkDeviceMemory m{}; void * p = nullptr; size_t size = 0; };

static uint32_t find_mem(Ctx & c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.memp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (c.memp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// host=false: pure VRAM (DEVICE_LOCAL only; this box has a 256 MiB BAR, no ReBAR), filled via staging.
// host=true : host-visible coherent staging memory.
static Buf make_buf(Ctx & c, size_t size, bool host = false) {
    Buf b; b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VKC(vkCreateBuffer(c.dev, &bi, nullptr, &b.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, b.b, &mr);
    uint32_t t = UINT32_MAX;
    for (uint32_t i = 0; i < c.memp.memoryTypeCount && t == UINT32_MAX; ++i) {
        const VkMemoryPropertyFlags f = c.memp.memoryTypes[i].propertyFlags;
        if (!(mr.memoryTypeBits & (1u << i))) continue;
        if (host ? (f & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                 : f == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) t = i;
    }
    if (t == UINT32_MAX) { fprintf(stderr, "no suitable memory type\n"); exit(1); }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = t;
    VKC(vkAllocateMemory(c.dev, &ai, nullptr, &b.m));
    VKC(vkBindBufferMemory(c.dev, b.b, b.m, 0));
    if (host) VKC(vkMapMemory(c.dev, b.m, 0, VK_WHOLE_SIZE, 0, &b.p));
    return b;
}
static void free_buf(Ctx & c, Buf & b) { if (b.p) vkUnmapMemory(c.dev, b.m); vkDestroyBuffer(c.dev, b.b, nullptr); vkFreeMemory(c.dev, b.m, nullptr); b = Buf{}; }

// one-shot buffer copy / fill (up: host staging -> VRAM, or back)
static void copy_buf(Ctx & c, Buf & src, Buf & dst, size_t size, bool fill_zero = false) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = c.pool; cai.commandBufferCount = 1;
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(c.dev, &cai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &bi));
    if (fill_zero) vkCmdFillBuffer(cb, dst.b, 0, VK_WHOLE_SIZE, 0);
    else { VkBufferCopy r{0, 0, size}; vkCmdCopyBuffer(cb, src.b, dst.b, 1, &r); }
    VKC(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VKC(vkQueueSubmit(c.q, 1, &si, VK_NULL_HANDLE)); VKC(vkQueueWaitIdle(c.q));
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
}

static const char * ctype_name(VkComponentTypeKHR t) {
    switch ((int) t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16"; case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "s8"; case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "s32"; case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
        case 1000141000: return "bf16"; case 1000491002: return "fp8e4m3"; case 1000491003: return "fp8e5m2";
        default: { static char buf[32]; snprintf(buf, sizeof buf, "type%d", (int) t); return buf; }
    }
}

static void init(Ctx & c, int devidx) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VKC(vkCreateInstance(&ici, nullptr, &c.inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(c.inst, &n, pds.data());
    if (devidx >= (int) n) { fprintf(stderr, "device %d not found (%u devices)\n", devidx, n); exit(1); }
    c.pd = pds[devidx];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.pd, &props);
    c.ts_period = props.limits.timestampPeriod;
    vkGetPhysicalDeviceMemoryProperties(c.pd, &c.memp);
    printf("device %d: %s (driver %u.%u.%u)\n", devidx, props.deviceName, VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion));

    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne); vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &ne, exts.data());
    auto has = [&](const char * e) { for (auto & x : exts) if (!strcmp(x.extensionName, e)) return true; return false; };
    if (!has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) { fprintf(stderr, "no VK_KHR_cooperative_matrix\n"); exit(1); }
    c.fp8 = has("VK_EXT_shader_float8");

    auto getp = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) vkGetInstanceProcAddr(c.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t np = 0; getp(c.pd, &np, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> cps(np, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    getp(c.pd, &np, cps.data());
    printf("cooperative matrix properties (%u):\n", np);
    for (auto & p : cps)
        printf("  %ux%ux%u  A=%s B=%s C=%s R=%s  scope=%s sat=%u\n", p.MSize, p.NSize, p.KSize, ctype_name(p.AType), ctype_name(p.BType),
               ctype_name(p.CType), ctype_name(p.ResultType), p.scope == VK_SCOPE_SUBGROUP_KHR ? "subgroup" : "other", p.saturatingAccumulation);
    printf("VK_EXT_shader_float8: %s\n", c.fp8 ? "yes" : "no");

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nq); vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nq, qfs.data());
    for (uint32_t i = 0; i < nq; ++i) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }

    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    f8.shaderFloat8 = VK_TRUE; f8.shaderFloat8CooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cm.cooperativeMatrix = VK_TRUE; cm.pNext = c.fp8 ? &f8 : nullptr;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.subgroupSizeControl = VK_TRUE; v13.computeFullSubgroups = VK_TRUE; v13.pNext = &cm;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.shaderFloat16 = VK_TRUE; v12.shaderInt8 = VK_TRUE; v12.storageBuffer8BitAccess = VK_TRUE;
    v12.vulkanMemoryModel = VK_TRUE; v12.vulkanMemoryModelDeviceScope = VK_TRUE; v12.pNext = &v13;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.storageBuffer16BitAccess = VK_TRUE; v11.pNext = &v12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.features.shaderInt16 = VK_TRUE; f2.pNext = &v11;

    std::vector<const char *> dext = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    if (c.fp8) dext.push_back("VK_EXT_shader_float8");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = c.qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t) dext.size(); dci.ppEnabledExtensionNames = dext.data();
    VKC(vkCreateDevice(c.pd, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.q);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = c.qf; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VKC(vkCreateCommandPool(c.dev, &cpi, nullptr, &c.pool));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
    VKC(vkCreateQueryPool(c.dev, &qpi, nullptr, &c.qp));
}

// ---- shader compile + pipeline ----
struct Pipe { VkPipeline p{}; VkPipelineLayout pl{}; VkDescriptorSetLayout dsl{}; VkDescriptorPool dp{}; VkDescriptorSet ds{}; int nbind = 0; };

static std::vector<uint32_t> compile(const std::string & src, const std::string & defs) {
    std::hash<std::string> h;
    std::string out = g_cache + "/" + src + "-" + std::to_string(h(defs + src)) + ".spv";
    std::string cmd = "glslc --target-env=vulkan1.3 -O -fshader-stage=compute " + defs + " " + g_shader_dir + "/" + src + " -o " + out;
    if (system(cmd.c_str()) != 0) { fprintf(stderr, "glslc failed: %s\n", cmd.c_str()); exit(1); }
    std::ifstream f(out, std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint32_t> code(sz / 4); f.read((char *) code.data(), sz);
    return code;
}

static Pipe make_pipe(Ctx & c, const std::string & src, const std::string & defs, int nbind, uint32_t pc_size, uint32_t sg) {
    Pipe P; P.nbind = nbind;
    auto code = compile(src, defs);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size() * 4; smi.pCode = code.data();
    VkShaderModule sm; VKC(vkCreateShaderModule(c.dev, &smi, nullptr, &sm));
    std::vector<VkDescriptorSetLayoutBinding> b(nbind);
    for (int i = 0; i < nbind; ++i) b[i] = { (uint32_t) i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = nbind; dli.pBindings = b.data();
    VKC(vkCreateDescriptorSetLayout(c.dev, &dli, nullptr, &P.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &P.dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VKC(vkCreatePipelineLayout(c.dev, &pli, nullptr, &P.pl));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; rs.requiredSubgroupSize = sg;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    if (sg) { cpi.stage.pNext = &rs; cpi.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT; }
    cpi.layout = P.pl;
    VKC(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &P.p));
    vkDestroyShaderModule(c.dev, sm, nullptr);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t) nbind};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VKC(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &P.dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = P.dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &P.dsl;
    VKC(vkAllocateDescriptorSets(c.dev, &dai, &P.ds));
    return P;
}
static void free_pipe(Ctx & c, Pipe & P) {
    vkDestroyPipeline(c.dev, P.p, nullptr); vkDestroyPipelineLayout(c.dev, P.pl, nullptr);
    vkDestroyDescriptorPool(c.dev, P.dp, nullptr); vkDestroyDescriptorSetLayout(c.dev, P.dsl, nullptr);
}
static void bind(Ctx & c, Pipe & P, std::vector<Buf *> bufs) {
    std::vector<VkDescriptorBufferInfo> bi(bufs.size()); std::vector<VkWriteDescriptorSet> w(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        bi[i] = { bufs[i]->b, 0, VK_WHOLE_SIZE };
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet = P.ds; w[i].dstBinding = (uint32_t) i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(c.dev, (uint32_t) w.size(), w.data(), 0, nullptr);
}

// Runs `reps` back-to-back dispatches (with barriers) and returns seconds per dispatch.
static double run(Ctx & c, Pipe & P, const void * pc, uint32_t pcs, uint32_t gx, uint32_t gy, int reps) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = c.pool; cai.commandBufferCount = 1;
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(c.dev, &cai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, c.qp, 0, 2);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pl, 0, 1, &P.ds, 0, nullptr);
    vkCmdPushConstants(cb, P.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcs, pc);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c.qp, 0);
    for (int r = 0; r < reps; ++r) {
        vkCmdDispatch(cb, gx, gy, 1);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c.qp, 1);
    VKC(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VKC(vkQueueSubmit(c.q, 1, &si, VK_NULL_HANDLE)); VKC(vkQueueWaitIdle(c.q));
    uint64_t ts[2]; VKC(vkGetQueryPoolResults(c.dev, c.qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    return (ts[1] - ts[0]) * (double) c.ts_period * 1e-9 / reps;
}

// ---- modes ----
static void peak(Ctx & c, const std::string & type, uint32_t iters, uint32_t sg_only) {
    struct T { const char * name, * a, * c, * extra; };
    std::vector<T> ts = { {"f16", "float16_t", "float", ""}, {"f16acc16", "float16_t", "float16_t", ""},
                          {"s8", "int8_t", "int32_t", ""}, {"fp8", "floate4m3_t", "float", "-DFP8"} };
    Buf o = make_buf(c, 1 << 20);
    for (auto & t : ts) {
        if (type != "all" && type != t.name) continue;
        if (!strcmp(t.name, "fp8") && !c.fp8) { printf("peak %-9s skipped: VK_EXT_shader_float8 not exposed\n", t.name); continue; }
        for (uint32_t sg : {32u, 64u}) {
            if (sg_only && sg != sg_only) continue;
            for (int nacc : {4, 8}) {
                const uint32_t wg = 256;
                std::string defs = std::string("-DA_T=") + t.a + " -DC_T=" + t.c + " -DNACC=" + std::to_string(nacc) + " -DWG=256 " + t.extra;
                Pipe P = make_pipe(c, "peak.comp", defs, 1, 8, sg);
                bind(c, P, { &o });
                struct { float x; uint32_t it; } pc = { 0.5f, iters };
                const uint32_t groups = 64 * 16;
                run(c, P, &pc, 8, groups, 1, 1);
                double best = 1e9;
                for (int r = 0; r < 5; ++r) best = std::min(best, run(c, P, &pc, 8, groups, 1, 1));
                const double ops = 2.0 * 16 * 16 * 16 * nacc * (double) iters * groups * (wg / sg);
                printf("peak %-9s wave%-2u nacc=%d : %8.1f %s\n", t.name, sg, nacc, ops / best / 1e12, !strcmp(t.name, "s8") ? "TOPS" : "TFLOPS");
                free_pipe(c, P);
            }
        }
    }
    free_buf(c, o);
}

static void bw(Ctx & c) {
    const size_t bytes = size_t(1) << 30;
    Buf src = make_buf(c, bytes), dst = make_buf(c, 64 << 20);
    copy_buf(c, src, src, 0, true);
    Pipe D = make_pipe(c, "bw.comp", "-DMODE_DRAM", 2, 8, 0); bind(c, D, { &src, &dst });
    struct { uint32_t n, it; } pc = { (uint32_t) (bytes / 16), 0 };
    for (uint32_t groups : {64u * 8, 64u * 32, 64u * 128}) {
        run(c, D, &pc, 8, groups, 1, 1);
        double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(c, D, &pc, 8, groups, 1, 2));
        printf("dram read  groups=%-5u : %7.1f GB/s\n", groups, bytes / best / 1e9);
    }
    Pipe L = make_pipe(c, "bw.comp", "-DMODE_LDS", 2, 8, 0); bind(c, L, { &src, &dst });
    pc = { 0, 4096 };
    const uint32_t groups = 64 * 16;
    run(c, L, &pc, 8, groups, 1, 1);
    double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(c, L, &pc, 8, groups, 1, 1));
    const double lb = 16.0 * 8 * 4096 * 256 * groups;
    printf("lds read   groups=%-5u : %7.1f GB/s (%.1f B/clk/CU at 2.35 GHz)\n", groups, lb / best / 1e9, lb / best / 64 / 2.35e9);
    free_pipe(c, D); free_pipe(c, L); free_buf(c, src); free_buf(c, dst);
}

struct GemmCfg { bool s8; uint32_t bm, bn, bk, sg, pad; bool nfast; };

static std::string gemm_defs(const GemmCfg & g, uint32_t & wm, uint32_t & wn) {
    const uint32_t warps = 256 / g.sg;              // 256-thread workgroups
    // warp grid: prefer square-ish warp tiles
    wm = 1; wn = warps;
    for (uint32_t a = 1; a <= warps; a *= 2) {
        uint32_t b = warps / a;
        if (g.bm / a < 16 || g.bn / b < 16) continue;
        double r = double(g.bm / a) / (g.bn / b);
        double cur = double(g.bm / wm) / (g.bn / wn);
        if (std::fabs(std::log(r)) < std::fabs(std::log(cur)) || g.bn / wn < 16) { wm = a; wn = b; }
    }
    char buf[256];
    snprintf(buf, sizeof buf, "%s%s -DBM=%u -DBN=%u -DBK=%u -DWM_WARPS=%u -DWN_WARPS=%u -DSG=%u -DPAD_U=%u",
             g.s8 ? "-DS8" : "", g.nfast ? " -DN_FAST" : "", g.bm, g.bn, g.bk, wm, wn, g.sg, g.pad);
    return buf;
}

static bool gemm_valid(const GemmCfg & g) {
    const uint32_t eb = g.s8 ? 1 : 2, row_u = g.bk * eb / 4;
    return 2u * (g.bm + g.bn) * (row_u + g.pad) * 4 <= 65536 && (g.bk * eb) % 16 == 0;
}

static uint16_t f2h(float f) { // round-to-nearest f32 -> f16 (normal range only; inputs are in [-1,1])
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t) s;
    uint32_t h = s | (e << 10) | (m >> 13);
    if ((m >> 12) & 1) h++;
    return (uint16_t) h;
}
static float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x = e == 0 ? s : (s | ((e - 15 + 127) << 23) | (m << 13)); float f; memcpy(&f, &x, 4); return f;
}

// returns TFLOPS/TOPS (or -1 if verification failed)
static double gemm(Ctx & c, const GemmCfg & g, uint32_t M, uint32_t N, uint32_t K, bool verify, bool quiet) {
    uint32_t wm, wn; std::string defs = gemm_defs(g, wm, wn);
    const uint32_t eb = g.s8 ? 1 : 2;
    Buf A = make_buf(c, (size_t) M * K * eb, true), B = make_buf(c, (size_t) N * K * eb, true), C = make_buf(c, (size_t) M * N * 4, true);
    Buf dA = make_buf(c, A.size), dB = make_buf(c, B.size), dC = make_buf(c, C.size);
    std::mt19937 rng(42);
    if (g.s8) {
        std::uniform_int_distribution<int> d(-16, 15);
        for (size_t i = 0; i < (size_t) M * K; ++i) ((int8_t *) A.p)[i] = (int8_t) d(rng);
        for (size_t i = 0; i < (size_t) N * K; ++i) ((int8_t *) B.p)[i] = (int8_t) d(rng);
    } else {
        std::uniform_real_distribution<float> d(-1, 1);
        for (size_t i = 0; i < (size_t) M * K; ++i) ((uint16_t *) A.p)[i] = f2h(d(rng));
        for (size_t i = 0; i < (size_t) N * K; ++i) ((uint16_t *) B.p)[i] = f2h(d(rng));
    }
    Pipe P = make_pipe(c, "gemm.comp", defs, 3, 12, g.sg);
    copy_buf(c, A, dA, A.size); copy_buf(c, B, dB, B.size);
    bind(c, P, { &dA, &dB, &dC });
    uint32_t pc[3] = { M, N, K };
    const uint32_t gx = g.nfast ? N / g.bn : M / g.bm, gy = g.nfast ? M / g.bm : N / g.bn;
    double res = 0;
    if (verify) {
        copy_buf(c, dC, dC, 0, true);
        run(c, P, pc, 12, gx, gy, 1);
        copy_buf(c, dC, C, C.size);
        double maxerr = 0, maxref = 0; size_t bad = 0;
        for (uint32_t n = 0; n < N; ++n) for (uint32_t m = 0; m < M; ++m) {
            double ref = 0;
            for (uint32_t k = 0; k < K; ++k) {
                if (g.s8) ref += (double) ((int8_t *) A.p)[(size_t) m * K + k] * ((int8_t *) B.p)[(size_t) n * K + k];
                else      ref += (double) h2f(((uint16_t *) A.p)[(size_t) m * K + k]) * h2f(((uint16_t *) B.p)[(size_t) n * K + k]);
            }
            double got = g.s8 ? (double) ((int32_t *) C.p)[(size_t) n * M + m] : (double) ((float *) C.p)[(size_t) n * M + m];
            maxerr = std::max(maxerr, std::fabs(got - ref)); maxref = std::max(maxref, std::fabs(ref));
            if (g.s8 ? got != ref : std::fabs(got - ref) > 1e-3 * (1 + std::fabs(ref))) bad++;
        }
        printf("verify %s M=%u N=%u K=%u %s: max_abs_err=%.3g (max |ref| %.3g), mismatches=%zu -> %s\n", g.s8 ? "s8" : "f16", M, N, K,
               defs.c_str(), maxerr, maxref, bad, bad ? "FAIL" : "PASS");
        res = bad ? -1 : 0;
    } else {
        run(c, P, pc, 12, gx, gy, 2);
        double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(c, P, pc, 12, gx, gy, 5));
        res = 2.0 * M * N * K / best / 1e12;
        if (!quiet) printf("gemm %-3s %5ux%-4ux%-5u tile=%ux%u bk=%u wave%u warps=%ux%u pad=%u order=%s : %7.1f %s  (%.3f ms)\n", g.s8 ? "s8" : "f16",
                           M, N, K, g.bm, g.bn, g.bk, g.sg, wm, wn, g.pad, g.nfast ? "n-fast" : "m-fast", res, g.s8 ? "TOPS" : "TFLOPS", best * 1e3);
    }
    free_pipe(c, P); free_buf(c, A); free_buf(c, B); free_buf(c, C); free_buf(c, dA); free_buf(c, dB); free_buf(c, dC);
    return res;
}

int main(int argc, char ** argv) {
    bool nfast = true;
    int dev = 1; std::string type, gemm_t; uint32_t iters = 4096, M = 17408, N = 512, K = 5120, bm = 128, bn = 128, bk = 32, sg = 0, pad = 4;
    bool do_bw = false, sweep = false, verify = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--device") dev = std::stoi(nx()); else if (a == "--type") type = nx(); else if (a == "--iters") iters = std::stoul(nx());
        else if (a == "--bw") do_bw = true; else if (a == "--gemm") gemm_t = nx();
        else if (a == "--m") M = std::stoul(nx()); else if (a == "--n") N = std::stoul(nx()); else if (a == "--k") K = std::stoul(nx());
        else if (a == "--tile") { std::string t = nx(); sscanf(t.c_str(), "%ux%u", &bm, &bn); }
        else if (a == "--bk") bk = std::stoul(nx()); else if (a == "--sg") sg = std::stoul(nx()); else if (a == "--pad") pad = std::stoul(nx());
        else if (a == "--sweep") sweep = true; else if (a == "--verify") verify = true;
        else if (a == "--order") nfast = nx() == "n";
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    const char * sd = getenv("WMMA_SHADER_DIR");
    if (sd) g_shader_dir = sd; else { std::string e = argv[0]; g_shader_dir = e.substr(0, e.find_last_of('/') + 1); if (g_shader_dir.empty()) g_shader_dir = "."; }
    const char * cd = getenv("WMMA_CACHE"); g_cache = cd ? cd : "/tmp/wmma_peak_cache";
    system(("mkdir -p " + g_cache).c_str());
    Ctx c; init(c, dev);
    if (!type.empty()) peak(c, type, iters, sg);
    if (do_bw) bw(c);
    if (!gemm_t.empty()) {
        const bool s8 = gemm_t == "s8";
        if (!sweep) {
            GemmCfg g{ s8, bm, bn, bk, sg ? sg : 64, pad, nfast };
            if (!gemm_valid(g)) { fprintf(stderr, "config exceeds 64 KiB LDS\n"); return 1; }
            if (verify) gemm(c, g, 512, 256, 512, true, false);
            gemm(c, g, M, N, K, false, false);
        } else {
            GemmCfg best{}; double bt = 0;
            for (uint32_t tbm : {128u, 256u}) for (uint32_t tsg : {32u, 64u}) for (uint32_t tbk : {32u, 64u}) for (uint32_t tpad : {0u, 2u, 4u}) for (bool tnf : {false, true}) {
                if (tpad == 0 && tnf) continue;  // pad 0 lost every m-fast comparison
                GemmCfg g{ s8, tbm, 128, tbk, tsg, tpad, tnf };
                if (!gemm_valid(g)) continue;
                if (verify && gemm(c, g, 512, 256, 512, true, true) < 0) continue;
                double t = gemm(c, g, M, N, K, false, false);
                if (t > bt) { bt = t; best = g; }
            }
            printf("BEST gemm %s %ux%ux%u: tile=%ux%u bk=%u wave%u pad=%u order=%s -> %.1f %s\n", s8 ? "s8" : "f16", M, N, K, best.bm, best.bn, best.bk,
                   best.sg, best.pad, best.nfast ? "n-fast" : "m-fast", bt, s8 ? "TOPS" : "TFLOPS");
        }
    }
    return 0;
}
