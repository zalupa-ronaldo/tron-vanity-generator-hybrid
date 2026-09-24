#include "vulkan_backend.h"

#include "crypto.h"
#include "rng.h"

#include "vulkan_curve_spv.h"
#include "vulkan_curve_batch4_spv.h"
#include "vulkan_keccak_spv.h"
#include "vulkan_checksum_spv.h"
#include "vulkan_base58_spv.h"
#include "vulkan_match_spv.h"
#include "test_dictionary.h"

#include <secp256k1.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
constexpr uint32_t kDefaultBatchKeys = 1u << 17;
constexpr uint32_t kSelfTestBatchKeys = 1u << 15;
constexpr uint32_t kBaseWindowKeys = 1u << 22;
constexpr uint32_t kGroupSize = 64;
constexpr uint32_t kBindings = 10;
constexpr uint32_t kRingWords = 29;
constexpr uint32_t kResidentDispatches = 4;
constexpr uint32_t kStageCount = 5;
constexpr uint32_t kQueryStride = kStageCount + 1;
constexpr std::array<uint32_t, kBindings> kWordsPerKey = {
    16, 6, 7, 9, 0, 18, 0, 0, 0, kRingWords
};

struct SecretScalar {
    std::array<unsigned char, 32> bytes{};
    SecretScalar() = default;
    SecretScalar(const SecretScalar&) = delete;
    SecretScalar& operator=(const SecretScalar&) = delete;
    ~SecretScalar() {
        volatile unsigned char* p = bytes.data();
        for (size_t i = 0; i < bytes.size(); ++i) p[i] = 0;
    }
    unsigned char* data() { return bytes.data(); }
    const unsigned char* data() const { return bytes.data(); }
};

bool addOffset(const unsigned char base[32], uint32_t offset, unsigned char result[32]) {
    uint64_t carry = offset;
    for (int i = 31; i >= 0; --i) {
        carry += base[i];
        result[i] = static_cast<unsigned char>(carry);
        carry >>= 8;
    }
    return carry == 0;
}

std::string hexUpper(const unsigned char* data, size_t size) {
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string result;
    result.reserve(size * 2);
    for (size_t i = 0; i < size; ++i) {
        result += digits[data[i] >> 4];
        result += digits[data[i] & 15];
    }
    return result;
}

struct Candidate {
    uint32_t gid = 0, count = 0, flags = 0;
    std::array<uint32_t, 16> ids{};
    std::string address;
};

struct PushConstants {
    uint32_t count, dfaOffset, outStartOffset, outLenOffset, outIdsOffset;
    uint32_t curveMode, offsetBase, ringMode, ringCapacity, recordBase;
    std::array<uint32_t, 16> baseWords{};
};
static_assert(sizeof(PushConstants) == 104, "Vulkan push-constant ABI changed");

class VulkanEngine {
public:
    explicit VulkanEngine(std::shared_ptr<const Dictionary> dictionary,
                          bool allowSoftware, uint32_t curveBatch, uint32_t batchKeys)
        : dictionary_(std::move(dictionary)), allowSoftware_(allowSoftware),
          curveBatch_(curveBatch), batchKeys_(batchKeys) {}
    VulkanEngine(const VulkanEngine&) = delete;
    VulkanEngine& operator=(const VulkanEngine&) = delete;
    ~VulkanEngine() {
        // On a timeout or device loss, destruction of in-flight Vulkan
        // objects is invalid and vkDeviceWaitIdle may hang. The process exits
        // after the backend reports failure; let the OS reclaim the handles.
        if (abandoned_) return;
        if (device_) vkDeviceWaitIdle(device_);
        if (device_ && mapped_) vkUnmapMemory(device_, memory_);
        if (device_ && fence_) vkDestroyFence(device_, fence_, nullptr);
        if (device_ && queryPool_) vkDestroyQueryPool(device_, queryPool_, nullptr);
        if (device_ && commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (VkPipeline pipeline : pipelines_)
            if (device_ && pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
        for (VkShaderModule shader : shaders_)
            if (device_ && shader) vkDestroyShaderModule(device_, shader, nullptr);
        if (device_ && pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        if (device_ && descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        if (device_ && descriptorLayout_) vkDestroyDescriptorSetLayout(device_, descriptorLayout_, nullptr);
        if (device_ && buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
        if (device_ && memory_) vkFreeMemory(device_, memory_, nullptr);
        if (device_) vkDestroyDevice(device_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    bool init(std::string& error);
    bool scan(const unsigned char basePub[64], uint32_t offsetBase, uint32_t count,
              std::vector<Candidate>& candidates, std::vector<std::string>* allAddresses,
              std::string& error);
    bool scanMany(const unsigned char basePub[64], uint32_t offsetBase, uint32_t count,
                  uint32_t dispatches, std::vector<Candidate>* candidates,
                  std::string& error);
    const std::string& deviceName() const { return deviceName_; }
    bool deviceLocalHostVisible() const { return deviceLocalHostVisible_; }
    bool timestampsSupported() const { return queryPool_ != VK_NULL_HANDLE; }
    void beginProfile() {
        profiling_ = true;
        stageSeconds_.fill(0.0);
        hostSeconds_.fill(0.0);
        gpuSeconds_ = 0.0;
        candidateRecords_ = 0;
    }
    void endProfile() { profiling_ = false; }
    const std::array<double, kStageCount>& stageSeconds() const { return stageSeconds_; }
    const std::array<double, 5>& hostSeconds() const { return hostSeconds_; }
    uint64_t candidateRecords() const { return candidateRecords_; }
    double gpuSeconds() const { return gpuSeconds_; }
    void injectDeviceLossOnceForTest() { injectDeviceLossOnceForTest_ = true; }

private:
    static VkDeviceSize aligned(VkDeviceSize size, VkDeviceSize alignment) {
        return (size + alignment - 1) / alignment * alignment;
    }
    bool createTable(std::vector<uint32_t>& table, std::string& error);
    std::string readAddress(uint32_t gid) const;
    static std::string decodeAddress(const uint32_t* words);
    bool scanInternal(const unsigned char basePub[64], uint32_t offsetBase, uint32_t count,
                      uint32_t dispatches, std::vector<Candidate>* candidates,
                      std::vector<std::string>* allAddresses, std::string& error);

    std::shared_ptr<const Dictionary> dictionary_;
    bool allowSoftware_ = false;
    std::string deviceName_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    VkDescriptorSetLayout descriptorLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    std::array<VkShaderModule, kStageCount> shaders_{};
    std::array<VkPipeline, kStageCount> pipelines_{};
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    std::array<VkDeviceSize, kBindings> offsets_{};
    std::array<VkDeviceSize, kBindings> sizes_{};
    PushConstants constants_{};
    uint32_t ringCapacity_ = 0;
    uint32_t curveBatch_ = 1;
    uint32_t batchKeys_ = kDefaultBatchKeys;
    std::array<double, kStageCount> stageSeconds_{};
    std::array<double, 5> hostSeconds_{};
    uint64_t candidateRecords_ = 0;
    double gpuSeconds_ = 0.0;
    double timestampPeriod_ = 0.0;
    uint32_t timestampValidBits_ = 0;
    bool profiling_ = false;
    bool deviceLocalHostVisible_ = false;
    bool abandoned_ = false;
    bool injectDeviceLossOnceForTest_ = false;
};

bool VulkanEngine::createTable(std::vector<uint32_t>& table, std::string& error) {
    auto* context = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!context) { error = "secp256k1 context creation failed"; return false; }
    table.assign(3u * 256u * 16u, 0u);
    for (uint32_t window = 0; window < 3; ++window) {
        for (uint32_t digit = 1; digit < 256; ++digit) {
            unsigned char scalar[32]{};
            scalar[31 - window] = static_cast<unsigned char>(digit);
            secp256k1_pubkey pub;
            unsigned char serialized[65];
            size_t length = sizeof(serialized);
            if (!secp256k1_ec_pubkey_create(context, &pub, scalar) ||
                !secp256k1_ec_pubkey_serialize(context, serialized, &length, &pub,
                                               SECP256K1_EC_UNCOMPRESSED) || length != 65) {
                secp256k1_context_destroy(context);
                error = "failed to generate Vulkan offset table";
                return false;
            }
            const size_t base = (window * 256u + digit) * 16u;
            for (size_t word = 0; word < 16; ++word) {
                const size_t j = 1 + word * 4;
                table[base + word] = uint32_t(serialized[j]) |
                    (uint32_t(serialized[j + 1]) << 8) |
                    (uint32_t(serialized[j + 2]) << 16) |
                    (uint32_t(serialized[j + 3]) << 24);
            }
        }
    }
    secp256k1_context_destroy(context);
    return true;
}

bool VulkanEngine::init(std::string& error) {
    auto check = [&](VkResult result, const char* call) {
        if (result == VK_SUCCESS) return true;
        error = std::string(call) + " failed: " + std::to_string(result);
        return false;
    };
    if (!dictionary_ || dictionary_->words.empty() || dictionary_->dfa.empty()) {
        error = "Vulkan requires a nonempty dictionary";
        return false;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "TRON vanity Vulkan resident";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;
    if (!check(vkCreateInstance(&instanceInfo, nullptr, &instance_), "vkCreateInstance")) return false;
    uint32_t count = 0;
    if (!check(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
               "vkEnumeratePhysicalDevices") || count == 0) {
        if (error.empty()) error = "no Vulkan physical devices";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (!check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
               "vkEnumeratePhysicalDevices")) return false;
    int bestScore = -1;
    VkPhysicalDeviceProperties selected{};
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceFeatures features{};
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceFeatures(candidate, &features);
        vkGetPhysicalDeviceProperties(candidate, &properties);
        const auto& limits = properties.limits;
        if ((!allowSoftware_ && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) ||
            !features.shaderInt64 || limits.maxComputeWorkGroupInvocations < kGroupSize ||
            limits.maxComputeWorkGroupSize[0] < kGroupSize ||
            limits.maxPerStageDescriptorStorageBuffers < kBindings ||
            limits.maxDescriptorSetStorageBuffers < kBindings ||
            limits.maxPushConstantsSize < sizeof(PushConstants)) continue;
        uint32_t familiesCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familiesCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familiesCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familiesCount, families.data());
        for (uint32_t i = 0; i < familiesCount; ++i) {
            if (!families[i].queueCount || !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 :
                              properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
            if (score > bestScore) {
                bestScore = score;
                physical_ = candidate;
                queueFamily_ = i;
                selected = properties;
            }
            break;
        }
    }
    if (!physical_) {
        error = "no hardware Vulkan compute device with shaderInt64 and 11 storage buffers";
        return false;
    }
    deviceName_ = selected.deviceName;
    uint32_t selectedFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &selectedFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> selectedFamilies(selectedFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &selectedFamilyCount,
                                             selectedFamilies.data());
    timestampValidBits_ = selectedFamilies[queueFamily_].timestampValidBits;
    timestampPeriod_ = selected.limits.timestampPeriod;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    if (!check(vkCreateDevice(physical_, &deviceInfo, nullptr, &device_), "vkCreateDevice")) return false;
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

    std::vector<uint32_t> automaton;
    automaton.insert(automaton.end(), dictionary_->dfa.begin(), dictionary_->dfa.end());
    constants_.outStartOffset = static_cast<uint32_t>(automaton.size());
    automaton.insert(automaton.end(), dictionary_->outStart.begin(), dictionary_->outStart.end());
    constants_.outLenOffset = static_cast<uint32_t>(automaton.size());
    automaton.insert(automaton.end(), dictionary_->outLen.begin(), dictionary_->outLen.end());
    constants_.outIdsOffset = static_cast<uint32_t>(automaton.size());
    automaton.insert(automaton.end(), dictionary_->outIds.begin(), dictionary_->outIds.end());
    if (automaton.size() > std::numeric_limits<uint32_t>::max()) {
        error = "Vulkan dictionary exceeds 32-bit index space"; return false;
    }
    std::vector<uint32_t> table;
    if (!createTable(table, error)) return false;
    ringCapacity_ = batchKeys_ * kResidentDispatches;
    for (uint32_t i = 0; i < kBindings; ++i) {
        sizes_[i] = i == 4 ? automaton.size() * sizeof(uint32_t) :
                    i == 6 ? 64 : i == 7 ? table.size() * sizeof(uint32_t) :
                    i == 8 ? 2 * sizeof(uint32_t) :
                    i == 5 ? 18 * sizeof(uint32_t) :
                    i == 9 ? VkDeviceSize(ringCapacity_) * kRingWords * sizeof(uint32_t) :
                    VkDeviceSize(batchKeys_) * kWordsPerKey[i] * sizeof(uint32_t);
    }
    const VkDeviceSize alignment = std::max<VkDeviceSize>(4, selected.limits.minStorageBufferOffsetAlignment);
    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < kBindings; ++i) {
        total = aligned(total, alignment);
        offsets_[i] = total;
        total += sizes_[i];
        if (sizes_[i] > selected.limits.maxStorageBufferRange) {
            error = "Vulkan storage buffer range too small"; return false;
        }
    }
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = total;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer_), "vkCreateBuffer")) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProps{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &memoryProps);
    uint32_t memoryType = memoryProps.memoryTypeCount;
    int memoryScore = -1;
    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memoryProps.memoryTypes[i].propertyFlags;
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (flags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            // On discrete GPUs, a host-visible GTT heap can make every
            // intermediate shader access traverse PCIe. Prefer BAR-mapped
            // device-local memory when the driver exposes it.
            const int score = ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 2 : 0) +
                              ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
            if (score > memoryScore) {
                memoryScore = score;
                memoryType = i;
                deviceLocalHostVisible_ = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            }
        }
    }
    if (memoryType == memoryProps.memoryTypeCount) {
        error = "no coherent host-visible Vulkan memory"; return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    if (!check(vkAllocateMemory(device_, &allocation, nullptr, &memory_), "vkAllocateMemory") ||
        !check(vkBindBufferMemory(device_, buffer_, memory_, 0), "vkBindBufferMemory") ||
        !check(vkMapMemory(device_, memory_, 0, total, 0, &mapped_), "vkMapMemory")) return false;
    auto* bytes = static_cast<unsigned char*>(mapped_);
    std::memset(bytes, 0, static_cast<size_t>(total));
    std::memcpy(bytes + offsets_[4], automaton.data(), sizes_[4]);
    std::memcpy(bytes + offsets_[7], table.data(), sizes_[7]);
    constants_.curveMode = curveBatch_ == 4 ? 2u : 1u;
    constants_.ringMode = 1;
    constants_.ringCapacity = ringCapacity_;
    std::array<VkDescriptorSetLayoutBinding, kBindings> bindings{};
    for (uint32_t i = 0; i < kBindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = kBindings;
    layoutInfo.pBindings = bindings.data();
    if (!check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorLayout_),
               "vkCreateDescriptorSetLayout")) return false;
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (!check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_),
               "vkCreateDescriptorPool")) return false;
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptorPool_;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &descriptorLayout_;
    if (!check(vkAllocateDescriptorSets(device_, &setInfo, &descriptorSet_), "vkAllocateDescriptorSets")) return false;
    std::array<VkDescriptorBufferInfo, kBindings> ranges{};
    std::array<VkWriteDescriptorSet, kBindings> writes{};
    for (uint32_t i = 0; i < kBindings; ++i) {
        ranges[i] = {buffer_, offsets_[i], sizes_[i]};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &ranges[i];
    }
    vkUpdateDescriptorSets(device_, kBindings, writes.data(), 0, nullptr);
    const std::array<const uint32_t*, kStageCount> codes = {
        curveBatch_ == 4 ? kVulkanCurveBatch4Spv : kVulkanCurveSpv,
        kVulkanKeccakSpv, kVulkanChecksumSpv,
        kVulkanBase58Spv, kVulkanMatchSpv
    };
    const std::array<size_t, kStageCount> codeSizes = {
        curveBatch_ == 4 ? sizeof(kVulkanCurveBatch4Spv) : sizeof(kVulkanCurveSpv),
        sizeof(kVulkanKeccakSpv), sizeof(kVulkanChecksumSpv),
        sizeof(kVulkanBase58Spv), sizeof(kVulkanMatchSpv)
    };
    for (uint32_t i = 0; i < kStageCount; ++i) {
        VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = codeSizes[i];
        shaderInfo.pCode = codes[i];
        if (!check(vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaders_[i]),
                   "vkCreateShaderModule")) return false;
    }
    VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &descriptorLayout_;
    pipelineLayout.pushConstantRangeCount = 1;
    pipelineLayout.pPushConstantRanges = &pushRange;
    if (!check(vkCreatePipelineLayout(device_, &pipelineLayout, nullptr, &pipelineLayout_),
               "vkCreatePipelineLayout")) return false;
    for (uint32_t i = 0; i < kStageCount; ++i) {
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaders_[i];
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout_;
        if (!check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                            nullptr, &pipelines_[i]), "vkCreateComputePipelines")) return false;
    }
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = queueFamily_;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (!check(vkCreateCommandPool(device_, &pool, nullptr, &commandPool_), "vkCreateCommandPool")) return false;
    VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandInfo.commandPool = commandPool_;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    if (!check(vkAllocateCommandBuffers(device_, &commandInfo, &command_), "vkAllocateCommandBuffers")) return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence")) return false;
    if (selected.limits.timestampComputeAndGraphics && timestampValidBits_ >= 36 &&
        timestampPeriod_ > 0.0) {
        VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = kResidentDispatches * kQueryStride;
        // Timestamp queries are optional instrumentation, not required for
        // correctness or normal wallet generation.
        if (vkCreateQueryPool(device_, &queryInfo, nullptr, &queryPool_) != VK_SUCCESS)
            queryPool_ = VK_NULL_HANDLE;
    }
    return true;
}

std::string VulkanEngine::decodeAddress(const uint32_t* words) {
    std::string address(34, '\0');
    for (uint32_t i = 0; i < 34; ++i)
        address[i] = static_cast<char>((words[i / 4] >> (8 * (i % 4))) & 255u);
    return address;
}

std::string VulkanEngine::readAddress(uint32_t gid) const {
    const auto* words = reinterpret_cast<const uint32_t*>(
        static_cast<const unsigned char*>(mapped_) + offsets_[3]) + size_t(gid) * 9;
    return decodeAddress(words);
}

bool VulkanEngine::scan(const unsigned char basePub[64], uint32_t offsetBase, uint32_t count,
                        std::vector<Candidate>& candidates,
                        std::vector<std::string>* allAddresses, std::string& error) {
    return scanInternal(basePub, offsetBase, count, 1, &candidates, allAddresses, error);
}

bool VulkanEngine::scanMany(const unsigned char basePub[64], uint32_t offsetBase, uint32_t count,
                            uint32_t dispatches, std::vector<Candidate>* candidates,
                            std::string& error) {
    return scanInternal(basePub, offsetBase, count, dispatches, candidates, nullptr, error);
}

bool VulkanEngine::scanInternal(const unsigned char basePub[64], uint32_t offsetBase,
                                uint32_t count, uint32_t dispatches,
                                std::vector<Candidate>* candidates,
                                std::vector<std::string>* allAddresses, std::string& error) {
    auto check = [&](VkResult result, const char* call) {
        if (result == VK_SUCCESS) return true;
        if (result == VK_ERROR_DEVICE_LOST) abandoned_ = true;
        error = std::string(call) + " failed: " + std::to_string(result);
        return false;
    };
    if (abandoned_) { error = "Vulkan device was abandoned after a GPU failure"; return false; }
    const uint64_t totalKeys64 = uint64_t(count) * dispatches;
    if (!mapped_ || count == 0 || count > batchKeys_ || dispatches == 0 ||
        dispatches > kResidentDispatches || totalKeys64 > ringCapacity_ ||
        totalKeys64 > kBaseWindowKeys || offsetBase > kBaseWindowKeys - totalKeys64 ||
        (allAddresses && dispatches != 1)) {
        error = "invalid Vulkan scan range"; return false;
    }
    const uint32_t totalKeys = static_cast<uint32_t>(totalKeys64);
    const auto profileStart = std::chrono::steady_clock::now();
    auto* bytes = static_cast<unsigned char*>(mapped_);
    // The base point is immutable for every invocation in this submit. Keep
    // it in the push-constant block so the curve shader can read it from the
    // constant path instead of issuing one storage-buffer load per key.
    std::memcpy(constants_.baseWords.data(), basePub,
                constants_.baseWords.size() * sizeof(uint32_t));
    if (allAddresses) {
        // Deterministic no-wallet tests must not pass because an earlier
        // dispatch left the expected addresses in the same mapped buffer.
        for (uint32_t binding = 0; binding < 4; ++binding)
            std::memset(bytes + offsets_[binding], 0xa5, sizes_[binding]);
    }
    constants_.count = count;
    const auto setupDone = std::chrono::steady_clock::now();
    // There is only one primary command buffer. Resetting the buffer directly
    // avoids invalidating the whole pool on every bounded scan and lets the
    // driver retain pool-level bookkeeping between submits.
    if (!check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer") ||
        !check(vkResetFences(device_, 1, &fence_), "vkResetFences")) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer")) return false;
    const bool queryThisBatch = profiling_ && queryPool_ != VK_NULL_HANDLE;
    if (queryThisBatch) {
        vkCmdResetQueryPool(command_, queryPool_, 0, dispatches * kQueryStride);
    }
    // The result counter is device-owned state. Clearing it on the command
    // stream avoids a host write to the mapped result buffer for every batch.
    vkCmdFillBuffer(command_, buffer_, offsets_[8], sizes_[8], 0u);
    VkMemoryBarrier initBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    initBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    initBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &initBarrier,
                         0, nullptr, 0, nullptr);
    vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                            0, 1, &descriptorSet_, 0, nullptr);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    for (uint32_t pass = 0; pass < dispatches; ++pass) {
        constants_.offsetBase = offsetBase + pass * count;
        constants_.recordBase = pass * count;
        vkCmdPushConstants(command_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants_), &constants_);
        const uint32_t queryBase = pass * kQueryStride;
        if (queryThisBatch)
            vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queryPool_, queryBase);
        for (uint32_t stage = 0; stage < kStageCount; ++stage) {
            vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[stage]);
            const uint32_t invocations = stage == 0 ? (count + curveBatch_ - 1) / curveBatch_ : count;
            vkCmdDispatch(command_, (invocations + kGroupSize - 1) / kGroupSize, 1, 1);
            if (queryThisBatch)
                vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    queryPool_, queryBase + stage + 1);
            if (stage + 1 < kStageCount)
                vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        if (pass + 1 < dispatches) {
            // The next curve pass overwrites the same intermediate buffers,
            // while the ring remains append-only. Make this an explicit
            // shader write -> shader read/write dependency and keep all of it
            // on the GPU.
            VkMemoryBarrier reuseBarrier = barrier;
            reuseBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &reuseBarrier, 0, nullptr, 0, nullptr);
        }
    }
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (!check(vkEndCommandBuffer(command_), "vkEndCommandBuffer")) return false;
    const auto recordDone = std::chrono::steady_clock::now();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    const VkResult submitted = injectDeviceLossOnceForTest_ ? VK_ERROR_DEVICE_LOST :
                               vkQueueSubmit(queue_, 1, &submit, fence_);
    injectDeviceLossOnceForTest_ = false;
    if (!check(submitted, "vkQueueSubmit")) return false;
    const auto submitDone = std::chrono::steady_clock::now();
    const VkResult waited = vkWaitForFences(device_, 1, &fence_, VK_TRUE, 30000000000ull);
    if (waited == VK_TIMEOUT) {
        abandoned_ = true;
        error = "Vulkan dispatch timed out after 30 seconds; stopping without wallet output";
        return false;
    }
    if (!check(waited, "vkWaitForFences")) {
        abandoned_ = true;
        return false;
    }
    const auto fenceDone = std::chrono::steady_clock::now();
    if (queryThisBatch) {
        std::vector<uint64_t> ticks(size_t(dispatches) * kQueryStride);
        if (!check(vkGetQueryPoolResults(device_, queryPool_, 0,
                                         static_cast<uint32_t>(ticks.size()),
                                         ticks.size() * sizeof(uint64_t), ticks.data(),
                                         sizeof(uint64_t),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                   "vkGetQueryPoolResults")) return false;
        const uint64_t mask = timestampValidBits_ == 64 ? ~uint64_t(0) :
                              ((uint64_t(1) << timestampValidBits_) - 1);
        for (uint32_t pass = 0; pass < dispatches; ++pass) {
            const uint32_t queryBase = pass * kQueryStride;
            for (uint32_t stage = 0; stage < kStageCount; ++stage) {
                const uint64_t elapsed = (ticks[queryBase + stage + 1] -
                                          ticks[queryBase + stage]) & mask;
                stageSeconds_[stage] += double(elapsed) * timestampPeriod_ * 1e-9;
            }
            const uint64_t elapsed = (ticks[queryBase + kStageCount] -
                                      ticks[queryBase]) & mask;
            gpuSeconds_ += double(elapsed) * timestampPeriod_ * 1e-9;
        }
    }
    const auto* meta = reinterpret_cast<const uint32_t*>(bytes + offsets_[8]);
    if (meta[0] > totalKeys || (meta[1] & 1u)) {
        error = "Vulkan result ring overflow; stopping to avoid dropped matches"; return false;
    }
    if (candidates) {
        candidates->clear();
        candidates->reserve(meta[0]);
        const auto* records = reinterpret_cast<const uint32_t*>(bytes + offsets_[9]);
        std::unordered_set<uint32_t> seen;
        seen.reserve(meta[0]);
        for (uint32_t slot = 0; slot < meta[0]; ++slot) {
            const uint32_t* record = records + size_t(slot) * kRingWords;
            if (record[0] >= totalKeys || !seen.insert(record[0]).second ||
                record[1] > 16 || (record[1] == 0 && record[2] == 0) ||
                record[3] != 0 || (record[2] & ~1u)) {
                error = "Vulkan result record invalid"; return false;
            }
            Candidate candidate;
            candidate.gid = record[0];
            candidate.count = record[1];
            candidate.flags = record[2];
            std::copy_n(record + 4, 16, candidate.ids.begin());
            candidate.address = decodeAddress(record + 20);
            candidates->push_back(std::move(candidate));
        }
    }
    if (allAddresses) {
        allAddresses->clear();
        allAddresses->reserve(count);
        for (uint32_t gid = 0; gid < count; ++gid) allAddresses->push_back(readAddress(gid));
    }
    if (profiling_) {
        const auto collectDone = std::chrono::steady_clock::now();
        const auto interval = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        hostSeconds_[0] += interval(profileStart, setupDone);
        hostSeconds_[1] += interval(setupDone, recordDone);
        hostSeconds_[2] += interval(recordDone, submitDone);
        hostSeconds_[3] += interval(submitDone, fenceDone);
        hostSeconds_[4] += interval(fenceDone, collectDone);
        candidateRecords_ += meta[0];
    }
    return true;
}

bool publicXY(secp256k1_context* context, const unsigned char scalar[32],
              unsigned char xy[64]) {
    secp256k1_pubkey pub{};
    unsigned char encoded[65]{};
    size_t length = sizeof(encoded);
    if (!secp256k1_ec_seckey_verify(context, scalar) ||
        !secp256k1_ec_pubkey_create(context, &pub, scalar) ||
        !secp256k1_ec_pubkey_serialize(context, encoded, &length, &pub,
                                       SECP256K1_EC_UNCOMPRESSED) || length != sizeof(encoded)) return false;
    std::memcpy(xy, encoded + 1, 64);
    return true;
}

bool verifyScan(VulkanEngine& engine, secp256k1_context* context,
                const Dictionary& dictionary, const unsigned char base[32],
                uint32_t offsetBase, uint32_t count, std::string& error) {
    unsigned char basePub[64]{};
    if (!publicXY(context, base, basePub)) { error = "Vulkan test base key invalid"; return false; }
    std::vector<Candidate> candidates;
    std::vector<std::string> addresses;
    if (!engine.scan(basePub, offsetBase, count, candidates, &addresses, error)) return false;
    std::vector<bool> matched(count, false);
    for (const auto& candidate : candidates) {
        if (candidate.address != addresses[candidate.gid]) {
            error = "Vulkan ring address differs from address stage"; return false;
        }
        const auto ids = dictionary.matchIds(candidate.address);
        if (ids.empty() || candidate.count != std::min<size_t>(16, ids.size()) ||
            candidate.flags != (ids.size() > 16 ? 1u : 0u) ||
            !std::equal(ids.begin(), ids.begin() + candidate.count, candidate.ids.begin())) {
            error = "Vulkan ring differs from CPU dictionary matching"; return false;
        }
        matched[candidate.gid] = true;
    }
    for (uint32_t gid = 0; gid < count; ++gid) {
        SecretScalar scalar;
        unsigned char xy[64]{};
        if (!addOffset(base, offsetBase + gid, scalar.data()) ||
            !publicXY(context, scalar.data(), xy) ||
            tronAddressFromPubXY(xy) != addresses[gid] ||
            matched[gid] != !dictionary.matchIds(addresses[gid]).empty()) {
            error = "Vulkan full address or match differs from CPU reference"; return false;
        }
    }
    return true;
}

bool verifyGroupedScan(VulkanEngine& engine, secp256k1_context* context,
                       const Dictionary& dictionary, const unsigned char base[32],
                       uint32_t offsetBase, uint32_t count, uint32_t dispatches,
                       std::string& error) {
    unsigned char basePub[64]{};
    if (!publicXY(context, base, basePub)) { error = "Vulkan grouped base key invalid"; return false; }
    std::vector<Candidate> candidates;
    if (!engine.scanMany(basePub, offsetBase, count, dispatches, &candidates, error)) return false;
    const uint32_t total = count * dispatches;
    if (candidates.size() != total) {
        error = "Vulkan grouped result ring dropped a full-alphabet record";
        return false;
    }
    std::vector<bool> seen(total, false);
    for (const auto& candidate : candidates) {
        if (candidate.gid >= total || seen[candidate.gid]) {
            error = "Vulkan grouped result IDs are not unique";
            return false;
        }
        seen[candidate.gid] = true;
        SecretScalar scalar;
        unsigned char xy[64]{};
        if (!addOffset(base, offsetBase + candidate.gid, scalar.data()) ||
            !publicXY(context, scalar.data(), xy) ||
            tronAddressFromPubXY(xy) != candidate.address) {
            error = "Vulkan grouped address differs from CPU reference";
            return false;
        }
        const auto ids = dictionary.matchIds(candidate.address);
        if (ids.empty() || candidate.count != std::min<size_t>(16, ids.size()) ||
            candidate.flags != (ids.size() > 16 ? 1u : 0u) ||
            !std::equal(ids.begin(), ids.begin() + candidate.count, candidate.ids.begin())) {
            error = "Vulkan grouped dictionary record differs from CPU reference";
            return false;
        }
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

class VulkanResidentBackend final : public Backend {
public:
    explicit VulkanResidentBackend(std::shared_ptr<const Dictionary> dictionary,
                                   bool allowSoftware, uint32_t curveBatch, uint32_t batchKeys)
        : dictionary_(std::move(dictionary)), engine_(dictionary_, allowSoftware, curveBatch, batchKeys),
          curveBatch_(curveBatch), batchKeys_(batchKeys) {
        context_ = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    }
    ~VulkanResidentBackend() override {
        if (context_) secp256k1_context_destroy(context_);
    }
    std::string name() const override { return "Vulkan GPU resident"; }
    bool available() const override {
        return const_cast<VulkanResidentBackend*>(this)->ensureReady();
    }
    std::string note() const override { return error_; }
    BackendInfo info() const override {
        BackendInfo out;
        out.kind = "GPU-resident";
        out.title = engine_.deviceName().empty() ? "Vulkan compute" : engine_.deviceName();
        out.lines.push_back("Native Vulkan: OS CSPRNG + CPU base expansion + GPU full address and dictionary scan");
        out.lines.push_back("CPU verifies every reported private key/address/match; no CPU fallback");
        out.lines.push_back(engine_.deviceLocalHostVisible() ?
                            "Host-visible device-local memory" :
                            "Host-visible non-device-local memory; PCIe staging may be faster");
        out.lines.push_back("Curve batch: " + std::to_string(curveBatch_));
        out.lines.push_back("Submit batch: " + std::to_string(batchKeys_) + " keys");
        out.lines.push_back("Resident queue group: " + std::to_string(kResidentDispatches) +
                            " batches per fence");
        return out;
    }
    double benchmark(double seconds) override {
        if (!ensureReady()) return 0.0;
        const auto start = std::chrono::steady_clock::now();
        uint64_t total = 0;
        do {
            uint32_t count = batchKeys_;
            if (!scanChunks(count, kResidentDispatches, nullptr)) break;
            total += uint64_t(count) * kResidentDispatches;
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return elapsed > 0 ? double(total) / elapsed : 0.0;
    }
    VulkanProfileResult profile(double seconds) {
        VulkanProfileResult result;
        if (!ensureReady()) { result.error = error_; return result; }
        result.device = engine_.deviceName();
        result.curveBatch = curveBatch_;
        result.batchKeys = batchKeys_;
        result.deviceLocalHostVisible = engine_.deviceLocalHostVisible();
        result.timestampsSupported = engine_.timestampsSupported();
        // One unmeasured full-size batch primes JIT compilation, caches and
        // the fixed-base table before the bounded wall-clock measurement.
        if (!scanChunk(batchKeys_, nullptr)) { result.error = error_; return result; }
        engine_.beginProfile();
        const auto start = std::chrono::steady_clock::now();
        do {
            if (!scanChunks(batchKeys_, kResidentDispatches, nullptr)) break;
            result.keys += uint64_t(batchKeys_) * kResidentDispatches;
            result.dispatches += kResidentDispatches;
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds);
        result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        engine_.endProfile();
        result.error = error_;
        result.gpuSeconds = engine_.gpuSeconds();
        result.stageSeconds = engine_.stageSeconds();
        result.hostSeconds = engine_.hostSeconds();
        result.candidateRecords = engine_.candidateRecords();
        return result;
    }
    void run(const RunConfig& cfg, RunState& state, const ReportFn& report) override {
        if (!ensureReady()) {
            std::cerr << "Vulkan unavailable: " << error_ << "\n";
            state.stop.store(true);
            return;
        }
        while (!state.stop.load(std::memory_order_relaxed)) {
            const uint64_t done = state.checked.load(std::memory_order_relaxed);
            if (cfg.maxAttempts && done >= cfg.maxAttempts) break;
            const uint32_t count = cfg.maxAttempts ?
                static_cast<uint32_t>(std::min<uint64_t>(batchKeys_, cfg.maxAttempts - done)) : batchKeys_;
            const uint64_t remaining = cfg.maxAttempts ? cfg.maxAttempts - done :
                                        uint64_t(batchKeys_) * kResidentDispatches;
            const uint32_t dispatches = static_cast<uint32_t>(std::min<uint64_t>(
                kResidentDispatches, std::max<uint64_t>(1, remaining / count)));
            const uint32_t scanOffset = offsetBase_;
            std::vector<Candidate> candidates;
            if (!scanChunks(count, dispatches, &candidates)) {
                std::cerr << "Vulkan scan stopped: " << error_ << "\n";
                state.stop.store(true);
                return;
            }
            for (const Candidate& candidate : candidates) {
                SecretScalar scalar;
                unsigned char xy[64]{};
                if (!addOffset(base_.data(), scanOffset + candidate.gid, scalar.data()) ||
                    !publicXY(context_, scalar.data(), xy) ||
                    tronAddressFromPubXY(xy) != candidate.address) {
                    error_ = "Vulkan candidate key/address verification failed";
                    state.stop.store(true);
                    break;
                }
                const auto ids = dictionary_->matchIds(candidate.address);
                if (ids.empty() || candidate.count != std::min<size_t>(16, ids.size()) ||
                    candidate.flags != (ids.size() > 16 ? 1u : 0u) ||
                    !std::equal(ids.begin(), ids.begin() + candidate.count, candidate.ids.begin())) {
                    error_ = "Vulkan candidate dictionary verification failed";
                    state.stop.store(true);
                    break;
                }
                FoundKey key;
                key.address = candidate.address;
                key.privHex = hexUpper(scalar.data(), 32);
                for (uint32_t id : ids) {
                    if (id >= dictionary_->words.size()) {
                        error_ = "Vulkan dictionary returned invalid ID";
                        state.stop.store(true);
                        break;
                    }
                    key.words.push_back(dictionary_->words[id]);
                }
                if (state.stop.load()) break;
                state.found.fetch_add(1, std::memory_order_relaxed);
                report(key);
            }
            const uint64_t submitted = uint64_t(count) * dispatches;
            state.checked.fetch_add(submitted, std::memory_order_relaxed);
            state.gpuChecked.fetch_add(submitted, std::memory_order_relaxed);
            if (state.stop.load() && !error_.empty())
                std::cerr << "Vulkan verification stopped: " << error_ << "\n";
        }
    }
    bool selfTestRollover() {
        if (!ensureReady()) return false;
        // Exercise the final valid offset and the next-base transition with
        // two one-key chunks. Scanning an entire full-alphabet batch here
        // would make the test spend most of its time re-deriving thousands of
        // deterministic public keys on the CPU, without testing another
        // rollover invariant.
        base_.bytes.fill(0);
        base_.bytes[31] = 1;
        if (!publicXY(context_, base_.data(), basePub_.data())) {
            error_ = "Vulkan rollover test base point failed";
            return false;
        }
        SecretScalar originalBase;
        std::memcpy(originalBase.data(), base_.data(), originalBase.bytes.size());
        baseReady_ = true;
        offsetBase_ = kBaseWindowKeys - 1;
        RunConfig config;
        config.dictionary = dictionary_;
        config.maxAttempts = 1;
        config.seconds = 0;
        RunState state;
        bool validReports = true;
        run(config, state, [&](const FoundKey& key) {
            validReports &= key.address.size() == 34 && key.privHex.size() == 64 &&
                            !key.words.empty();
        });
        if (!validReports || !error_.empty() || state.stop.load() ||
            state.checked.load() != 1 || state.gpuChecked.load() != 1 ||
            state.found.load() != 1 || offsetBase_ != kBaseWindowKeys) {
            if (error_.empty()) error_ = "Vulkan base rollover or bounded production run failed";
            return false;
        }
        std::vector<Candidate> candidates;
        if (!scanChunk(1, &candidates) || !error_.empty() || candidates.size() != 1 ||
            offsetBase_ != 1 || base_.bytes == originalBase.bytes) {
            if (error_.empty()) error_ = "Vulkan CSPRNG base did not roll over after the final offset";
            return false;
        }
        unsigned char xy[64]{};
        if (!publicXY(context_, base_.data(), xy) ||
            tronAddressFromPubXY(xy) != candidates.front().address) {
            error_ = "Vulkan post-rollover base does not match the scanned address";
            return false;
        }
        return true;
    }
    bool selfTestDeviceLoss() {
        if (!ensureReady()) return false;
        std::cout << "Vulkan self-test: injecting a synthetic device-loss return; "
                     "no hardware fault is expected\n";
        engine_.injectDeviceLossOnceForTest();
        RunConfig config;
        config.dictionary = dictionary_;
        config.maxAttempts = 1;
        config.seconds = 0;
        RunState state;
        bool emitted = false;
        run(config, state, [&](const FoundKey&) { emitted = true; });
        if (!state.stop.load() || state.checked.load() != 0 ||
            state.gpuChecked.load() != 0 || state.found.load() != 0 || emitted ||
            error_.find("vkQueueSubmit failed") == std::string::npos) {
            error_ = "Vulkan injected device loss did not stop without wallet output";
            return false;
        }
        std::vector<Candidate> candidates;
        if (scanChunk(1, &candidates) ||
            error_ != "Vulkan device was abandoned after a GPU failure") {
            error_ = "Vulkan device-loss retry was not rejected";
            return false;
        }
        return true;
    }
private:
    bool ensureReady() {
        if (tried_) return ready_;
        tried_ = true;
        if (curveBatch_ != 1 && curveBatch_ != 4) {
            error_ = "Vulkan curve batch must be 1 or 4";
            return false;
        }
        if (batchKeys_ != 32768 && batchKeys_ != 65536 &&
            batchKeys_ != 131072 && batchKeys_ != 262144 &&
            batchKeys_ != 524288 && batchKeys_ != 1048576) {
            error_ = "Vulkan submit batch must be 32768, 65536, 131072, 262144, 524288 or 1048576";
            return false;
        }
        if (!context_) { error_ = "secp256k1 context creation failed"; return false; }
        if (!engine_.init(error_)) return false;
        SecretScalar test;
        test.bytes[31] = 1;
        if (!verifyScan(engine_, context_, *dictionary_, test.data(), 255, 65, error_)) return false;
        ready_ = true;
        return true;
    }
    bool prepareBase() {
        for (int attempt = 0; attempt < 128; ++attempt) {
            SecretScalar top;
            if (!randBytes(base_.data(), 32)) {
                error_ = "OS CSPRNG failed"; return false;
            }
            if (!addOffset(base_.data(), kBaseWindowKeys - 1, top.data()) ||
                !publicXY(context_, base_.data(), basePub_.data()) ||
                !secp256k1_ec_seckey_verify(context_, top.data())) continue;
            offsetBase_ = 0;
            baseReady_ = true;
            return true;
        }
        error_ = "OS CSPRNG did not produce a valid secp256k1 range";
        return false;
    }
    bool scanChunk(uint32_t count, std::vector<Candidate>* out) {
        return scanChunks(count, 1, out);
    }
    bool scanChunks(uint32_t count, uint32_t dispatches, std::vector<Candidate>* out) {
        const uint64_t total = uint64_t(count) * dispatches;
        if (!count || !dispatches || total > kBaseWindowKeys) {
            error_ = "invalid Vulkan resident dispatch group";
            return false;
        }
        if (!baseReady_ || offsetBase_ > kBaseWindowKeys - total)
            if (!prepareBase()) return false;
        const uint32_t scanOffset = offsetBase_;
        std::vector<Candidate> local;
        if (!engine_.scanMany(basePub_.data(), scanOffset, count, dispatches,
                              out ? &local : nullptr, error_))
            return false;
        offsetBase_ += static_cast<uint32_t>(total);
        if (out) *out = std::move(local);
        return true;
    }
    std::shared_ptr<const Dictionary> dictionary_;
    VulkanEngine engine_;
    uint32_t curveBatch_ = 1;
    uint32_t batchKeys_ = kDefaultBatchKeys;
    secp256k1_context* context_ = nullptr;
    SecretScalar base_;
    std::array<unsigned char, 64> basePub_{};
    uint32_t offsetBase_ = 0;
    bool tried_ = false, ready_ = false, baseReady_ = false;
    std::string error_;
};
}

std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary> dictionary, uint32_t curveBatch, uint32_t batchKeys) {
    // Software Vulkan is for reproducible tests only; production selection
    // must not silently replace the requested GPU with CPU llvmpipe.
    const char* allow = std::getenv("TRON_VULKAN_ALLOW_SOFTWARE");
    return std::make_unique<VulkanResidentBackend>(std::move(dictionary),
                                                   allow && std::strcmp(allow, "1") == 0,
                                                   curveBatch, batchKeys);
}

VulkanProfileResult profileVulkanResident(std::shared_ptr<const Dictionary> dictionary,
                                          double seconds, uint32_t curveBatch, uint32_t batchKeys) {
    const char* allow = std::getenv("TRON_VULKAN_ALLOW_SOFTWARE");
    VulkanResidentBackend backend(std::move(dictionary), allow && std::strcmp(allow, "1") == 0,
                                  curveBatch, batchKeys);
    return backend.profile(seconds);
}

int vulkanResidentSelfTest() {
    std::string error;
    auto dictionary = vulkanFullAlphabetTestDictionary();
    if (!dictionary) { std::cerr << "Vulkan test dictionary: " << error << "\n"; return 1; }
    auto* context = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!context) return 1;
    SecretScalar base;
    base.bytes[31] = 1;
    bool passed = true;
    for (uint32_t batch : {1u, 4u}) {
        // Different SPIR-V modules are selected at pipeline creation. Do
        // not switch just the push constant on an already-created pipeline.
        // The full-alphabet test deliberately reports every address. Keep its
        // bounded CPU verification independent from the production throughput
        // batch; increasing the latter must not turn `test-vulkan` into a
        // multi-minute test or make the launcher kill it.
        VulkanEngine engine(dictionary, true, batch, kSelfTestBatchKeys);
        if (!engine.init(error)) {
            secp256k1_context_destroy(context);
            std::cerr << "Vulkan resident setup batch " << batch << ": " << error << "\n";
            return error.find("no Vulkan") != std::string::npos ? 77 : 1;
        }
        std::vector<std::pair<uint32_t, uint32_t>> cases;
        if (batch == 1)
            cases = {{0, 1}, {255, 65}, {65535, 65},
                     {kBaseWindowKeys - 65, 65}, {0, 65}};
        else
            cases = {{0, 1}, {0, 3}, {0, 4}, {0, 5}, {255, 5},
                     {65535, 5}, {kBaseWindowKeys - 5, 5}, {0, 65}};
        for (const auto& test : cases) {
            if (!verifyScan(engine, context, *dictionary, base.data(), test.first, test.second,
                            error)) {
                error = "curve batch " + std::to_string(batch) + ", offset " +
                        std::to_string(test.first) + ", count " +
                        std::to_string(test.second) + ": " + error;
                passed = false;
                break;
            }
        }
        if (passed && !verifyGroupedScan(engine, context, *dictionary, base.data(),
                                         255, 17, kResidentDispatches, error)) {
            error = "curve batch " + std::to_string(batch) +
                    ", grouped resident submit: " + error;
            passed = false;
        }
        if (!passed) break;
    }
    secp256k1_context_destroy(context);
    if (!passed) { std::cerr << "Vulkan resident test: " << error << "\n"; return 1; }
    for (uint32_t batch : {1u, 4u}) {
        auto backend = std::make_unique<VulkanResidentBackend>(dictionary, true, batch,
                                                              kSelfTestBatchKeys);
        if (!backend || !backend->available()) {
            std::cerr << "Vulkan production backend batch " << batch << ": "
                      << (backend ? backend->note() : "not built") << "\n";
            return 1;
        }
        RunConfig config;
        config.dictionary = dictionary;
        config.maxAttempts = 65;
        config.seconds = 0;
        RunState state;
        bool validReports = true;
        backend->run(config, state, [&](const FoundKey& key) {
            validReports &= key.address.size() == 34 && key.privHex.size() == 64 &&
                            !key.words.empty();
        });
        if (!validReports || !backend->note().empty() ||
            state.checked.load() != 65 || state.gpuChecked.load() != 65 ||
            state.found.load() != 65) {
            std::cerr << "Vulkan production backend batch " << batch
                      << " bounded no-wallet run failed: " << backend->note() << "\n";
            return 1;
        }
        if (!backend->selfTestRollover()) {
            std::cerr << "Vulkan production backend batch " << batch
                      << " rollover failed: " << backend->note() << "\n";
            return 1;
        }
        if (batch == 4 && !backend->selfTestDeviceLoss()) {
            std::cerr << "Vulkan production backend device-loss test failed: "
                      << backend->note() << "\n";
            return 1;
        }
    }
    std::cout << "Vulkan resident repeated dispatch + CPU verification PASS (no wallets)\n";
    return 0;
}
