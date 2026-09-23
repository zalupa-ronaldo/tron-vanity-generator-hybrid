#include "vulkan_backend.h"

#include "crypto.h"
#include "rng.h"

#include "vulkan_curve_spv.h"
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
constexpr uint32_t kBatchKeys = 1u << 15;
constexpr uint32_t kBaseWindowKeys = 1u << 22;
constexpr uint32_t kGroupSize = 64;
constexpr uint32_t kBindings = 10;
constexpr uint32_t kRingWords = 20;
constexpr uint32_t kStageCount = 5;
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
    uint32_t curveMode, offsetBase, ringMode, ringCapacity;
};
static_assert(sizeof(PushConstants) == 36, "Vulkan push-constant ABI changed");

class VulkanEngine {
public:
    explicit VulkanEngine(std::shared_ptr<const Dictionary> dictionary, bool allowSoftware)
        : dictionary_(std::move(dictionary)), allowSoftware_(allowSoftware) {}
    VulkanEngine(const VulkanEngine&) = delete;
    VulkanEngine& operator=(const VulkanEngine&) = delete;
    ~VulkanEngine() {
        // On a driver timeout, destruction of in-flight Vulkan objects is
        // invalid and vkDeviceWaitIdle may hang. The process exits after the
        // backend reports failure; let the OS reclaim abandoned handles.
        if (abandoned_) return;
        if (device_) vkDeviceWaitIdle(device_);
        if (device_ && mapped_) vkUnmapMemory(device_, memory_);
        if (device_ && fence_) vkDestroyFence(device_, fence_, nullptr);
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
    const std::string& deviceName() const { return deviceName_; }

private:
    static VkDeviceSize aligned(VkDeviceSize size, VkDeviceSize alignment) {
        return (size + alignment - 1) / alignment * alignment;
    }
    bool createTable(std::vector<uint32_t>& table, std::string& error);
    std::string readAddress(uint32_t gid) const;

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
    std::array<VkDeviceSize, kBindings> offsets_{};
    std::array<VkDeviceSize, kBindings> sizes_{};
    PushConstants constants_{};
    bool abandoned_ = false;
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
        error = "no hardware Vulkan compute device with shaderInt64 and 10 storage buffers";
        return false;
    }
    deviceName_ = selected.deviceName;
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
    for (uint32_t i = 0; i < kBindings; ++i) {
        sizes_[i] = i == 4 ? automaton.size() * sizeof(uint32_t) :
                    i == 6 ? 64 : i == 7 ? table.size() * sizeof(uint32_t) :
                    i == 8 ? 2 * sizeof(uint32_t) :
                    i == 5 ? 18 * sizeof(uint32_t) :
                    VkDeviceSize(kBatchKeys) * kWordsPerKey[i] * sizeof(uint32_t);
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
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer_), "vkCreateBuffer")) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProps{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &memoryProps);
    uint32_t memoryType = memoryProps.memoryTypeCount;
    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memoryProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryType = i;
            break;
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
    constants_.curveMode = 1;
    constants_.ringMode = 1;
    constants_.ringCapacity = kBatchKeys;
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
        kVulkanCurveSpv, kVulkanKeccakSpv, kVulkanChecksumSpv,
        kVulkanBase58Spv, kVulkanMatchSpv
    };
    const std::array<size_t, kStageCount> codeSizes = {
        sizeof(kVulkanCurveSpv), sizeof(kVulkanKeccakSpv), sizeof(kVulkanChecksumSpv),
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
    return check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence");
}

std::string VulkanEngine::readAddress(uint32_t gid) const {
    const auto* words = reinterpret_cast<const uint32_t*>(
        static_cast<const unsigned char*>(mapped_) + offsets_[3]) + size_t(gid) * 9;
    std::string address(34, '\0');
    for (uint32_t i = 0; i < 34; ++i)
        address[i] = static_cast<char>((words[i / 4] >> (8 * (i % 4))) & 255u);
    return address;
}

bool VulkanEngine::scan(const unsigned char basePub[64], uint32_t offsetBase, uint32_t count,
                        std::vector<Candidate>& candidates,
                        std::vector<std::string>* allAddresses, std::string& error) {
    auto check = [&](VkResult result, const char* call) {
        if (result == VK_SUCCESS) return true;
        error = std::string(call) + " failed: " + std::to_string(result);
        return false;
    };
    if (abandoned_) { error = "Vulkan device was abandoned after a timeout"; return false; }
    if (!mapped_ || count == 0 || count > kBatchKeys ||
        offsetBase > kBaseWindowKeys - count) {
        error = "invalid Vulkan scan range"; return false;
    }
    auto* bytes = static_cast<unsigned char*>(mapped_);
    std::memcpy(bytes + offsets_[6], basePub, 64);
    std::memset(bytes + offsets_[8], 0, sizes_[8]);
    constants_.count = count;
    constants_.offsetBase = offsetBase;
    if (!check(vkResetCommandPool(device_, commandPool_, 0), "vkResetCommandPool") ||
        !check(vkResetFences(device_, 1, &fence_), "vkResetFences")) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer")) return false;
    vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                            0, 1, &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(command_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(constants_), &constants_);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    for (uint32_t stage = 0; stage < kStageCount; ++stage) {
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[stage]);
        vkCmdDispatch(command_, (count + kGroupSize - 1) / kGroupSize, 1, 1);
        if (stage + 1 < kStageCount)
            vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (!check(vkEndCommandBuffer(command_), "vkEndCommandBuffer")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    if (!check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit")) return false;
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
    const auto* meta = reinterpret_cast<const uint32_t*>(bytes + offsets_[8]);
    if (meta[0] > kBatchKeys || (meta[1] & 1u)) {
        error = "Vulkan result ring overflow; stopping to avoid dropped matches"; return false;
    }
    candidates.clear();
    candidates.reserve(meta[0]);
    const auto* records = reinterpret_cast<const uint32_t*>(bytes + offsets_[9]);
    std::unordered_set<uint32_t> seen;
    for (uint32_t slot = 0; slot < meta[0]; ++slot) {
        const uint32_t* record = records + size_t(slot) * kRingWords;
        if (record[0] >= count || !seen.insert(record[0]).second ||
            record[1] > 16 || (record[1] == 0 && record[2] == 0) ||
            record[3] != 0 || (record[2] & ~1u)) {
            error = "Vulkan result record invalid"; return false;
        }
        Candidate candidate;
        candidate.gid = record[0];
        candidate.count = record[1];
        candidate.flags = record[2];
        std::copy_n(record + 4, 16, candidate.ids.begin());
        candidate.address = readAddress(candidate.gid);
        candidates.push_back(std::move(candidate));
    }
    if (allAddresses) {
        allAddresses->clear();
        allAddresses->reserve(count);
        for (uint32_t gid = 0; gid < count; ++gid) allAddresses->push_back(readAddress(gid));
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

class VulkanResidentBackend final : public Backend {
public:
    explicit VulkanResidentBackend(std::shared_ptr<const Dictionary> dictionary, bool allowSoftware)
        : dictionary_(std::move(dictionary)), engine_(dictionary_, allowSoftware) {
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
        return out;
    }
    double benchmark(double seconds) override {
        if (!ensureReady()) return 0.0;
        const auto start = std::chrono::steady_clock::now();
        uint64_t total = 0;
        do {
            uint32_t count = kBatchKeys;
            if (!scanChunk(count, nullptr)) break;
            total += count;
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return elapsed > 0 ? double(total) / elapsed : 0.0;
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
                static_cast<uint32_t>(std::min<uint64_t>(kBatchKeys, cfg.maxAttempts - done)) : kBatchKeys;
            std::vector<Candidate> candidates;
            if (!scanChunk(count, &candidates)) {
                std::cerr << "Vulkan scan stopped: " << error_ << "\n";
                state.stop.store(true);
                return;
            }
            for (const Candidate& candidate : candidates) {
                SecretScalar scalar;
                unsigned char xy[64]{};
                if (!addOffset(base_.data(), offsetBase_ - count + candidate.gid, scalar.data()) ||
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
            state.checked.fetch_add(count, std::memory_order_relaxed);
            state.gpuChecked.fetch_add(count, std::memory_order_relaxed);
            if (state.stop.load() && !error_.empty())
                std::cerr << "Vulkan verification stopped: " << error_ << "\n";
        }
    }
private:
    bool ensureReady() {
        if (tried_) return ready_;
        tried_ = true;
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
        if (!baseReady_ || offsetBase_ > kBaseWindowKeys - count)
            if (!prepareBase()) return false;
        std::vector<Candidate> local;
        if (!engine_.scan(basePub_.data(), offsetBase_, count, local, nullptr, error_)) return false;
        offsetBase_ += count;
        if (out) *out = std::move(local);
        return true;
    }
    std::shared_ptr<const Dictionary> dictionary_;
    VulkanEngine engine_;
    secp256k1_context* context_ = nullptr;
    SecretScalar base_;
    std::array<unsigned char, 64> basePub_{};
    uint32_t offsetBase_ = 0;
    bool tried_ = false, ready_ = false, baseReady_ = false;
    std::string error_;
};
}

std::unique_ptr<Backend> makeVulkanResidentBackend(
    std::shared_ptr<const Dictionary> dictionary) {
    // Software Vulkan is for reproducible tests only; production selection
    // must not silently replace the requested GPU with CPU llvmpipe.
    const char* allow = std::getenv("TRON_VULKAN_ALLOW_SOFTWARE");
    return std::make_unique<VulkanResidentBackend>(std::move(dictionary),
                                                   allow && std::strcmp(allow, "1") == 0);
}

int vulkanResidentSelfTest() {
    std::string error;
    auto dictionary = vulkanTestDictionary();
    if (!dictionary) { std::cerr << "Vulkan test dictionary: " << error << "\n"; return 1; }
    auto* context = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!context) return 1;
    VulkanEngine engine(dictionary, true);
    if (!engine.init(error)) {
        secp256k1_context_destroy(context);
        std::cerr << "Vulkan resident setup: " << error << "\n";
        return error.find("no Vulkan") != std::string::npos ? 77 : 1;
    }
    SecretScalar base;
    base.bytes[31] = 1;
    bool passed = true;
    for (const auto& test : {std::pair<uint32_t,uint32_t>{0, 1}, {255, 65},
                             {65535, 65}, {kBaseWindowKeys - 65, 65}, {0, 65}}) {
        if (!verifyScan(engine, context, *dictionary, base.data(), test.first, test.second, error)) {
            passed = false;
            break;
        }
    }
    secp256k1_context_destroy(context);
    if (!passed) { std::cerr << "Vulkan resident test: " << error << "\n"; return 1; }
    auto backend = std::make_unique<VulkanResidentBackend>(dictionary, true);
    if (!backend || !backend->available()) {
        std::cerr << "Vulkan production backend test: "
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
        state.found.load() == 0) {
        std::cerr << "Vulkan production backend bounded no-wallet run failed: "
                  << backend->note() << "\n";
        return 1;
    }
    std::cout << "Vulkan resident repeated dispatch + CPU verification PASS (no wallets)\n";
    return 0;
}
