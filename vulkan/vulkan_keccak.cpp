#include "vulkan_probe.h"
#include "vulkan_keccak_spv.h"
#include "vulkan_checksum_spv.h"
#include "vulkan_base58_spv.h"
#include "vulkan_match_spv.h"
#include "vulkan_curve_spv.h"
#include "crypto.h"
#include "dictionary.h"

#include <vulkan/vulkan.h>
#include <secp256k1.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
// Include a partial fifth workgroup in the maximum-size test. All buffers
// have room for kItems, while each dispatch may process a smaller prefix.
constexpr uint32_t kItems = 257;
constexpr size_t kPublicWords = 16;
constexpr size_t kPayloadWords = 6;
constexpr size_t kFullWords = 7;
constexpr size_t kAddressWords = 9;
constexpr size_t kMatchWords = 18;
using PublicBatch = std::array<uint32_t, kItems * kPublicWords>;
using PayloadBatch = std::array<uint32_t, kItems * kPayloadWords>;
using FullBatch = std::array<uint32_t, kItems * kFullWords>;
using AddressBatch = std::array<uint32_t, kItems * kAddressWords>;
using MatchBatch = std::array<uint32_t, kItems * kMatchWords>;

// These are deterministic, unfunded *test* keys. Never use them as wallets.
bool makeTestVectors(PublicBatch& basePubs, PublicBatch& pubs, PayloadBatch& payloads,
                     FullBatch& fulls, AddressBatch& addresses,
                     MatchBatch& matches, const Dictionary& dictionary,
                     std::string& error) {
    using SecpContext = std::unique_ptr<secp256k1_context, decltype(&secp256k1_context_destroy)>;
    SecpContext ctx(secp256k1_context_create(SECP256K1_CONTEXT_NONE), secp256k1_context_destroy);
    if (!ctx) { error = "secp256k1 context creation failed"; return false; }
    for (uint32_t i = 0; i < kItems; ++i) {
        unsigned char scalar[32]{};
        scalar[30] = static_cast<unsigned char>((i + 1) >> 8);
        scalar[31] = static_cast<unsigned char>(i + 1);
        secp256k1_pubkey point;
        if (!secp256k1_ec_pubkey_create(ctx.get(), &point, scalar)) {
            error = "test public key generation failed"; return false;
        }
        unsigned char serialized[65];
        size_t length = sizeof(serialized);
        if (!secp256k1_ec_pubkey_serialize(ctx.get(), serialized, &length, &point,
                                           SECP256K1_EC_UNCOMPRESSED) || length != 65) {
            error = "test public key serialization failed"; return false;
        }
        // The Vulkan curve stage computes (i+1)G + G. Address stages must
        // therefore process (i+2)G, independently derived by libsecp256k1.
        const unsigned char* xy = serialized + 1;
        for (size_t word = 0; word < kPublicWords; ++word) {
            const size_t j = word * 4;
            basePubs[i * kPublicWords + word] = uint32_t(xy[j]) |
                (uint32_t(xy[j + 1]) << 8) | (uint32_t(xy[j + 2]) << 16) |
                (uint32_t(xy[j + 3]) << 24);
        }
        scalar[30] = static_cast<unsigned char>((i + 2) >> 8);
        scalar[31] = static_cast<unsigned char>(i + 2);
        if (!secp256k1_ec_pubkey_create(ctx.get(), &point, scalar)) {
            error = "next test public key generation failed"; return false;
        }
        length = sizeof(serialized);
        if (!secp256k1_ec_pubkey_serialize(ctx.get(), serialized, &length, &point,
                                           SECP256K1_EC_UNCOMPRESSED) || length != 65) {
            error = "next test public key serialization failed"; return false;
        }
        for (size_t word = 0; word < kPublicWords; ++word) {
            const size_t j = word * 4;
            pubs[i * kPublicWords + word] = uint32_t(xy[j]) |
                (uint32_t(xy[j + 1]) << 8) | (uint32_t(xy[j + 2]) << 16) |
                (uint32_t(xy[j + 3]) << 24);
        }
        unsigned char full[28]{};
        tronFullFromPubXY(xy, full);
        for (size_t word = 0; word < kPayloadWords; ++word) {
            const size_t j = word * 4;
            payloads[i * kPayloadWords + word] = uint32_t(full[j]) |
                (uint32_t(full[j + 1]) << 8) | (uint32_t(full[j + 2]) << 16) |
                (uint32_t(full[j + 3]) << 24);
        }
        // The last payload word must have three zero padding bytes, not
        // checksum bytes from the full 25-byte address buffer.
        payloads[i * kPayloadWords + 5] &= 0xffu;
        for (size_t word = 0; word < kFullWords; ++word) {
            const size_t j = word * 4;
            fulls[i * kFullWords + word] = uint32_t(full[j]) |
                (uint32_t(full[j + 1]) << 8) | (uint32_t(full[j + 2]) << 16) |
                (uint32_t(full[j + 3]) << 24);
        }
        const std::string address = tronAddressFromPubXY(xy);
        if (address.size() != 34 || address[0] != 'T') {
            error = "CPU TRON test address is not 34 chars starting with T"; return false;
        }
        unsigned char padded[36]{};
        std::memcpy(padded, address.data(), address.size());
        for (size_t word = 0; word < kAddressWords; ++word) {
            const size_t j = word * 4;
            addresses[i * kAddressWords + word] = uint32_t(padded[j]) |
                (uint32_t(padded[j + 1]) << 8) | (uint32_t(padded[j + 2]) << 16) |
                (uint32_t(padded[j + 3]) << 24);
        }
        const auto ids = dictionary.matchIds(address);
        matches[i * kMatchWords] = static_cast<uint32_t>(std::min<size_t>(16, ids.size()));
        matches[i * kMatchWords + 1] = ids.size() > 16 ? 1u : 0u;
        for (size_t j = 0; j < std::min<size_t>(16, ids.size()); ++j)
            matches[i * kMatchWords + 2 + j] = ids[j];
    }
    return true;
}

struct State {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkShaderModule checksumShader = VK_NULL_HANDLE;
    VkShaderModule base58Shader = VK_NULL_HANDLE;
    VkShaderModule matchShader = VK_NULL_HANDLE;
    VkShaderModule curveShader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline checksumPipeline = VK_NULL_HANDLE;
    VkPipeline base58Pipeline = VK_NULL_HANDLE;
    VkPipeline matchPipeline = VK_NULL_HANDLE;
    VkPipeline curvePipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    ~State() {
        if (device) vkDeviceWaitIdle(device);
        if (device && fence) vkDestroyFence(device, fence, nullptr);
        if (device && commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device && pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (device && checksumPipeline) vkDestroyPipeline(device, checksumPipeline, nullptr);
        if (device && base58Pipeline) vkDestroyPipeline(device, base58Pipeline, nullptr);
        if (device && matchPipeline) vkDestroyPipeline(device, matchPipeline, nullptr);
        if (device && curvePipeline) vkDestroyPipeline(device, curvePipeline, nullptr);
        if (device && shader) vkDestroyShaderModule(device, shader, nullptr);
        if (device && checksumShader) vkDestroyShaderModule(device, checksumShader, nullptr);
        if (device && base58Shader) vkDestroyShaderModule(device, base58Shader, nullptr);
        if (device && matchShader) vkDestroyShaderModule(device, matchShader, nullptr);
        if (device && curveShader) vkDestroyShaderModule(device, curveShader, nullptr);
        if (device && pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (device && descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (device && descriptorLayout) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        if (device && buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (device && memory) vkFreeMemory(device, memory, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

bool runKeccak(uint32_t activeItems, std::string& error) {
    if (!activeItems || activeItems > kItems) {
        error = "invalid Vulkan test batch size";
        return false;
    }
    PublicBatch basePubs{};
    PublicBatch pubs{};
    PayloadBatch expectedPayload{};
    FullBatch expectedFull{};
    AddressBatch expectedAddresses{};
    MatchBatch expectedMatches{};
    auto dictionary = Dictionary::load(VULKAN_TEST_WORDS_PATH, true, &error);
    if (!dictionary) return false;
    if (!makeTestVectors(basePubs, pubs, expectedPayload, expectedFull, expectedAddresses,
                         expectedMatches, *dictionary, error)) return false;
    size_t overflows = 0, inRange = 0;
    for (uint32_t i = 0; i < activeItems; ++i) {
        if (expectedMatches[i * kMatchWords + 1]) ++overflows;
        else ++inRange;
    }
    if (activeItems == kItems && (!overflows || !inRange)) {
        error = "dictionary vectors did not cover both bounded and overflow matches";
        return false;
    }
    std::vector<uint32_t> automaton;
    struct PushConstants {
        uint32_t count, dfaOffset, outStartOffset, outLenOffset, outIdsOffset;
    } constants{};
    constants.count = activeItems;
    constants.dfaOffset = 0;
    automaton.insert(automaton.end(), dictionary->dfa.begin(), dictionary->dfa.end());
    constants.outStartOffset = static_cast<uint32_t>(automaton.size());
    automaton.insert(automaton.end(), dictionary->outStart.begin(), dictionary->outStart.end());
    constants.outLenOffset = static_cast<uint32_t>(automaton.size());
    automaton.insert(automaton.end(), dictionary->outLen.begin(), dictionary->outLen.end());
    constants.outIdsOffset = static_cast<uint32_t>(automaton.size());
    automaton.insert(automaton.end(), dictionary->outIds.begin(), dictionary->outIds.end());
    State state;
    auto check = [&](VkResult result, const char* call) {
        if (result == VK_SUCCESS) return true;
        error = std::string(call) + " failed: " + std::to_string(result);
        return false;
    };
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "TRON vanity Vulkan Keccak stage test";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;
    if (!check(vkCreateInstance(&instanceInfo, nullptr, &state.instance), "vkCreateInstance")) return false;
    uint32_t deviceCount = 0;
    if (!check(vkEnumeratePhysicalDevices(state.instance, &deviceCount, nullptr),
               "vkEnumeratePhysicalDevices") || !deviceCount) {
        if (error.empty()) error = "no Vulkan physical devices";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (!check(vkEnumeratePhysicalDevices(state.instance, &deviceCount, devices.data()),
               "vkEnumeratePhysicalDevices")) return false;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceFeatures features{};
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceFeatures(device, &features);
        vkGetPhysicalDeviceProperties(device, &properties);
        if (!features.shaderInt64 || properties.limits.maxComputeWorkGroupInvocations < 64 ||
            properties.limits.maxComputeWorkGroupSize[0] < 64) continue;
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());
        for (uint32_t i = 0; i < familyCount; ++i) {
            if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                state.physical = device;
                state.queueFamily = i;
                break;
            }
        }
        if (state.physical) break;
    }
    if (!state.physical) { error = "no Vulkan compute device with shaderInt64"; return false; }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(state.physical, &properties);
    std::cout << "Vulkan Keccak stage device: " << properties.deviceName << "\n";
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = state.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    if (!check(vkCreateDevice(state.physical, &deviceInfo, nullptr, &state.device), "vkCreateDevice")) return false;
    vkGetDeviceQueue(state.device, state.queueFamily, 0, &state.queue);

    constexpr VkDeviceSize inBytes = sizeof(PublicBatch);
    constexpr VkDeviceSize outBytes = sizeof(PayloadBatch);
    constexpr VkDeviceSize fullBytes = sizeof(FullBatch);
    constexpr VkDeviceSize addressBytes = sizeof(AddressBatch);
    constexpr VkDeviceSize matchBytes = sizeof(MatchBatch);
    const VkDeviceSize automatonBytes = automaton.size() * sizeof(uint32_t);
    const VkDeviceSize alignment = properties.limits.minStorageBufferOffsetAlignment;
    const VkDeviceSize outputOffset = (inBytes + alignment - 1) / alignment * alignment;
    const VkDeviceSize fullOffset = (outputOffset + outBytes + alignment - 1) / alignment * alignment;
    const VkDeviceSize addressOffset = (fullOffset + fullBytes + alignment - 1) / alignment * alignment;
    const VkDeviceSize automatonOffset = (addressOffset + addressBytes + alignment - 1) / alignment * alignment;
    const VkDeviceSize matchOffset = (automatonOffset + automatonBytes + alignment - 1) / alignment * alignment;
    const VkDeviceSize baseOffset = (matchOffset + matchBytes + alignment - 1) / alignment * alignment;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = baseOffset + inBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(state.device, &bufferInfo, nullptr, &state.buffer), "vkCreateBuffer")) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(state.device, state.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProps{};
    vkGetPhysicalDeviceMemoryProperties(state.physical, &memoryProps);
    uint32_t memoryType = memoryProps.memoryTypeCount;
    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memoryProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryType = i; break;
        }
    }
    if (memoryType == memoryProps.memoryTypeCount) {
        error = "no coherent host-visible Vulkan memory"; return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    if (!check(vkAllocateMemory(state.device, &allocation, nullptr, &state.memory), "vkAllocateMemory") ||
        !check(vkBindBufferMemory(state.device, state.buffer, state.memory, 0), "vkBindBufferMemory")) return false;
    void* mapped = nullptr;
    if (!check(vkMapMemory(state.device, state.memory, 0, bufferInfo.size, 0, &mapped),
               "vkMapMemory(input)")) return false;
    constexpr unsigned char kCanary = 0xa5;
    std::memset(mapped, kCanary, inBytes);
    // A non-zero canary catches accidental writes by inactive lanes in the
    // final workgroup, which a comparison of only active records would miss.
    std::memset(static_cast<unsigned char*>(mapped) + outputOffset, kCanary, outBytes);
    std::memset(static_cast<unsigned char*>(mapped) + fullOffset, kCanary, fullBytes);
    std::memset(static_cast<unsigned char*>(mapped) + addressOffset, kCanary, addressBytes);
    std::memcpy(static_cast<unsigned char*>(mapped) + automatonOffset,
                automaton.data(), automatonBytes);
    std::memset(static_cast<unsigned char*>(mapped) + matchOffset, kCanary, matchBytes);
    std::memcpy(static_cast<unsigned char*>(mapped) + baseOffset, basePubs.data(), inBytes);
    vkUnmapMemory(state.device, state.memory);

    VkDescriptorSetLayoutBinding bindings[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings;
    if (!check(vkCreateDescriptorSetLayout(state.device, &layoutInfo, nullptr, &state.descriptorLayout),
               "vkCreateDescriptorSetLayout")) return false;
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (!check(vkCreateDescriptorPool(state.device, &poolInfo, nullptr, &state.descriptorPool),
               "vkCreateDescriptorPool")) return false;
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = state.descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &state.descriptorLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!check(vkAllocateDescriptorSets(state.device, &setInfo, &set), "vkAllocateDescriptorSets")) return false;
    VkDescriptorBufferInfo ranges[7] = {
        {state.buffer, 0, inBytes}, {state.buffer, outputOffset, outBytes},
        {state.buffer, fullOffset, fullBytes}, {state.buffer, addressOffset, addressBytes},
        {state.buffer, automatonOffset, automatonBytes}, {state.buffer, matchOffset, matchBytes},
        {state.buffer, baseOffset, inBytes}
    };
    VkWriteDescriptorSet writes[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &ranges[i];
    }
    vkUpdateDescriptorSets(state.device, 7, writes, 0, nullptr);

    VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = sizeof(kVulkanKeccakSpv);
    shaderInfo.pCode = kVulkanKeccakSpv;
    if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.shader),
               "vkCreateShaderModule")) return false;
    shaderInfo.codeSize = sizeof(kVulkanChecksumSpv);
    shaderInfo.pCode = kVulkanChecksumSpv;
    if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.checksumShader),
               "vkCreateShaderModule(checksum)")) return false;
    shaderInfo.codeSize = sizeof(kVulkanBase58Spv);
    shaderInfo.pCode = kVulkanBase58Spv;
    if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.base58Shader),
               "vkCreateShaderModule(base58)")) return false;
    shaderInfo.codeSize = sizeof(kVulkanMatchSpv);
    shaderInfo.pCode = kVulkanMatchSpv;
    if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.matchShader),
               "vkCreateShaderModule(match)")) return false;
    shaderInfo.codeSize = sizeof(kVulkanCurveSpv);
    shaderInfo.pCode = kVulkanCurveSpv;
    if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.curveShader),
               "vkCreateShaderModule(curve)")) return false;
    VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &state.descriptorLayout;
    pipelineLayout.pushConstantRangeCount = 1;
    pipelineLayout.pPushConstantRanges = &pushRange;
    if (!check(vkCreatePipelineLayout(state.device, &pipelineLayout, nullptr, &state.pipelineLayout),
               "vkCreatePipelineLayout")) return false;
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = state.shader;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = state.pipelineLayout;
    if (!check(vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                        nullptr, &state.pipeline), "vkCreateComputePipelines")) return false;
    pipelineInfo.stage.module = state.checksumShader;
    if (!check(vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                        nullptr, &state.checksumPipeline),
               "vkCreateComputePipelines(checksum)")) return false;
    pipelineInfo.stage.module = state.base58Shader;
    if (!check(vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                        nullptr, &state.base58Pipeline),
               "vkCreateComputePipelines(base58)")) return false;
    pipelineInfo.stage.module = state.matchShader;
    if (!check(vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                        nullptr, &state.matchPipeline),
               "vkCreateComputePipelines(match)")) return false;
    pipelineInfo.stage.module = state.curveShader;
    if (!check(vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                        nullptr, &state.curvePipeline),
               "vkCreateComputePipelines(curve)")) return false;
    VkCommandPoolCreateInfo commandPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPoolInfo.queueFamilyIndex = state.queueFamily;
    if (!check(vkCreateCommandPool(state.device, &commandPoolInfo, nullptr, &state.commandPool),
               "vkCreateCommandPool")) return false;
    VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandInfo.commandPool = state.commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (!check(vkAllocateCommandBuffers(state.device, &commandInfo, &command),
               "vkAllocateCommandBuffers")) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer")) return false;
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.curvePipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout,
                            0, 1, &set, 0, nullptr);
    vkCmdPushConstants(command, state.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(constants), &constants);
    vkCmdDispatch(command, (activeItems + 63u) / 64u, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    vkCmdDispatch(command, (activeItems + 63u) / 64u, 1, 1);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.checksumPipeline);
    vkCmdDispatch(command, (activeItems + 63u) / 64u, 1, 1);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.base58Pipeline);
    vkCmdDispatch(command, (activeItems + 63u) / 64u, 1, 1);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.matchPipeline);
    vkCmdDispatch(command, (activeItems + 63u) / 64u, 1, 1);
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (!check(vkEndCommandBuffer(command), "vkEndCommandBuffer")) return false;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!check(vkCreateFence(state.device, &fenceInfo, nullptr, &state.fence), "vkCreateFence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (!check(vkQueueSubmit(state.queue, 1, &submit, state.fence), "vkQueueSubmit") ||
        !check(vkWaitForFences(state.device, 1, &state.fence, VK_TRUE, 10000000000ull),
               "vkWaitForFences")) return false;
    if (!check(vkMapMemory(state.device, state.memory, 0, bufferInfo.size, 0, &mapped),
               "vkMapMemory(output)")) return false;
    auto validOutput = [&](VkDeviceSize offset, const void* expected, size_t stride) {
        const auto* actual = static_cast<const unsigned char*>(mapped) + offset;
        const size_t activeBytes = size_t(activeItems) * stride;
        if (std::memcmp(actual, expected, activeBytes) != 0) return false;
        for (size_t i = activeBytes; i < size_t(kItems) * stride; ++i)
            if (actual[i] != kCanary) return false;
        return true;
    };
    bool validPublic = validOutput(0, pubs.data(), kPublicWords * 4);
    bool validPayload = validOutput(outputOffset, expectedPayload.data(), kPayloadWords * 4);
    bool validFull = validOutput(fullOffset, expectedFull.data(), kFullWords * 4);
    bool validAddresses = validOutput(addressOffset, expectedAddresses.data(), kAddressWords * 4);
    bool validMatches = validOutput(matchOffset, expectedMatches.data(), kMatchWords * 4);
    vkUnmapMemory(state.device, state.memory);
    if (!validPublic) { error = "Vulkan secp256k1 public point differs from libsecp256k1"; return false; }
    if (!validPayload) { error = "Vulkan Keccak payload differs from CPU reference"; return false; }
    if (!validFull) { error = "Vulkan SHA-256d checksum differs from CPU reference"; return false; }
    if (!validAddresses) { error = "Vulkan Base58Check address differs from CPU reference"; return false; }
    if (!validMatches) { error = "Vulkan dictionary matches/overflow differ from CPU reference"; return false; }
    return true;
}
}

int vulkanKeccakSelfTest() {
    std::string error;
    for (uint32_t count : {1u, 63u, 64u, 65u, kItems}) {
        if (!runKeccak(count, error)) {
            std::cerr << "Vulkan address pipeline test failed at count " << count
                      << ": " << error << "\n";
            return error == "no Vulkan compute device with shaderInt64" ? 77 : 1;
        }
    }
    std::cout << "Vulkan curve + address + dictionary stages PASS "
                 "(1/63/64/65/257 full TRON addresses; no wallets)\n";
    return 0;
}
