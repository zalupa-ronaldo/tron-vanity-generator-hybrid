#include "vulkan_probe.h"
#include "vulkan_keccak_spv.h"
#include "keccak.h"

#include <vulkan/vulkan.h>
#include <secp256k1.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kItems = 256;
constexpr size_t kPublicWords = 16;
constexpr size_t kPayloadWords = 6;
using PublicBatch = std::array<uint32_t, kItems * kPublicWords>;
using PayloadBatch = std::array<uint32_t, kItems * kPayloadWords>;

// These are deterministic, unfunded *test* keys. Never use them as wallets.
bool makeTestVectors(PublicBatch& pubs, PayloadBatch& payloads, std::string& error) {
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
        const unsigned char* xy = serialized + 1;
        for (size_t word = 0; word < kPublicWords; ++word) {
            const size_t j = word * 4;
            pubs[i * kPublicWords + word] = uint32_t(xy[j]) |
                (uint32_t(xy[j + 1]) << 8) | (uint32_t(xy[j + 2]) << 16) |
                (uint32_t(xy[j + 3]) << 24);
        }
        unsigned char hash[32];
        keccak256(xy, 64, hash);
        unsigned char payload[24]{};
        payload[0] = 0x41;
        std::memcpy(payload + 1, hash + 12, 20);
        for (size_t word = 0; word < kPayloadWords; ++word) {
            const size_t j = word * 4;
            payloads[i * kPayloadWords + word] = uint32_t(payload[j]) |
                (uint32_t(payload[j + 1]) << 8) | (uint32_t(payload[j + 2]) << 16) |
                (uint32_t(payload[j + 3]) << 24);
        }
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
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    ~State() {
        if (device) vkDeviceWaitIdle(device);
        if (device && fence) vkDestroyFence(device, fence, nullptr);
        if (device && commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device && pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (device && shader) vkDestroyShaderModule(device, shader, nullptr);
        if (device && pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (device && descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (device && descriptorLayout) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        if (device && buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (device && memory) vkFreeMemory(device, memory, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

bool runKeccak(std::string& error) {
    PublicBatch pubs{};
    PayloadBatch expected{};
    if (!makeTestVectors(pubs, expected, error)) return false;
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
    const VkDeviceSize alignment = properties.limits.minStorageBufferOffsetAlignment;
    const VkDeviceSize outputOffset = (inBytes + alignment - 1) / alignment * alignment;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = outputOffset + outBytes;
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
    std::memcpy(mapped, pubs.data(), inBytes);
    std::memset(static_cast<unsigned char*>(mapped) + outputOffset, 0, outBytes);
    vkUnmapMemory(state.device, state.memory);

    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (!check(vkCreateDescriptorSetLayout(state.device, &layoutInfo, nullptr, &state.descriptorLayout),
               "vkCreateDescriptorSetLayout")) return false;
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
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
    VkDescriptorBufferInfo ranges[2] = {
        {state.buffer, 0, inBytes}, {state.buffer, outputOffset, outBytes}
    };
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &ranges[i];
    }
    vkUpdateDescriptorSets(state.device, 2, writes, 0, nullptr);

    VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = sizeof(kVulkanKeccakSpv);
    shaderInfo.pCode = kVulkanKeccakSpv;
    if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.shader),
               "vkCreateShaderModule")) return false;
    VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
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
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout,
                            0, 1, &set, 0, nullptr);
    vkCmdPushConstants(command, state.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(kItems), &kItems);
    vkCmdDispatch(command, kItems / 64, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
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
    bool valid = std::memcmp(static_cast<const unsigned char*>(mapped) + outputOffset,
                             expected.data(), outBytes) == 0;
    vkUnmapMemory(state.device, state.memory);
    if (!valid) { error = "Vulkan Keccak payload differs from CPU reference"; return false; }
    return true;
}
}

int vulkanKeccakSelfTest() {
    std::string error;
    if (!runKeccak(error)) {
        std::cerr << "Vulkan Keccak stage test failed: " << error << "\n";
        return error == "no Vulkan compute device with shaderInt64" ? 77 : 1;
    }
    std::cout << "Vulkan Keccak-256 TRON payload stage PASS (256 public keys; no wallets)\n";
    return 0;
}
