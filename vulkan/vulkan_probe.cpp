#include "vulkan_probe.h"
#include "vulkan_probe_spv.h"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kItems = 256;

struct State {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;

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

bool createInstance(State& state, std::string& error) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "TRON vanity Vulkan compute probe";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    const VkResult result = vkCreateInstance(&info, nullptr, &state.instance);
    if (result != VK_SUCCESS) {
        error = "vkCreateInstance failed: " + std::to_string(result);
        return false;
    }
    return true;
}

bool selectComputeDevice(State& state, std::string& error) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(state.instance, &count, nullptr) != VK_SUCCESS || !count) {
        error = "no Vulkan physical devices";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(state.instance, &count, devices.data()) != VK_SUCCESS) {
        error = "enumerating Vulkan devices failed";
        return false;
    }
    for (auto physical : devices) {
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
        std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, props.data());
        for (uint32_t i = 0; i < families; ++i) {
            if (props[i].queueCount && (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                state.physical = physical;
                state.queueFamily = i;
                return true;
            }
        }
    }
    error = "no Vulkan compute queue";
    return false;
}

bool check(VkResult result, const char* call, std::string& error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(call) + " failed: " + std::to_string(result);
    return false;
}
}

std::string vulkanDeviceSummary() {
    State state;
    std::string error;
    if (!createInstance(state, error) || !selectComputeDevice(state, error)) return "Vulkan unavailable: " + error;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(state.physical, &props);
    return std::string(props.deviceName) + " (Vulkan compute queue " + std::to_string(state.queueFamily) + ")";
}

int vulkanComputeSelfTest() {
    State state;
    std::string error;
    if (!createInstance(state, error) || !selectComputeDevice(state, error)) {
        std::cerr << "Vulkan probe: " << error << "\n";
        return 1;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(state.physical, &props);
    std::cout << "Vulkan compute probe: " << props.deviceName << "\n";
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = state.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (!check(vkCreateDevice(state.physical, &deviceInfo, nullptr, &state.device), "vkCreateDevice", error)) goto fail;

    {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = sizeof(uint32_t) * kItems;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!check(vkCreateBuffer(state.device, &bufferInfo, nullptr, &state.buffer), "vkCreateBuffer", error)) goto fail;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(state.device, state.buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memoryProps{};
        vkGetPhysicalDeviceMemoryProperties(state.physical, &memoryProps);
        uint32_t type = memoryProps.memoryTypeCount;
        for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memoryProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                type = i; break;
            }
        if (type == memoryProps.memoryTypeCount) { error = "no coherent host-visible Vulkan buffer memory"; goto fail; }
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        if (!check(vkAllocateMemory(state.device, &allocate, nullptr, &state.memory), "vkAllocateMemory", error) ||
            !check(vkBindBufferMemory(state.device, state.buffer, state.memory, 0), "vkBindBufferMemory", error)) goto fail;
    }
    {
        void* mapped = nullptr;
        if (!check(vkMapMemory(state.device, state.memory, 0, sizeof(uint32_t) * kItems, 0, &mapped),
                   "vkMapMemory input", error)) goto fail;
        auto* values = static_cast<uint32_t*>(mapped);
        for (uint32_t i = 0; i < kItems; ++i) values[i] = i;
        vkUnmapMemory(state.device, state.memory);
    }
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = 1;
        info.pBindings = &binding;
        if (!check(vkCreateDescriptorSetLayout(state.device, &info, nullptr, &state.descriptorLayout),
                   "vkCreateDescriptorSetLayout", error)) goto fail;
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
        if (!check(vkCreateDescriptorPool(state.device, &pool, nullptr, &state.descriptorPool),
                   "vkCreateDescriptorPool", error)) goto fail;
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = state.descriptorPool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &state.descriptorLayout;
        VkDescriptorSet descriptors = VK_NULL_HANDLE;
        if (!check(vkAllocateDescriptorSets(state.device, &allocation, &descriptors),
                   "vkAllocateDescriptorSets", error)) goto fail;
        VkDescriptorBufferInfo buffer{state.buffer, 0, sizeof(uint32_t) * kItems};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptors; write.dstBinding = 0; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &buffer;
        vkUpdateDescriptorSets(state.device, 1, &write, 0, nullptr);

        VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = sizeof(kVulkanProbeSpv);
        shaderInfo.pCode = kVulkanProbeSpv;
        if (!check(vkCreateShaderModule(state.device, &shaderInfo, nullptr, &state.shader),
                   "vkCreateShaderModule", error)) goto fail;
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1; layout.pSetLayouts = &state.descriptorLayout;
        if (!check(vkCreatePipelineLayout(state.device, &layout, nullptr, &state.pipelineLayout),
                   "vkCreatePipelineLayout", error)) goto fail;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = state.shader;
        pipeline.stage.pName = "main";
        pipeline.layout = state.pipelineLayout;
        if (!check(vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &state.pipeline),
                   "vkCreateComputePipelines", error)) goto fail;
        VkCommandPoolCreateInfo commandPool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        commandPool.queueFamilyIndex = state.queueFamily;
        if (!check(vkCreateCommandPool(state.device, &commandPool, nullptr, &state.commandPool),
                   "vkCreateCommandPool", error)) goto fail;
        VkCommandBufferAllocateInfo commandAllocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commandAllocate.commandPool = state.commandPool;
        commandAllocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocate.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (!check(vkAllocateCommandBuffers(state.device, &commandAllocate, &command),
                   "vkAllocateCommandBuffers", error)) goto fail;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer", error)) goto fail;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout,
                                0, 1, &descriptors, 0, nullptr);
        vkCmdDispatch(command, kItems / 64, 1, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        if (!check(vkEndCommandBuffer(command), "vkEndCommandBuffer", error)) goto fail;
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (!check(vkCreateFence(state.device, &fenceInfo, nullptr, &state.fence), "vkCreateFence", error)) goto fail;
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(state.device, state.queueFamily, 0, &queue);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        if (!check(vkQueueSubmit(queue, 1, &submit, state.fence), "vkQueueSubmit", error) ||
            !check(vkWaitForFences(state.device, 1, &state.fence, VK_TRUE, 10000000000ull),
                   "vkWaitForFences", error)) goto fail;
    }
    {
        void* mapped = nullptr;
        if (!check(vkMapMemory(state.device, state.memory, 0, sizeof(uint32_t) * kItems, 0, &mapped),
                   "vkMapMemory output", error)) goto fail;
        const auto* values = static_cast<const uint32_t*>(mapped);
        bool valid = true;
        for (uint32_t i = 0; i < kItems; ++i)
            if (values[i] != (i ^ 0x9e3779b9u) * 1664525u + 1013904223u) {
                valid = false; break;
            }
        vkUnmapMemory(state.device, state.memory);
        if (!valid) { error = "Vulkan compute output differs from CPU reference"; goto fail; }
    }
    std::cout << "Vulkan deterministic compute PASS (256 values; no wallet data)\n";
    return 0;
fail:
    std::cerr << "Vulkan compute probe failed: " << error << "\n";
    return 1;
}
