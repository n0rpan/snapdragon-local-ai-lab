#pragma once
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
// Called only inside an explicitly released GPU fixture, never in build/CPU-half.
static void owned_gpu_identity() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "local-ai-owned-tiny";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("identity instance creation failed");
    try {
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count != 1)
            throw std::runtime_error("exactly one physical device under pinned ICD required");
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        if (vkEnumeratePhysicalDevices(instance, &count, &physical) != VK_SUCCESS || count != 1)
            throw std::runtime_error("identity enumeration changed");
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &driver;
        vkGetPhysicalDeviceProperties2(physical, &props);
        const auto & p = props.properties;
        if (p.vendorID != 0x5143 || p.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            driver.driverID != VK_DRIVER_ID_MESA_TURNIP || p.apiVersion < VK_API_VERSION_1_2 ||
            !std::strstr(p.deviceName, "Adreno"))
            throw std::runtime_error("Qualcomm integrated Adreno with Mesa Turnip required; no software proxy");
        std::printf("OWNED_GPU_ID vendor=%u device=%u type=%u driver=%u api=%u namehex=",
                    p.vendorID,p.deviceID,unsigned(p.deviceType),unsigned(driver.driverID),p.apiVersion);
        for (size_t n=0;n<sizeof(p.deviceName) && p.deviceName[n];++n)
            std::printf("%02x",unsigned(static_cast<unsigned char>(p.deviceName[n])));
        std::puts("");std::fflush(stdout);
    } catch (...) { vkDestroyInstance(instance, nullptr); throw; }
    vkDestroyInstance(instance, nullptr);
}
