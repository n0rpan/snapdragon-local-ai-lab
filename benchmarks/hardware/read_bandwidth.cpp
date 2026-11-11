// Direct Adreno Vulkan resident reads, optionally with simultaneous CPU reads.
// argv: shader.spv 512|1024 CPU_threads(0=GPUonly) seconds [MONOTONIC_start_ns]
#include <vulkan/vulkan.h>
#include "gpu_identity.h"
#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
constexpr uint32_t GROUPS=256,PASSES=32;
static void need(bool v,const char *s){if(!v)throw std::runtime_error(s);}
static void vkcheck(VkResult r,const char *s){need(r==VK_SUCCESS,s);}
static uint64_t now(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
struct Buffer { VkBuffer b{};VkDeviceMemory m{};void *map{};uint64_t bytes{},base{}; };
struct GPU {
    VkInstance instance{};VkDevice device{};VkPhysicalDevice physical{};VkQueue queue{};uint32_t family{};
    VkCommandPool pool{};VkFence fence{};VkDescriptorPool descriptors{};VkDescriptorSetLayout setlayout{};
    VkPipelineLayout layout{};VkPipeline pipeline{};VkShaderModule shader{};
    VkPhysicalDeviceMemoryProperties memory{};VkPhysicalDeviceProperties props{};
    std::vector<Buffer> buffers;
    ~GPU(){
        if(device)vkDeviceWaitIdle(device);
        for(auto &b:buffers){if(b.map)vkUnmapMemory(device,b.m);if(b.b)vkDestroyBuffer(device,b.b,nullptr);if(b.m)vkFreeMemory(device,b.m,nullptr);}
        if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(shader)vkDestroyShaderModule(device,shader,nullptr);
        if(layout)vkDestroyPipelineLayout(device,layout,nullptr);if(setlayout)vkDestroyDescriptorSetLayout(device,setlayout,nullptr);
        if(descriptors)vkDestroyDescriptorPool(device,descriptors,nullptr);if(fence)vkDestroyFence(device,fence,nullptr);
        if(pool)vkDestroyCommandPool(device,pool,nullptr);if(device)vkDestroyDevice(device,nullptr);if(instance)vkDestroyInstance(instance,nullptr);
    }
    size_t buffer(uint64_t bytes,VkMemoryPropertyFlags wanted){
        buffers.push_back({});auto &b=buffers.back();b.bytes=bytes;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=bytes;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        vkcheck(vkCreateBuffer(device,&bi,nullptr,&b.b),"buffer create");VkMemoryRequirements mr{};vkGetBufferMemoryRequirements(device,b.b,&mr);
        uint32_t type=memory.memoryTypeCount;
        for(uint32_t i=0;i<memory.memoryTypeCount;++i)if((mr.memoryTypeBits&(1u<<i))&&(memory.memoryTypes[i].propertyFlags&wanted)==wanted){type=i;break;}
        need(type<memory.memoryTypeCount,"required coherent UMA memory type unavailable");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=mr.size;ai.memoryTypeIndex=type;
        vkcheck(vkAllocateMemory(device,&ai,nullptr,&b.m),"memory allocate");vkcheck(vkBindBufferMemory(device,b.b,b.m,0),"memory bind");
        vkcheck(vkMapMemory(device,b.m,0,bytes,0,&b.map),"map coherent memory");return buffers.size()-1;
    }
    void init(){
        owned_gpu_identity();VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ic.pApplicationInfo=&app;vkcheck(vkCreateInstance(&ic,nullptr,&instance),"instance");
        uint32_t count=0;vkcheck(vkEnumeratePhysicalDevices(instance,&count,nullptr),"enumeration");need(count==1,"one pinned Adreno device required");
        vkcheck(vkEnumeratePhysicalDevices(instance,&count,&physical),"physical device");vkGetPhysicalDeviceProperties(physical,&props);vkGetPhysicalDeviceMemoryProperties(physical,&memory);
        need(props.vendorID==0x5143&&props.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,"no CPU/software fallback");
        need(props.limits.maxComputeWorkGroupInvocations>=256&&props.limits.maxComputeWorkGroupSize[0]>=256,"256-lane workgroups unsupported");
        need(props.limits.maxComputeSharedMemorySize>=4096,"shared reduction storage unsupported");
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
        family=count;for(uint32_t i=0;i<count;++i)if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT){family=i;break;}need(family<count,"compute queue missing");
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;vkcheck(vkCreateDevice(physical,&di,nullptr,&device),"device");vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;vkcheck(vkCreateCommandPool(device,&pi,nullptr,&pool),"command pool");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};vkcheck(vkCreateFence(device,&fi,nullptr,&fence),"fence");
    }
};
struct CPU {
    std::vector<uint64_t> data,passes,checks,starts,ends;std::vector<std::thread> workers;std::vector<int> cpus;
    std::atomic<bool> go{false},stop{false},bad{false};std::atomic<size_t> ready{0};size_t count;
    CPU(uint64_t bytes,size_t threads):data(threads?bytes/8:0),passes(threads),checks(threads),starts(threads),ends(threads),count(threads){
        for(size_t i=0;i<data.size();++i)data[i]=i+1;
        cpu_set_t mask;CPU_ZERO(&mask);need(sched_getaffinity(0,sizeof(mask),&mask)==0,"CPU affinity query");
        for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&mask))cpus.push_back(i);need(threads<=cpus.size(),"too many CPU readers");
    }
    void finish(){stop=true;for(auto &w:workers)if(w.joinable())w.join();}
    ~CPU(){finish();}
    void start(){for(size_t t=0;t<count;++t)workers.emplace_back([this,t]{
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpus[t],&mask);if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask))bad=true;
        const size_t blocks=data.size()/8,lo=blocks*t/count,hi=blocks*(t+1)/count;
        const uint64_t expected=uint64_t((__uint128_t(lo*8+1+hi*8)*(hi-lo)*8)/2);
        ++ready;while(!go&&!stop)std::this_thread::yield();if(stop)return;starts[t]=now();
        while(!stop){
            asm volatile("":::"memory");
            auto a=vdupq_n_u64(0),b=a,c=a,d=a;
            for(size_t i=lo;i<hi;++i){const uint64_t *p=data.data()+i*8;a=vaddq_u64(a,vld1q_u64(p));b=vaddq_u64(b,vld1q_u64(p+2));c=vaddq_u64(c,vld1q_u64(p+4));d=vaddq_u64(d,vld1q_u64(p+6));}
            const uint64_t sum=vaddvq_u64(vaddq_u64(vaddq_u64(a,b),vaddq_u64(c,d)));
            if(sum!=expected){bad=true;stop=true;break;}checks[t]+=sum;++passes[t];ends[t]=now();
        }
    });while(ready<count)std::this_thread::yield();need(!bad,"CPU reader affinity setup");}
};
int main(int argc,char **argv){try{
    need(argc==5||argc==6,"usage: gpu-bw shader.spv 512|1024 CPU_threads seconds [MONOTONIC_start_ns]");
    uint64_t bytes=std::stoull(argv[2])*1024*1024;need(bytes==512ull*1024*1024||bytes==1024ull*1024*1024,"input must be512MiB or1GiB");
    const size_t threads=std::stoul(argv[3]);need(threads<=12,"CPU readers bounded to12");const double duration=std::stod(argv[4]);need(duration>=3&&duration<=30,"duration must be3–30seconds");
    const uint64_t requested=argc==6?std::stoull(argv[5]):0;
    GPU g;g.init();CPU cpu(bytes,threads);
    const uint64_t limit=(std::min<uint64_t>(g.props.limits.maxStorageBufferRange,128ull*1024*1024)/16)*16;need(limit>=16,"storage range");
    need(g.props.limits.maxStorageBufferRange>=uint64_t(PASSES)*GROUPS*16,"output storage range");
    const uint32_t chunks=uint32_t((bytes+limit-1)/limit);need(chunks<=64,"bounded storage chunks");
    std::vector<size_t> inputs,outputs;
    for(uint64_t base=0;base<bytes;base+=limit){
        size_t i=g.buffer(std::min(limit,bytes-base),VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);inputs.push_back(i);g.buffers[i].base=base/16;
        auto *p=static_cast<uint32_t*>(g.buffers[i].map);for(size_t word=0;word<g.buffers[i].bytes/4;++word)p[word]=uint32_t(base/4+word+1);
        size_t o=g.buffer(uint64_t(PASSES)*GROUPS*16,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);outputs.push_back(o);
    }
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);need(bool(file),"SPIR-V open");size_t size=size_t(file.tellg());need(size>0&&size<=1024*1024&&size%4==0,"bounded SPIR-V");file.seekg(0);std::vector<uint32_t> code(size/4);file.read(reinterpret_cast<char*>(code.data()),size);need(bool(file),"SPIR-V read");
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};si.codeSize=size;si.pCode=code.data();vkcheck(vkCreateShaderModule(g.device,&si,nullptr,&g.shader),"shader");
    VkDescriptorSetLayoutBinding bindings[2]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=2;sl.pBindings=bindings;vkcheck(vkCreateDescriptorSetLayout(g.device,&sl,nullptr,&g.setlayout),"set layout");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,8};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&g.setlayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&range;vkcheck(vkCreatePipelineLayout(g.device,&pl,nullptr,&g.layout),"pipeline layout");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=g.shader;ci.stage.pName="main";ci.layout=g.layout;vkcheck(vkCreateComputePipelines(g.device,VK_NULL_HANDLE,1,&ci,nullptr,&g.pipeline),"compute pipeline");
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2*chunks};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=chunks;dp.poolSizeCount=1;dp.pPoolSizes=&ps;vkcheck(vkCreateDescriptorPool(g.device,&dp,nullptr,&g.descriptors),"descriptor pool");
    std::vector<VkDescriptorSetLayout> layouts(chunks,g.setlayout);std::vector<VkDescriptorSet> sets(chunks);VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=g.descriptors;da.descriptorSetCount=chunks;da.pSetLayouts=layouts.data();vkcheck(vkAllocateDescriptorSets(g.device,&da,sets.data()),"descriptor sets");
    for(uint32_t i=0;i<chunks;++i){VkDescriptorBufferInfo info[2]={{g.buffers[inputs[i]].b,0,g.buffers[inputs[i]].bytes},{g.buffers[outputs[i]].b,0,g.buffers[outputs[i]].bytes}};VkWriteDescriptorSet writes[2]{};for(uint32_t j=0;j<2;++j){writes[j].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[j].dstSet=sets[i];writes[j].dstBinding=j;writes[j].descriptorCount=1;writes[j].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[j].pBufferInfo=&info[j];}vkUpdateDescriptorSets(g.device,2,writes,0,nullptr);}
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=g.pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;VkCommandBuffer command;vkcheck(vkAllocateCommandBuffers(g.device,&ca,&command),"command buffer");
    VkCommandBufferBeginInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};vkcheck(vkBeginCommandBuffer(command,&cb),"record begin");vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,g.pipeline);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    for(uint32_t pass=0;pass<PASSES;++pass){for(uint32_t i=0;i<chunks;++i){uint32_t push[2]={uint32_t(g.buffers[inputs[i]].bytes/16),pass};vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,g.layout,0,1,&sets[i],0,nullptr);vkCmdPushConstants(command,g.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,push);vkCmdDispatch(command,GROUPS,1,1);}vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);}
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);vkcheck(vkEndCommandBuffer(command),"record end");
    auto batch=[&]{VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;vkcheck(vkResetFences(g.device,1,&g.fence),"fence reset");vkcheck(vkQueueSubmit(g.queue,1,&submit,g.fence),"submit");vkcheck(vkWaitForFences(g.device,1,&g.fence,VK_TRUE,120000000000ull),"120second GPU deadline");};
    batch(); // all initialization, first shader/JIT and warmup excluded
    cpu.start();std::cout<<"READY monotonic_ns="<<now()<<" requested_start_ns="<<requested<<std::endl;
    if(requested){need(requested>now(),"requested start already passed after initialization");while(now()<requested)std::this_thread::sleep_for(std::chrono::microseconds(50));}
    const uint64_t gpu_start=now();cpu.go=true;uint64_t batches=0,gpu_end=0;
    do{batch();++batches;gpu_end=now();need(!cpu.bad,"CPU checksum or affinity failure");}while(double(gpu_end-gpu_start)/1e9<duration);
    cpu.finish();uint64_t cpu_bytes=0,cpu_start=gpu_start,cpu_end=gpu_end,cpu_checksum=0,cpu_passes=0;
    if(threads){cpu_start=*std::min_element(cpu.starts.begin(),cpu.starts.end());cpu_end=*std::max_element(cpu.ends.begin(),cpu.ends.end());}
    for(size_t t=0;t<threads;++t){size_t blocks=cpu.data.size()/8;cpu_bytes+=(blocks*(t+1)/threads-blocks*t/threads)*64*cpu.passes[t];cpu_checksum+=cpu.checks[t];cpu_passes+=cpu.passes[t];}
    uint64_t gpu_checksum=0;
    for(uint32_t chunk=0;chunk<chunks;++chunk){auto &input=g.buffers[inputs[chunk]];auto *actual=static_cast<uint32_t*>(g.buffers[outputs[chunk]].map);uint64_t vectors=input.bytes/16;
        for(uint32_t group=0;group<GROUPS;++group){uint64_t count=0,sum=0;for(uint64_t first=group*256;first<vectors;first+=GROUPS*256){uint64_t length=std::min<uint64_t>(256,vectors-first);count+=length;sum+=length*(2*first+length-1)/2;}
            for(uint32_t pass=0;pass<PASSES;++pass)for(uint32_t c=0;c<4;++c){uint32_t expected=uint32_t(4*sum+(input.base*4+c+1)*count+(c==0?pass:0));uint32_t got=actual[(pass*GROUPS+group)*4+c];need(got==expected,"independent GPU checksum failure");gpu_checksum+=got;}}
    }
    need(!cpu.bad,"CPU checksum failed");const uint64_t start=std::min(gpu_start,cpu_start),end=std::max(gpu_end,cpu_end),gpu_bytes=bytes*PASSES*batches;
    const double seconds=double(end-start)/1e9,overlap=threads?double(std::min(gpu_end,cpu_end)-std::max(gpu_start,cpu_start))/1e9:0;
    std::cout<<"{\"gpu_bytes\":"<<gpu_bytes<<",\"cpu_bytes\":"<<cpu_bytes<<",\"gpu_passes\":"<<PASSES*batches<<",\"cpu_partition_passes\":"<<cpu_passes<<",\"cpu_threads\":"<<threads<<",\"resident_input_bytes\":"<<bytes<<",\"chunks\":"<<chunks<<",\"max_storage_buffer_range\":"<<g.props.limits.maxStorageBufferRange<<",\"gpu_start_ns\":"<<gpu_start<<",\"gpu_end_ns\":"<<gpu_end<<",\"cpu_start_ns\":"<<cpu_start<<",\"cpu_end_ns\":"<<cpu_end<<",\"union_seconds\":"<<seconds<<",\"overlap_seconds\":"<<overlap<<",\"gpu_union_GBps\":"<<gpu_bytes/seconds/1e9<<",\"cpu_union_GBps\":"<<cpu_bytes/seconds/1e9<<",\"combined_union_GBps\":"<<(gpu_bytes+cpu_bytes)/seconds/1e9<<",\"gpu_checksum\":"<<gpu_checksum<<",\"cpu_checksum\":"<<cpu_checksum<<",\"checksums_passed\":true,\"timing\":\"init/transfers/warmup excluded; union includes small launch/drain tails\"}"<<std::endl;
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"gpu_bw failed: %s\n",e.what());return 1;}}
