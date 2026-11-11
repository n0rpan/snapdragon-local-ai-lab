// Operator-only measured throughput. Compile against existing ggml libraries.
// No model, scheduler, backend discovery, CPU fallback, download or SDK build.
#include "ggml.h"
#include "ggml-backend.h"
#ifdef OWN_VULKAN
#include "ggml-vulkan.h"
#include "gpu_identity.h" // existing Adreno/Turnip-only identity check
#else
#include "ggml-hexagon.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void demand(bool value,const char *message){if(!value)throw std::runtime_error(message);}
struct Owned {
    ggml_backend_t backend=nullptr;
    ggml_context *ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ~Owned(){
        if(backend)ggml_backend_synchronize(backend);
        if(buffer)ggml_backend_buffer_free(buffer);
        if(ctx)ggml_free(ctx);
        if(backend)ggml_backend_free(backend);
    }
};

int main(int argc,char **argv){
    int status=0,n=0;std::string type_name,backend_name;
    double seconds[3]={},relative_rmse=0;size_t bytes=0;
    try{
        demand(argc==3,"usage: matmul-peak 1024|2048 f32|f16|q4_0");
        n=std::stoi(argv[1]);demand(n==1024||n==2048,"bounded square shape required");
        type_name=argv[2];ggml_type type;
        if(type_name=="f32")type=GGML_TYPE_F32;
        else if(type_name=="f16")type=GGML_TYPE_F16;
        else if(type_name=="q4_0")type=GGML_TYPE_Q4_0;
        else throw std::runtime_error("supported type is f32, f16 or q4_0");
        Owned own;
#ifdef OWN_VULKAN
        owned_gpu_identity();own.backend=ggml_backend_vk_init(0);
#else
        auto registry=ggml_backend_hexagon_reg();
        demand(registry&&ggml_backend_reg_dev_count(registry)==1,"one explicit HTP device required");
        auto dev=ggml_backend_reg_dev_get(registry,0);
        demand(!std::strncmp(ggml_backend_dev_name(dev),"HTP",3),"HTP backend required");
        own.backend=ggml_backend_dev_init(dev,nullptr);
#endif
        demand(own.backend,"explicit accelerator initialization failed");
        backend_name=ggml_backend_name(own.backend);
        own.ctx=ggml_init({4*1024*1024,nullptr,true});demand(own.ctx,"context allocation failed");
        auto a=ggml_new_tensor_2d(own.ctx,type,n,n);
        auto b=ggml_new_tensor_2d(own.ctx,GGML_TYPE_F32,n,n);
        auto c=ggml_mul_mat(own.ctx,a,b);
        auto graph=ggml_new_graph(own.ctx);ggml_build_forward_expand(graph,c);
        demand(ggml_backend_dev_supports_op(ggml_backend_get_device(own.backend),c),"MUL_MAT unsupported; no CPU fallback");
        for(auto tensor:{a,b,c})demand(ggml_nbytes(tensor)<256*1024*1024,"individual tensor exceeds256MiB");
        bytes=ggml_nbytes(a)+ggml_nbytes(b)+ggml_nbytes(c);
        own.buffer=ggml_backend_alloc_ctx_tensors(own.ctx,own.backend);demand(own.buffer,"accelerator buffer allocation failed");
        demand(ggml_backend_buffer_get_size(own.buffer)<256*1024*1024,"combined operator buffer exceeds256MiB");
        std::vector<float> weights(size_t(n)*n),input(weights.size()),output(weights.size()),row(n);
        std::mt19937 rng(42);std::uniform_real_distribution<float> random(-.5f,.5f);
        for(auto &value:weights)value=random(rng);
        for(auto &value:input)value=random(rng);
        std::vector<unsigned char> encoded(ggml_nbytes(a));
        if(type==GGML_TYPE_F32)std::memcpy(encoded.data(),weights.data(),encoded.size());
        else demand(ggml_quantize_chunk(type,weights.data(),encoded.data(),0,n,n,nullptr)==encoded.size(),"reference conversion failed");
        ggml_backend_tensor_set(a,encoded.data(),0,encoded.size());
        ggml_backend_tensor_set(b,input.data(),0,input.size()*sizeof(float));
        // Warmup includes shader/JIT initialization; tensors remain resident.
        demand(ggml_backend_graph_compute(own.backend,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(own.backend);
        for(int repeat=0;repeat<3;++repeat){
            const auto start=std::chrono::steady_clock::now();
            demand(ggml_backend_graph_compute(own.backend,graph)==GGML_STATUS_SUCCESS,"timed compute failed");
            ggml_backend_synchronize(own.backend);
            seconds[repeat]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            demand(std::isfinite(seconds[repeat])&&seconds[repeat]>0,"invalid synchronized duration");
        }
        ggml_backend_tensor_get(c,output.data(),0,output.size()*sizeof(float));
        for(auto value:output)demand(std::isfinite(value),"nonfinite accelerator output");
        // Cheap independent32-output scalar sanity check; conversion and copies
        // are outside timed device compute. This is not a model KLD gate.
        double error2=0,reference2=0;
        for(size_t sample=0;sample<32;++sample){
            const size_t position=(sample*131071)%output.size(),r=position%n,col=position/n;
            if(type==GGML_TYPE_F32)std::memcpy(row.data(),weights.data()+r*n,n*sizeof(float));
            else{
                auto traits=ggml_get_type_traits(type);demand(traits&&traits->to_float,"reference dequantizer missing");
                traits->to_float(encoded.data()+r*ggml_row_size(type,n),row.data(),n);
            }
            double reference=0;for(int k=0;k<n;++k)reference+=double(row[k])*input[col*n+k];
            const double error=output[position]-reference;error2+=error*error;reference2+=reference*reference;
        }
        relative_rmse=std::sqrt(error2/std::max(reference2,1e-30));
        demand(std::isfinite(relative_rmse)&&relative_rmse<=.03,"sampled scalar sanity error exceeds3percent");
    }catch(const std::exception &error){std::fprintf(stderr,"matmul_peak failed: %s\n",error.what());status=1;}
#ifdef OWN_VULKAN
    const bool clean=ggml_backend_vk_final_shutdown();
    const bool twice=ggml_backend_vk_final_shutdown();
    if(!clean||!twice)status=1;
#endif
    if(status)return status;
    auto sorted=std::vector<double>(seconds,seconds+3);std::sort(sorted.begin(),sorted.end());
    const double operations=2.*n*n*n/1e9;
    std::cout<<std::setprecision(10)<<"{\"backend\":"<<std::quoted(backend_name)<<",\"op\":\"MUL_MAT\",\"weight_type\":"<<std::quoted(type_name)
        <<",\"activation_type\":\"f32\",\"K\":"<<n<<",\"M\":"<<n<<",\"N\":"<<n<<",\"tensor_bytes\":"<<bytes
        <<",\"warmups\":1,\"seconds\":["<<seconds[0]<<","<<seconds[1]<<","<<seconds[2]<<"],\"median_gflops\":"<<operations/sorted[1]
        <<",\"min_gflops\":"<<operations/sorted[2]<<",\"max_gflops\":"<<operations/sorted[0]<<",\"scalar_samples\":32,\"relative_rmse\":"<<relative_rmse
        <<",\"timing_scope\":\"resident_backend_compute_plus_sync; transfers_and_initialization_excluded\"}"<<std::endl;
    return 0;
}
