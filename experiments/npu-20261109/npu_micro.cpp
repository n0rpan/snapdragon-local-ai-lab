#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
static double sec(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
int main(int argc,char**argv){
 setvbuf(stdout,nullptr,_IONBF,0);ggml_backend_t be=nullptr;ggml_backend_buffer_t buf=nullptr;ggml_context*ctx=nullptr;
 auto release=[&]{if(buf){ggml_backend_buffer_free(buf);buf=nullptr;}if(ctx){ggml_free(ctx);ctx=nullptr;}};
 try{
  auto reg=ggml_backend_hexagon_reg();if(!reg||!ggml_backend_reg_dev_count(reg))throw std::runtime_error("no HTP device");
  auto dev=ggml_backend_reg_dev_get(reg,0);if(strncmp(ggml_backend_dev_name(dev),"HTP",3))throw std::runtime_error("not HTP");
  be=ggml_backend_dev_init(dev,nullptr);if(!be)throw std::runtime_error("HTP init failed");
  bool scale=argc>1&&!strcmp(argv[1],"scale");double duration=argc>2?atof(argv[2]):0;
  std::vector<ggml_type> types={GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_MXFP4,GGML_TYPE_IQ4_NL,GGML_TYPE_Q6_K};
  if(scale){
   int chains=argc>3?atoi(argv[3]):1;if(chains<1||chains>8)throw std::runtime_error("scale batch outside 1..8");
   ctx=ggml_init({1024*1024,nullptr,true});auto g=ggml_new_graph(ctx);
   std::vector<ggml_backend_buffer_t> buffers;std::vector<ggml_context*> contexts;std::vector<ggml_tensor*> outputs;
   std::vector<float> v(8*1024*1024,.25f);
   for(int i=0;i<chains;i++){
    auto cx=ggml_init({1024*1024,nullptr,true});auto a=ggml_new_tensor_1d(cx,GGML_TYPE_F32,8*1024*1024);auto c=ggml_scale(cx,a,1.001f);
    if(!ggml_backend_dev_supports_op(dev,c))throw std::runtime_error("scale unsupported");
    auto bf=ggml_backend_alloc_ctx_tensors(cx,be);if(!bf)throw std::runtime_error("scale allocation failed");
    contexts.push_back(cx);buffers.push_back(bf);outputs.push_back(c);ggml_backend_tensor_set(a,v.data(),0,ggml_nbytes(a));ggml_build_forward_expand(g,c);
   }
   printf("MICRO_BEGIN scale batch=%d\n",chains);auto start=Clock::now();int count=0;double total=0;
   do{auto t=Clock::now();if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("scale compute failed");ggml_backend_synchronize(be);double dt=sec(t);if(count)total+=dt;count++;}while(count<10||sec(start)<duration);
   for(auto c:outputs){float check=0;ggml_backend_tensor_get(c,&check,0,sizeof(check));if(std::abs(check-.25025f)>1e-4)throw std::runtime_error("scale incorrect");}
   size_t bytes=size_t(chains)*64*1024*1024;
   printf("{\"kind\":\"scale-effective-read-write\",\"supported\":true,\"batch_graphs\":%d,\"bytes_per_iteration\":%zu,\"iterations\":%d,\"seconds\":%.9f,\"gb_s\":%.6f,\"pure_read_bandwidth\":false}\n",chains,bytes,count-1,total,bytes*(count-1)/total/1e9);
   for(auto b:buffers)ggml_backend_buffer_free(b);for(auto c:contexts)ggml_free(c);release();ggml_backend_free(be);return 0;
  }
  std::mt19937 rng(42);std::uniform_real_distribution<float> rand(-.5f,.5f);
  const int shapes[][2]={{512,2048},{2048,512},{2560,640},{640,2560},{2560,10240}};
  for(auto type:types)for(auto &shape:shapes)for(int n:{1,4,16,512}){
   int k=shape[0],m=shape[1];if(argc>6&&(strcmp(argv[3],ggml_type_name(type))||atoi(argv[4])!=k||atoi(argv[5])!=m||atoi(argv[6])!=n))continue;if(k%ggml_blck_size(type)){printf("{\"kind\":\"matmul\",\"type\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"supported\":false,\"reason\":\"K not block aligned\"}\n",ggml_type_name(type),k,m,n);continue;}ctx=ggml_init({1024*1024,nullptr,true});if(!ctx)throw std::runtime_error("host context failed");
   auto a=ggml_new_tensor_2d(ctx,type,k,m);auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);auto c=ggml_mul_mat(ctx,a,b);
   if(!ggml_backend_dev_supports_op(dev,c)){printf("{\"kind\":\"matmul\",\"type\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"supported\":false}\n",ggml_type_name(type),k,m,n);release();continue;}
   size_t bytes=ggml_nbytes(a)+ggml_nbytes(b)+ggml_nbytes(c);if(bytes+4096>128ull*1024*1024)throw std::runtime_error("128 MiB buffer budget exceeded");
   auto g=ggml_new_graph(ctx);ggml_build_forward_expand(g,c);buf=ggml_backend_alloc_ctx_tensors(ctx,be);if(!buf)throw std::runtime_error("HTP allocation failed");
   auto tr=ggml_get_type_traits(type);if(!tr||!tr->from_float_ref||!tr->to_float)throw std::runtime_error("quantization traits missing");
   std::vector<float> row(k),act(k*n);std::vector<unsigned char> encoded(ggml_nbytes(a));
   for(int j=0;j<m;j++){for(auto&v:row)v=rand(rng);tr->from_float_ref(row.data(),encoded.data()+j*ggml_row_size(type,k),k);}
   for(auto&v:act)v=rand(rng);ggml_backend_tensor_set(a,encoded.data(),0,encoded.size());ggml_backend_tensor_set(b,act.data(),0,act.size()*sizeof(float));
   std::vector<double> samples;auto begin=Clock::now();int count=0;
   do{auto t=Clock::now();if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("HTP compute failed");ggml_backend_synchronize(be);if(count>=2)samples.push_back(sec(t));count++;}while(count<11||(duration>0&&sec(begin)<duration));
   // Sparse f64 reference against the dequantized weights; no CPU fallback.
   double err=0,ref2=0;for(int j=0;j<17;j++){int r=(j*127)%m,col=(j*31)%n;tr->to_float(encoded.data()+r*ggml_row_size(type,k),row.data(),k);double ref=0;for(int i=0;i<k;i++)ref+=double(row[i])*act[col*k+i];float got=0;ggml_backend_tensor_get(c,&got,(col*m+r)*sizeof(float),sizeof(float));if(!std::isfinite(got))throw std::runtime_error("nonfinite result");err+=(got-ref)*(got-ref);ref2+=ref*ref;}
   double rel=std::sqrt(err/std::max(ref2,1e-30));if(rel>.03)throw std::runtime_error("matmul error >3% sparse relative RMSE");
   std::sort(samples.begin(),samples.end());double dt=samples[samples.size()/2];printf("{\"kind\":\"quantized-matmul\",\"type\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"supported\":true,\"buffer_bytes\":%zu,\"median_seconds\":%.9f,\"samples\":%zu,\"effective_tops\":%.6f,\"sparse_relative_rmse\":%.9f}\n",ggml_type_name(type),k,m,n,bytes,dt,samples.size(),2.0*k*m*n/dt/1e12,rel);
   release();
  }
  ggml_backend_free(be);puts("MICRO CLEAN EXIT");return 0;
 }catch(const std::exception&e){fprintf(stderr,"MICRO STOP: %s\n",e.what());release();if(be)ggml_backend_free(be);return 1;}
}
