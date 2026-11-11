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
#include <string>
#include <vector>
using C=std::chrono::steady_clock;
static double elapsed(C::time_point t){return std::chrono::duration<double>(C::now()-t).count();}
static bool capture=false;static std::vector<double> dsp;
static void logging(ggml_log_level,const char*text,void*){
 if(strstr(text,"profile-op")){if(capture&&strstr(text,"ceiling_mm_")&&!strstr(text,"OPBATCH")){auto u=strstr(text,"|usec ");if(u)dsp.push_back(atof(u+6));}return;}
 fputs(text,stderr);
}
static double med(std::vector<double>v){if(v.empty())return NAN;std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**){
 setvbuf(stdout,nullptr,_IONBF,0);ggml_log_set(logging,nullptr);
 auto reg=ggml_backend_hexagon_reg();if(!reg||!ggml_backend_reg_dev_count(reg))return 2;auto dev=ggml_backend_reg_dev_get(reg,0);if(strncmp(ggml_backend_dev_name(dev),"HTP",3))return 2;auto be=ggml_backend_dev_init(dev,nullptr);if(!be)return 3;
 std::mt19937 rng(42);std::uniform_real_distribution<float>rand(-.5f,.5f);
 for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL})for(int m:{512,8192})for(int n:{512,1024,2048}){
  if(argc>1&&(m!=8192||n!=2048))continue;
  constexpr int k=2048;int maxops=(m==8192&&n==2048)?8:16;
  auto start=C::now();auto ctx=ggml_init({4*1024*1024,nullptr,true});auto a=ggml_new_tensor_2d(ctx,type,k,m);auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);std::vector<ggml_tensor*>cs;
  for(int i=0;i<maxops;i++){auto c=ggml_mul_mat(ctx,a,b);ggml_format_name(c,"ceiling_mm_%d",i);if(!ggml_backend_dev_supports_op(dev,c)){fprintf(stderr,"UNSUPPORTED ceiling type/shape\n");return 4;}cs.push_back(c);}
  std::vector<int>counts={1,4,8};if(maxops==16)counts.push_back(16);std::vector<ggml_cgraph*>gs;
  for(int count:counts){auto g=ggml_new_graph_custom(ctx,128,false);for(int i=0;i<count;i++)ggml_build_forward_expand(g,cs[i]);gs.push_back(g);}
  double build_s=elapsed(start);size_t bytes=ggml_nbytes(a)+ggml_nbytes(b)+maxops*ggml_nbytes(cs[0]);if(bytes+64ull*1024*1024>1024ull*1024*1024){fprintf(stderr,"budget refusal\n");return 5;}
  start=C::now();auto buf=ggml_backend_alloc_ctx_tensors(ctx,be);if(!buf)return 6;double alloc_s=elapsed(start);
  std::vector<float>row(k),act(k*n);std::vector<unsigned char>enc(ggml_nbytes(a));auto tr=ggml_get_type_traits(type);
  for(int j=0;j<m;j++){for(auto&v:row)v=rand(rng);tr->from_float_ref(row.data(),enc.data()+j*ggml_row_size(type,k),k);}for(auto&v:act)v=rand(rng);
  start=C::now();ggml_backend_tensor_set(a,enc.data(),0,enc.size());ggml_backend_tensor_set(b,act.data(),0,act.size()*4);double upload_s=elapsed(start);
  std::vector<double>wall,submit,sync,prof;std::vector<int>samples;
  for(size_t gi=0;gi<gs.size();gi++){
   for(int w=0;w<2;w++)if(ggml_backend_graph_compute(be,gs[gi])!=GGML_STATUS_SUCCESS)return 7;
   std::vector<double>wt,st,sy;dsp.clear();capture=true;
   for(int it=0;it<5;it++){auto t=C::now();auto status=ggml_backend_graph_compute_async(be,gs[gi]);double submitted=elapsed(t);ggml_backend_synchronize(be);double total=elapsed(t);if(status!=GGML_STATUS_SUCCESS)return 7;wt.push_back(total);st.push_back(submitted);sy.push_back(total-submitted);}
   capture=false;wall.push_back(med(wt));submit.push_back(med(st));sync.push_back(med(sy));prof.push_back(med(dsp));samples.push_back(dsp.size());
   // Every distinct output is materialized and checked; no alias or CSE shortcut.
   for(int op=0;op<counts[gi];op++){
    double err2=0,ref2=0;
    for(int j=0;j<17;j++){int rr=(j*127)%m,col=(j*31)%n;tr->to_float(enc.data()+rr*ggml_row_size(type,k),row.data(),k);double ref=0;for(int z=0;z<k;z++)ref+=double(row[z])*act[col*k+z];float got;ggml_backend_tensor_get(cs[op],&got,(col*m+rr)*4,4);if(!std::isfinite(got))return 8;err2+=(got-ref)*(got-ref);ref2+=ref*ref;}
    double rel=sqrt(err2/std::max(ref2,1e-30));if(rel>.03){fprintf(stderr,"ceiling sparse RMSE exceeds3 percent\n");return 8;}
    fprintf(stderr,"CHECK type=%s M=%d N=%d ops=%d output=%d sparse_relative_rmse=%.9f\n",ggml_type_name(type),m,n,counts[gi],op,rel);
   }
  }
  double sx=0,sy=0,sxx=0,sxy=0,num=counts.size();for(size_t i=0;i<counts.size();i++){sx+=counts[i];sy+=wall[i];sxx+=counts[i]*counts[i];sxy+=counts[i]*wall[i];}double slope=(num*sxy-sx*sy)/(num*sxx-sx*sx),intercept=(sy-slope*sx)/num;
  printf("{\"kind\":\"npu-prefill-ceiling\",\"type\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"buffer_bytes\":%zu,\"build_seconds\":%.9f,\"allocation_seconds\":%.9f,\"upload_seconds\":%.9f,\"slope_seconds_per_op\":%.9f,\"intercept_seconds\":%.9f,\"effective_tops_slope\":%.6f,\"measurements\":[",ggml_type_name(type),k,m,n,bytes,build_s,alloc_s,upload_s,slope,intercept,2.0*k*m*n/slope/1e12);
  for(size_t i=0;i<counts.size();i++)printf("%s{\"ops\":%d,\"wall_seconds\":%.9f,\"submit_seconds\":%.9f,\"sync_seconds\":%.9f,\"dsp_op_us\":%.3f,\"profile_samples\":%d,\"effective_tops_wall\":%.6f}",i?",":"",counts[i],wall[i],submit[i],sync[i],prof[i],samples[i],2.0*k*m*n*counts[i]/wall[i]/1e12);
  puts("]}");ggml_backend_buffer_free(buf);ggml_free(ctx);
 }
 ggml_backend_free(be);return 0;
}
