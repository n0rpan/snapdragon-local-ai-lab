#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
static void inspect(const char *phase, int threads, int pin, int reset) {
    int masks[12]={0};
    const int placement[4]={0,1,4,8};
    #pragma omp parallel num_threads(threads)
    {
        int index=omp_get_thread_num();
        cpu_set_t mask;
        if (pin || reset) {
            CPU_ZERO(&mask);
            if (pin) CPU_SET(placement[index],&mask);
            else for(int cpu=0;cpu<12;cpu++)CPU_SET(cpu,&mask);
            if(sched_setaffinity(0,sizeof(mask),&mask)!=0)masks[index]=-1;
        }
        CPU_ZERO(&mask);
        if(sched_getaffinity(0,sizeof(mask),&mask)==0) {
            for(int cpu=0;cpu<12;cpu++)if(CPU_ISSET(cpu,&mask))masks[index]|=1<<cpu;
        }
    }
    printf("%s",phase);
    for(int i=0;i<threads;i++)printf(" %03x",masks[i]);
    puts("");
}
int main(void) {
    inspect("initial-prefill",10,0,0);
    inspect("pinned-decode",4,1,0);
    inspect("implicit-prefill",10,0,0);
    inspect("explicit-reset-prefill",10,0,1);
    return 0;
}
