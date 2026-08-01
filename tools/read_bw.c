/* Read-only bandwidth kernel, STREAM discipline (McCalpin rules):
 *   - array >> 4x LLC (18 MB here) : 1.5 GiB
 *   - NTIMES=20, best-of, first iteration discarded
 *   - parallel first touch, threads pinned to physical cores
 * Read-only has no write-allocate, so no correction factor applies.
 * STREAM itself has no pure-read kernel; LLM decode streams weights read-only. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N       200000000L
#define NTIMES  20

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static double a[N];

int main(void){
    #pragma omp parallel for              /* first touch by the owning thread */
    for(long i=0;i<N;i++) a[i]=1.0;

    double best=0, worst=1e30, sum=0, checksum=0;
    for(int r=0;r<NTIMES;r++){
        double t=now(), acc=0.0;
        #pragma omp parallel for reduction(+:acc)
        for(long i=0;i<N;i++) acc+=a[i];
        double dt=now()-t; checksum=acc;
        if(r==0) continue;                /* discard first, per STREAM */
        double gb=(N*8.0)/dt/1e9;
        if(gb>best) best=gb;
        if(gb<worst) worst=gb;
        sum+=gb;
    }
    printf("threads        : %d\n", omp_get_max_threads());
    printf("array          : %.1f MiB   (LLC = 18 MiB)\n", N*8.0/1048576.0);
    printf("READ best      : %.1f GB/s\n", best);
    printf("READ avg       : %.1f GB/s\n", sum/(NTIMES-1));
    printf("READ worst     : %.1f GB/s\n", worst);
    printf("checksum       : %.0f (expect %ld)\n", checksum, N);
    return 0;
}
