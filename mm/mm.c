#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
// #include <omp.h>
#include <time.h>   // for time measurements
#include <hero-target.h>

#define ARM_CLK_FREQ_MHZ 799

struct timespec start, stop;
double start_ns, stop_ns, exe_time;

#pragma omp declare target
int matmul(uint32_t * __restrict__ a, uint32_t * __restrict__ b, uint32_t * __restrict__ c, uint32_t width, uint32_t height)
{
    #pragma omp parallel for shared(a,b,c) collapse(2)
    for (int i=0; i<width; i++){
        for (int j=0; j<height; j++){
            int sum = 0;
            for (int k=0; k<width; k++)
                sum = sum + a[i*width+k] * b[k*width+j];
            c[i*width+j] = sum;
        }
    }
    return 0;
}
void mat_print(uint32_t * __restrict__ a,  uint32_t width, uint32_t height)
{
    for (int i=0; i<width; i++) {
        for (int  j=0; j<height; j++)
            printf("%d,", a[i*width+j]);
        printf("\n");
    }
    printf("\n");
}
#pragma omp end declare target

int main(int argc, char *argv[])
{    
    int width  = 64;
    int height = 64;
    uint32_t *a;
    uint32_t *b;
    uint32_t *c;
    uint32_t *g;

    if( argc > 1 )
    {
        width = atoi(argv[1]);
        height = atoi(argv[1]);
    }

    a = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
    b = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
    c = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
    g = (uint32_t *)malloc(sizeof(uint32_t)*width*height);

    /* Init */    
    for (int i=0; i<width; i++){
        for (int j=0; j<height; j++){
            a[i*width+j] = i*width+j;
            b[i*width+j] = i == j ? 2 : 0;
            c[i*width+j] = 0;
            g[i*width+j] = 0;
        }
    }

    for(int iter=0; iter < 4; ++iter)
    {
        printf("MatMul DMA! Width %d Height %d, a %x, b %x, c %x\n", (int) width, (int) height, (int) a, (int) b, (int) c);
        clock_gettime(CLOCK_REALTIME,&start);
        #pragma omp target map(to: a[0:width*height], b[0:width*height]) map(from: c[0:width*height])
        {
            uint8_t *local_space = (uint8_t *)hero_l1malloc(3*width*height*sizeof(uint32_t));
            uint32_t *local_a = (uint32_t *) &local_space[0*width*height*sizeof(uint32_t)];
            uint32_t *local_b = (uint32_t *) &local_space[1*width*height*sizeof(uint32_t)];
            uint32_t *local_c = (uint32_t *) &local_space[2*width*height*sizeof(uint32_t)];
            uint32_t local_w = width;
            uint32_t local_h = height;

            hero_dma_job_t dma0 = hero_dma_memcpy_async(local_a, a, local_w*local_h*sizeof(uint32_t));
            hero_dma_job_t dma1 = hero_dma_memcpy_async(local_b, b, local_w*local_h*sizeof(uint32_t));
            hero_dma_wait(dma0);
            hero_dma_wait(dma1);
            
            matmul(local_a, local_b, local_c, local_w, local_h);
            hero_dma_memcpy(c, local_c, local_w*local_h*sizeof(uint32_t));
            hero_l1free(local_space);
        }
        clock_gettime(CLOCK_REALTIME,&stop);        
        start_ns = ((double)(start.tv_sec))*1000000000 + (double)(start.tv_nsec);
        stop_ns  = ((double)(stop.tv_sec))*1000000000  + (double)(stop.tv_nsec);
        exe_time = (stop_ns - start_ns)/1000000000;
        printf("Exec Time [host cycles] = %.0f (%f s)\n", exe_time*ARM_CLK_FREQ_MHZ*1000000, exe_time);

        /* Golden Verion */
        matmul(a, b, g, width, height);

        /*Checksum */
        for (int i=0; i<width; i++) {
            for (int  j=0; j<height; j++) {
                // printf("%d,", c[i*width+j]);
                if(g[i*width+j] != c[i*width+j] ) {
                    printf("Checksum Error (iteration %d)!\n", iter);
                    exit(-1);
                }
            }
        }

        for (int i=0; i<width; i++)
           for (int j=0; j<height; j++){
               c[i*width+j] = 0;
               g[i*width+j] = 0;
           }
    }

    for(int iter=0; iter < 4; ++iter)
    {
        printf("MatMul no DMA! Width %d Height %d, a %x, b %x, c %x\n", (int) width, (int) height, (int) a, (int) b, (int) c);
        clock_gettime(CLOCK_REALTIME,&start);
        #pragma omp target map(to: a[0:width*height], b[0:width*height]) map(from: c[0:width*height])
        {
            matmul(a, b, c, width, height);
        }
        clock_gettime(CLOCK_REALTIME,&stop);        
        start_ns = ((double)(start.tv_sec))*1000000000 + (double)(start.tv_nsec);
        stop_ns  = ((double)(stop.tv_sec))*1000000000  + (double)(stop.tv_nsec);
        exe_time = (stop_ns - start_ns)/1000000000;
        printf("Exec Time [host cycles] = %.0f (%f s)\n", exe_time*ARM_CLK_FREQ_MHZ*1000000, exe_time);

        /* Golden Verion */
        matmul(a, b, g, width, height);

        /*Checksum */
        for (int i=0; i<width; i++){
            for (int  j=0; j<height; j++){
                // printf("%d,", c[i*width+j]);
                if(g[i*width+j] != c[i*width+j] ) {
                    printf("Checksum Error (iteration no-DMA %d)!\n", iter);
                    exit(-1);
                }
            }
        }

        /*Checksum */
        for (int i=0; i<width; i++) {
            for (int  j=0; j<height; j++) {
                // printf("%d,", c[i*width+j]);
                if(g[i*width+j] != c[i*width+j] ) {
                    printf("Checksum Error!\n");
                    exit(-1);
                }
            }
        }

        for (int i=0; i<width; i++)
           for (int j=0; j<height; j++){
               c[i*width+j] = 0;
               g[i*width+j] = 0;
           }
    }

    free(a);
    free(b);
    free(c);
    free(g);
    return 0;
}
