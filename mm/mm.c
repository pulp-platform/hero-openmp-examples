#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>   // for time measurements
#include <pulp-api.h>

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

int main ()
{    
    int width  = 32;
    int height = 32;
    int error  = 0;
    uint32_t a[width*height];
    uint32_t b[width*height];
    uint32_t c[width*height];
    uint32_t g[width*height];

    /* Init */    
    for (int i=0; i<width; i++){
        for (int j=0; j<height; j++){
            a[i*width+j] = i*width+j;
            b[i*width+j] = i == j ? 2 : 0;
            c[i*width+j] = 0;
            g[i*width+j] = 0;
        }
    }

    for(int i=0; i < 4; ++i)
    {
        error = 0;
        clock_gettime(CLOCK_REALTIME,&start);
        #pragma omp target map(to: a[0:width*height], b[0:width*height]) map(from: c[0:width*height])
        {
            uint8_t *local_space = (uint8_t *)pulp_l1malloc(3*width*height*sizeof(uint32_t));
            uint32_t *local_a = (uint32_t *) &local_space[0*width*height*sizeof(uint32_t)];
            uint32_t *local_b = (uint32_t *) &local_space[1*width*height*sizeof(uint32_t)];
            uint32_t *local_c = (uint32_t *) &local_space[2*width*height*sizeof(uint32_t)];

            
            // pulp_memcpy(local_a, a, width*height*sizeof(uint32_t));
            // pulp_memcpy(local_b, a, width*height*sizeof(uint32_t));

            printf("MatMul DMA! Width %d Height %d, a %x, b %x, c %x\n", (int) width, (int) height, (int) a, (int) b, (int) c);
            mat_print(c, width, height);
            // matmul(a, b, c, width, height);

            // pulp_memcpy(c, local_c, width*height*sizeof(uint32_t));
            pulp_l1free(local_space);
        }
        clock_gettime(CLOCK_REALTIME,&stop);        
        start_ns = ((double)(start.tv_sec))*1000000000 + (double)(start.tv_nsec);
        stop_ns  = ((double)(stop.tv_sec))*1000000000  + (double)(stop.tv_nsec);
        exe_time = (stop_ns - start_ns)/1000000000;
        printf("Exec Time [host cycles] = %.0f (%f s)\n", exe_time*ARM_CLK_FREQ_MHZ*1000000, exe_time);

        matmul(a, b, g, width, height);
        mat_print(g, width, height);

        /*Checksum */
        for (int i=0; i<width; i++) {
            for (int  j=0; j<height; j++) {
                // printf("%d,", c[i*width+j]);
                if(g[i*width+j] != c[i*width+j] ) {
                    error = 1;
                    printf("WRONG!\n");
                    break;
                }
            }
            if(error)
                break;
        }

        error = 0;
        for (int i=0; i<width; i++)
           for (int j=0; j<height; j++){
               c[i*width+j] = 0;
               g[i*width+j] = 0;
           }
    }

    for(int i=0; i < 4; ++i)
    {
        clock_gettime(CLOCK_REALTIME,&start);
        #pragma omp target map(to: a[0:width*height], b[0:width*height]) map(from: c[0:width*height])
        {
            printf("MatMul no DMA! Width %d Height %d, a %x, b %x, c %x\n", (int) width, (int) height, (int) a, (int) b, (int) c);
            matmul(a, b, c, width, height);
        }
        clock_gettime(CLOCK_REALTIME,&stop);        
        start_ns = ((double)(start.tv_sec))*1000000000 + (double)(start.tv_nsec);
        stop_ns  = ((double)(stop.tv_sec))*1000000000  + (double)(stop.tv_nsec);
        exe_time = (stop_ns - start_ns)/1000000000;
        printf("Exec Time [host cycles] = %.0f (%f s)\n", exe_time*ARM_CLK_FREQ_MHZ*1000000, exe_time);

        matmul(a, b, g, width, height);
        /*Checksum */
        for (int i=0; i<width; i++){
            for (int  j=0; j<height; j++){
                // printf("%d,", c[i*width+j]);
                if(g[i*width+j] != c[i*width+j] ) {
                    printf("WRONG!\n");
                    break;
                }
            }
        }

        /*Checksum */
        for (int i=0; i<width; i++) {
            for (int  j=0; j<height; j++) {
                // printf("%d,", c[i*width+j]);
                if(g[i*width+j] != c[i*width+j] ) {
                    error = 1;
                    printf("WRONG!\n");
                    break;
                }
            }
            if(error)
                break;
        }

        error = 0;
        for (int i=0; i<width; i++)
           for (int j=0; j<height; j++){
               c[i*width+j] = 0;
               g[i*width+j] = 0;
           }
    }

    return 0;
}
