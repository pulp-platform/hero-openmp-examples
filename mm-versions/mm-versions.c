#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
//#include <omp.h>
#include <time.h>         // for time measurements
#include <unistd.h>
#include <errno.h>        // for error codes
#include <hero-target.h>

void get_exe_time(struct timespec * start, struct timespec * stop, double * exe_time)
{
  double start_ns, stop_ns;

  start_ns = ((double)(start->tv_sec))*1000000000 + (double)(start->tv_nsec);
  stop_ns  = ((double)(stop->tv_sec))*1000000000  + (double)(stop->tv_nsec);

  *exe_time = (stop_ns - start_ns)/1000000000;
}

void compare_matrices(uint32_t* a, uint32_t* b, int width, int height)
{
  for (int i=0; i<width; i++) {
    for (int j=0; j<height; j++) {
      if(a[i*width+j] != b[i*width+j] ) {
        printf("ERROR: Result mismatch in Row %i, Column %i!\n", j, i);
        exit(-1);
      }
    }
  }
}

int main(int argc, char *argv[])
{
  printf("HERO matrix multiplication started.\n");

  // Global variables
  struct timespec start, stop;
  double exe_time;
  unsigned host_clk_freq_mhz = 0;

  int width  = 128;
  int height = width;

  uint32_t *a;
  uint32_t *b;
  uint32_t *c;
  uint32_t *d;

  if( argc > 1 ) {
    width  = atoi(argv[1]);
    height = atoi(argv[1]);
  }

  // Allocate memory
  a = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
  b = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
  c = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
  d = (uint32_t *)malloc(sizeof(uint32_t)*width*height);
  if ( (a == NULL) || (b == NULL) || (c == NULL) || (d == NULL) ) {
    printf("ERROR: malloc() failed!\n");
    return -ENOMEM;
  }
  printf("width = %d, height = %d, a @ %p, b @ %p, c @ %p\n", width, height, a, b, c);

  // Init matrices
  for (int i=0; i<width; i++) {
    for (int j=0; j<height; j++) {
      a[i*width+j] = i*width+j;
      b[i*width+j] = i == j ? 2 : 0;
    }
  }
  memset((void *)c, 0, (size_t)(width*height));
  memset((void *)d, 0, (size_t)(width*height));

  // Get host clock frequency
  if( access("/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq", F_OK ) != -1 ) {
    char host_clk_freq_khz_string[20];
    FILE *fp;

    if((fp = fopen("/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq", "r")) == NULL)
      printf("ERROR: Could not open sysfs.\n");
    else if ( fgets(host_clk_freq_khz_string, 20, fp) != NULL)
      host_clk_freq_mhz = (strtoul(host_clk_freq_khz_string, NULL, 10)+1)/1000;

    fclose(fp);
  }

  /*
   * Make sure PULP is ready - speeds up the first target
   *
   * Actually, we should not use both devices at the same time as it is not safe. OpenMP will load
   * or boot both of them. But in reality only one accelerator is there.
   */
  unsigned tmp_1 = 1;
  unsigned tmp_2 = 2;
  #pragma omp target device(1) map(to: tmp_1) map(from: tmp_2)
  {
    tmp_2 = tmp_1;
  }
  tmp_1 = tmp_2;

  /*
   * Execute on host
   */
  printf("\nHost Execution\n");

  clock_gettime(CLOCK_REALTIME,&start);

  #pragma omp parallel firstprivate(a, b, d, width, height)
  {
    for (int i=0; i<width; i++) {
      for (int j=0; j<height; j++) {
        int sum = 0;
        for (int k=0; k<width; k++)
          sum = sum + a[i*width+k] * b[k*width+j];
        d[i*width+j] = sum;
      }
    }
  }

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time [host cycles] = %.0f (%f s)\n", exe_time*host_clk_freq_mhz*1000000, exe_time);

  /*
   * Excute on PULP - Single-threaded, copy-based, no DMA
   */
  printf("\nPULP Execution: Single-threaded, copy-based, no DMA\n");

  clock_gettime(CLOCK_REALTIME,&start);

  #pragma omp target device(1) map(to: a[0:width*height], b[0:width*height], width, height) map(from: c[0:width*height])
  {
    for (int i=0; i<width; i++) {
      for (int j=0; j<height; j++) {
        int sum = 0;
        for (int k=0; k<width; k++)
          sum = sum + a[i*width+k] * b[k*width+j];
        c[i*width+j] = sum;
      }
    }
  }

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time, entire offload [host cycles] = %.0f (%f s)\n", exe_time*host_clk_freq_mhz*1000000, exe_time);

  compare_matrices(c, d, width, height);
  memset((void *)c, 0, (size_t)(width*height));

  /*
   * Excute on PULP - Parallel, copy-based, no DMA
   */
  printf("\nPULP Execution: Parallel, copy-based, no DMA\n");

  clock_gettime(CLOCK_REALTIME,&start);

  #pragma omp target device(1) map(to: a[0:width*height], b[0:width*height], width, height) map(from: c[0:width*height])
  {

    #pragma omp parallel for collapse(2) firstprivate(a, b, c, width, height)
      for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
          int sum = 0;
          for (int k=0; k<width; k++)
            sum = sum + a[i*width+k] * b[k*width+j];
          c[i*width+j] = sum;
        }
      }
  }

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time, entire offload [host cycles] = %.0f (%f s)\n", exe_time*host_clk_freq_mhz*1000000, exe_time);

  compare_matrices(c, d, width, height);
  memset((void *)c, 0, (size_t)(width*height));

  /*
   * Excute on PULP - Parallel, copy-based, DMA
   */
  printf("\nPULP Execution: Parallel, copy-based, DMA\n");

  clock_gettime(CLOCK_REALTIME,&start);

  #pragma omp target device(1) map(to: a[0:width*height], b[0:width*height], width, height) map(from: c[0:width*height])
  {
    uint32_t * a_local = (uint32_t *)hero_l1malloc(width*height*sizeof(uint32_t));
    uint32_t * b_local = (uint32_t *)hero_l1malloc(width*height*sizeof(uint32_t));
    uint32_t * c_local = (uint32_t *)hero_l1malloc(width*height*sizeof(uint32_t));
    if ( (a_local == NULL) || (b_local == NULL) || (c_local == NULL) ) {
      printf("ERROR: Memory allocation failed!");
    }

    hero_dma_job_t dma0 = hero_dma_memcpy_async(a_local, a, width*height*sizeof(uint32_t));
    hero_dma_job_t dma1 = hero_dma_memcpy_async(b_local, b, width*height*sizeof(uint32_t));
    hero_dma_wait(dma0);
    hero_dma_wait(dma1);

    #pragma omp parallel for collapse(2) firstprivate(a_local, b_local, c_local, width, height) //num_threads(7)
      for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
          int sum = 0;
          for (int k=0; k<width; k++)
            sum = sum + a_local[i*width+k] * b_local[k*width+j];
          c_local[i*width+j] = sum;
        }
      }

    hero_dma_memcpy(c, c_local, width*height*sizeof(uint32_t));

    hero_l1free(a_local);
    hero_l1free(b_local);
    hero_l1free(c_local);
  }

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time, entire offload [host cycles] = %.0f (%f s)\n", exe_time*host_clk_freq_mhz*1000000, exe_time);

  compare_matrices(c, d, width, height);
  memset((void *)c, 0, (size_t)(width*height));

  /*
   * Make sure PULP is ready - speeds up the first target
   *
   * Actually, we should not use both devices at the same time as it is not safe. OpenMP will load
   * or boot both of them. But in reality only one accelerator is there.
   */
  #pragma omp target device(0) map(to: tmp_1) map(from: tmp_2)
  {
    hero_trywrite(&tmp_2, hero_tryread(&tmp_1));
  }
  tmp_1 = tmp_2;

  /*
   * Excute on PULP - Parallel, SVM, DMA
   */
  printf("\nPULP Execution: Parallel, SVM, DMA\n");

  clock_gettime(CLOCK_REALTIME,&start);

  #pragma omp target device(0) map(to: a[0:width*height], b[0:width*height], width, height) map(from: c[0:width*height])
  {
    unsigned sync;

    #pragma omp parallel default(none) shared(a, b, c, width, height, sync) num_threads(2)
    {
      // Spawn the miss-handler thread
      if (omp_get_thread_num() == 0) {
        const int core_id = hero_rt_core_id();
        //#if RT_LOG_INFOS(LOG_LVL_VMM)
        //  rt_info("Starting miss handling on core %d.\n", core_id);
        //#endif
        int ret;
        do {
          ret = hero_handle_rab_misses();
          if (!(ret == 0 || ret == -ENOENT)) {

            //#if RT_LOG_ERRORS(LOG_LVL_VMM)
            //  rt_error("RAB miss handling returned nonzero error: %d!\n", -ret);
            //#endif
          }
        } while (sync == 0);
      } // omp_get_thread_num() == 0

      // Worker threads...
      else {
        int width_local, height_local;
        width_local  = hero_tryread_prefetch((unsigned int *)&width);
        height_local = hero_tryread_prefetch((unsigned int *)&height);
        width_local  = hero_tryread((unsigned int *)&width);
        height_local = hero_tryread((unsigned int *)&height);

        uint32_t * a_local = (uint32_t *)hero_l1malloc(width_local*height_local*sizeof(uint32_t));
        uint32_t * b_local = (uint32_t *)hero_l1malloc(width_local*height_local*sizeof(uint32_t));
        uint32_t * c_local = (uint32_t *)hero_l1malloc(width_local*height_local*sizeof(uint32_t));
        if ( (a_local == NULL) || (b_local == NULL) || (c_local == NULL) ) {
          printf("ERROR: Memory allocation failed!");
        }

        hero_dma_job_t dma0 = hero_dma_memcpy_async(a_local, a, width_local*height_local*sizeof(uint32_t));
        hero_dma_job_t dma1 = hero_dma_memcpy_async(b_local, b, width_local*height_local*sizeof(uint32_t));
        hero_dma_wait(dma0);
        hero_dma_wait(dma1);

        #pragma omp parallel for collapse(2) firstprivate(a_local, b_local, c_local, width_local, height_local)
        for (int i=0; i<width_local; i++) {
          for (int j=0; j<height_local; j++) {
            int sum = 0;
            for (int k=0; k<width_local; k++)
              sum = sum + a_local[i*width_local+k] * b_local[k*width_local+j];
            c_local[i*width_local+j] = sum;
          }
        }

        hero_dma_memcpy(c, c_local, width_local*height_local*sizeof(uint32_t));

        hero_l1free(a_local);
        hero_l1free(b_local);
        hero_l1free(c_local);

        // tell the miss-handler thread that we are done
        sync = 1;
      } // else ... omp_get_thread_num() == 0
    } // parallel
  } // target

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time, entire offload [host cycles] = %.0f (%f s)\n",
    exe_time*host_clk_freq_mhz*1000000, exe_time);

  compare_matrices(c, d, width, height);
  memset((void *)c, 0, (size_t)(width*height));


  /*
   * Excute on PULP - Parallel, copy-based, no DMA
   */
  printf("\nPULP Execution: Intrinsics\n");

  clock_gettime(CLOCK_REALTIME,&start);

#if 0
  #pragma omp target device(1) map(to: a[0:width*height], b[0:width*height], width, height) map(from: c[0:width*height])
  {

    signed short a __attribute__((vector_size (4)));
    signed short b __attribute__((vector_size (4)));
    signed short c __attribute__((vector_size (4)));

    //hero_sdotsp2(a, b, c);

  }

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time, entire offload [host cycles] = %.0f (%f s)\n", exe_time*host_clk_freq_mhz*1000000, exe_time);

  /*
   * Excute on PULP - Parallel, SVM, double-buffered DMA
   */
  printf("PULP Execution: Parallel, SVM, double-buffered DMA\n");

  clock_gettime(CLOCK_REALTIME,&start);

  clock_gettime(CLOCK_REALTIME,&stop);
  get_exe_time(&start, &stop, &exe_time);
  printf("Execution time, entire offload [host cycles] = %.0f (%f s)\n", exe_time*host_clk_freq_mhz*1000000, exe_time);

  compare_matrices(c, d, width, height);
  memset((void *)c, 0, (size_t)(width*height));
#endif

  // free memory
  free(a);
  free(b);
  free(c);
  free(d);

  return 0;
}
