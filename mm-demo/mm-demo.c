#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>        // for error codes
#include "bench.h"
#include <hero-target.h>
#include "helpers.h"

int main(int argc, char *argv[])
{
  // Initialize application memory on host {{{
  const size_t n = 128;

  uint32_t *a, *b, *c, *d;
  alloc_matrices(&a, &b, &c, &d, n);
  init_matrices(a, b, c, d, n);
  // }}}

  // Execute on host

  bench_start("Host"); // {{{
  #pragma omp parallel for \
    firstprivate(a, b, d, n) \
    collapse(2)
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      uint32_t sum = 0;
      for (unsigned k = 0; k < n; ++k)
        sum += a[i*n+k] * b[k*n+j];
      d[i*n+j] = sum;
    }
  }
  bench_stop(); // }}}

  // Execute on PULP

  // Make sure PULP is ready (not necessary but speeds up the first target). {{{
  unsigned tmp_1 = 1, tmp_2 = 2;
  #pragma omp target device(BIGPULP_MEMCPY) \
    map(to: tmp_1) \
    map(from: tmp_2)
  {
    tmp_2 = tmp_1;
  }
  tmp_1 = tmp_2; // }}}

  bench_start("PULP: Single-threaded, no DMA"); // {{{
  #pragma omp target device(BIGPULP_MEMCPY) \
    map(to: a[0:n*n], b[0:n*n], n) \
    map(from: c[0:n*n])
  {
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = 0; j < n; ++j) {
        uint32_t sum = 0;
        for (unsigned k = 0; k < n; ++k)
          sum += a[i*n+k] * b[k*n+j];
        c[i*n+j] = sum;
      }
    }
  }
  bench_stop();
  compare_matrices(c, d, n);
  memset((void*)c, 0, n*n); // }}}

  bench_start("PULP: Parallel, no DMA"); // {{{
  #pragma omp target device(BIGPULP_MEMCPY) \
    map(to: a[0:n*n], b[0:n*n], n) \
    map(from: c[0:n*n])
  {
    #pragma omp parallel for \
      firstprivate(a, b, c, n) \
      collapse(2)
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = 0; j < n; ++j) {
        uint32_t sum = 0;
        for (unsigned k = 0; k < n; ++k)
          sum += a[i*n+k] * b[k*n+j];
        c[i*n+j] = sum;
      }
    }
  }
  bench_stop();
  compare_matrices(c, d, n);
  memset((void*)c, 0, n*n); // }}}

  bench_start("PULP: Parallel, DMA"); // {{{
  #pragma omp target device(BIGPULP_MEMCPY) \
    map(to: a[0:n*n], b[0:n*n], n) \
    map(from: c[0:n*n])
  {
    uint32_t* const a_local = (uint32_t*)hero_l1malloc(n*n*sizeof(uint32_t));
    uint32_t* const b_local = (uint32_t*)hero_l1malloc(n*n*sizeof(uint32_t));
    uint32_t* const c_local = (uint32_t*)hero_l1malloc(n*n*sizeof(uint32_t));
    if (!a_local || !b_local || !c_local) {
      printf("ERROR: Memory allocation failed!\n");
    }

    hero_dma_job_t dma0 =
        hero_dma_memcpy_async(a_local, a, n*n*sizeof(uint32_t));
    hero_dma_job_t dma1 =
        hero_dma_memcpy_async(b_local, b, n*n*sizeof(uint32_t));
    hero_dma_wait(dma0);
    hero_dma_wait(dma1);

    #pragma omp parallel for \
      firstprivate(a_local, b_local, c_local, n) \
      collapse(2)
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = 0; j < n; ++j) {
        uint32_t sum = 0;
        for (unsigned k = 0; k < n; ++k)
          sum += a_local[i*n+k] * b_local[k*n+j];
        c_local[i*n+j] = sum;
      }
    }

    hero_dma_memcpy(c, c_local, n*n*sizeof(uint32_t));

    hero_l1free(a_local);
    hero_l1free(b_local);
    hero_l1free(c_local);
  }
  bench_stop();
  compare_matrices(c, d, n);
  memset((void*)c, 0, n*n); // }}}

  // Free memory. {{{
  free(a);
  free(b);
  free(c);
  free(d);
  // }}}

  return 0;
}
