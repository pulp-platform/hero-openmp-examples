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

  bench_start("PULP"); // {{{
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

  // Free memory. {{{
  free(a);
  free(b);
  free(c);
  free(d);
  // }}}

  return 0;
}
