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
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      uint32_t sum = 0;
      for (unsigned k = 0; k < n; ++k)
        sum += a[i*n+k] * b[k*n+j];
      d[i*n+j] = sum;
    }
  }
  bench_stop(); // }}}

  // Free memory. {{{
  free(a);
  free(b);
  free(c);
  free(d);
  // }}}

  return 0;
}
