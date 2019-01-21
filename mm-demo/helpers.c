#include "helpers.h"
#include <errno.h>  // -ENOMEM
#include <stdio.h>  // printf()
#include <stdlib.h> // exit()
#include <string.h> // memset()

int compare_matrices(const uint32_t* const a, const uint32_t* const b,
    const size_t n)
{
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      if (a[i*n+j] != b[i*n+j]) {
        printf("ERROR: Result mismatch in row %u, column %u!\n", j, i);
        return -1;
      }
    }
  }
  return 0;
}

void init_matrices(uint32_t* const a, uint32_t* const b, uint32_t* const c,
    uint32_t* const d, const size_t n)
{
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      a[i*n+j] = i * n + j;
      b[i*n+j] = (i == j) ? 2 : 0;
    }
  }
  memset((void*)c, 0, n*n);
  memset((void*)d, 0, n*n);
}

void alloc_matrices(uint32_t** a, uint32_t** b, uint32_t** c, uint32_t** d,
    const size_t n)
{
  *a = (uint32_t*)malloc(sizeof(uint32_t)*n*n);
  *b = (uint32_t*)malloc(sizeof(uint32_t)*n*n);
  *c = (uint32_t*)malloc(sizeof(uint32_t)*n*n);
  *d = (uint32_t*)malloc(sizeof(uint32_t)*n*n);
  if (!*a || !*b || !*c || !*d) {
    printf("ERROR: alloc_matrices() failed!\n");
    exit(-ENOMEM);
  }
}
