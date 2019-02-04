/*
 * HERO Matrix-Matrix Multiplication with Double Buffering Example Application
 *
 * Copyright 2018 ETH Zurich, University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>        // for error codes
#include "bench.h"


#pragma omp declare target
#include <hero-target.h>
#define SIZE 128

// int add4(int8_t * __restrict__ a, int8_t * __restrict__ b, int32_t * __restrict__ c)
// {
//   int32_t S = 0;
//   hero_v4s *_va = ( hero_v4s *) a;
//   hero_v4s *_vb = ( hero_v4s *) b;
//   hero_v4s *_vc = ( hero_v4s *) c;

//   for (unsigned i=0; i<SIZE; i+=4) {
//     hero_v4s VA = _va[i];
//     hero_v4s VB = _vb[i];
//     asm volatile("": : :"memory");
//     hero_v4s VC = _vc[i];
//     VC = __builtin_hero_add4(VA, VB);
//   }

//   return 0;
// }

#pragma omp end declare target

int main(int argc, char *argv[])
{
  // Allocate memory
  int8_t * __restrict__ a = (int8_t *)malloc(sizeof(int8_t)*SIZE);
  int8_t * __restrict__ b = (int8_t *)malloc(sizeof(int8_t)*SIZE);
  int8_t * __restrict__ c = (int8_t *)malloc(sizeof(int8_t)*SIZE);
  memset((void *)c, 0, (size_t)(SIZE));

  
  #pragma omp target device(1) map(to: a[0:SIZE], b[0:SIZE]) \
    map(from: c[0:SIZE])
  {
    for (unsigned i=0; i<SIZE; i+=4) {
      c[i] = __builtin_hero_add4(a[i], b[i]);
    }
  }

  // free memory
  free(a);
  free(b);
  free(c);

  return 0;
}
