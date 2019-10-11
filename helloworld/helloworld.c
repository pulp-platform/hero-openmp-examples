/*
 * HERO HelloWorld Example Application
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
#include <stdint.h>
#include <time.h>
#include <hero-target.h>

struct timespec start, stop;
double start_ns, stop_ns, exe_time;

// #define ARCHI_ACCEL_ADDR_BASE (ARCHI_CLUSTER_ADDR + ARCHI_CLUSTER_PERIPHERALS_OFFSET + 0x00001000)
#define ARCHI_ACCEL_ADDR_BASE (0x1B000000 + 0x00200000 + 0x00001000)

#define pulp_write32(add, val_) (*(volatile unsigned int *)(long)(add) = val_)
#define pulp_read32(add) (*(volatile unsigned int *)(long)(add))


#define OPERAND_A_ADDR 0x00
#define OPERAND_B_ADDR 0x04
#define RESULT_ADDR    0x08
#define START          0x0C
#define STATUS         0x10

#pragma omp declare target
void helloworld ()
{
	#pragma omp parallel
	printf("Hello World, I am thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());

    printf("Hello !\n");

    // Allocate on l1
    unsigned *a = hero_l1malloc(sizeof(unsigned));
    unsigned *b = hero_l1malloc(sizeof(unsigned));
    unsigned *c = hero_l1malloc(sizeof(unsigned));
    if (!a || !b || !c) {
        printf("Allocation failed!\n");
    }

    *a = 0xDEAFBEEF;
    *b = 0xABBA;
    *c = 0;

    printf("Addresses are: %8x %8x %8x\n", (unsigned) a, (unsigned) b, (unsigned) c);
    printf("Values are: %8x %8x %8x\n", *a, *b, *c);

    // Write to accelerator
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + OPERAND_A_ADDR, (unsigned) a);
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + OPERAND_B_ADDR, (unsigned) b);
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + RESULT_ADDR, (unsigned) c);
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + START, 1);

    while (!pulp_read32(ARCHI_ACCEL_ADDR_BASE + STATUS));

    printf("Values are: %8x %8x %8x\n", *a, *b, *c);

    printf("Start is: %x\n", pulp_read32(ARCHI_ACCEL_ADDR_BASE + START));
    printf("Status is: %x\n", pulp_read32(ARCHI_ACCEL_ADDR_BASE + STATUS));

    *a = 0xDEAFBEEF;
    *b = 0xABBA;
    *c = 0;

    printf("Addresses are: %8x %8x %8x\n", (unsigned) a, (unsigned) b, (unsigned) c);
    printf("Values are: %8x %8x %8x\n", *a, *b, *c);

    // Write to accelerator
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + OPERAND_A_ADDR, (unsigned) a);
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + OPERAND_B_ADDR, (unsigned) b);
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + RESULT_ADDR, (unsigned) c);
    pulp_write32(ARCHI_ACCEL_ADDR_BASE + START, 1);

    while (!pulp_read32(ARCHI_ACCEL_ADDR_BASE + STATUS));

    printf("Values are: %8x %8x %8x\n", *a, *b, *c);

    printf("Start is: %x\n", pulp_read32(ARCHI_ACCEL_ADDR_BASE + START));
    printf("Status is: %x\n", pulp_read32(ARCHI_ACCEL_ADDR_BASE + STATUS));

    volatile unsigned a = 0;
    while (a < 10000000) {
        a++;
    }
}
#pragma omp end declare target

int main(int argc, char *argv[])
{
	omp_set_default_device(BIGPULP_MEMCPY);

	#pragma omp target
	helloworld();

	return 0;
}
