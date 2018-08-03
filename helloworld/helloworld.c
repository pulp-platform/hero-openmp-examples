#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
//#include <pulp-api.h>

struct timespec start, stop;
double start_ns, stop_ns, exe_time;

#pragma omp declare target
void helloworld ()
{
	#pragma omp parallel
	printf("Hello World, I am thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
}
#pragma omp end declare target

int main(int argc, char *argv[])
{    
	#pragma omp target
	helloworld();

	helloworld();
	return 0;
}

