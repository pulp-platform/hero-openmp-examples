// #include <stdlib.h>
// #include <stdio.h>
// #include <stdint.h>
// #include <omp.h>
// #include <time.h>   // for time measurements
// #include <pulp-api.h>

// struct timespec start, stop;
// double start_ns, stop_ns, exe_time;

#define N 32
#define TYPE float

void vec_mult (TYPE *p, TYPE *v1, TYPE *v2)
{
	#pragma omp target map(to: v1[0:N], v2[0:N]) map(from: p[0:N])
	{
		#pragma omp parallel for
		for (int i = 0; i < N; i++)
			p[i] = v1[i] * v2[i];
	}
}

// void vec_mult2 ()
// {
// 	TYPE p[N], v1[N], v2[N];

// 	#pragma omp target map(to: v1[0:N], v2[0:N]) map(from: p[0:N])
// 	{	

// 		int i = 0;
// 		// #pragma omp parallel for
// 		for (int i = 0; i < N; i++)
// 			p[i] = v1[i] * v2[i];
// 	}
// }

int main(int argc, char *argv[])
{    
	TYPE p[N], v1[N], v2[N];
	vec_mult(p,v1,v2);
	// vec_mult2();
	return 0;
}

