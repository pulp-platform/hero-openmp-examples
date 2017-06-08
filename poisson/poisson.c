# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

# define NX 161
# define NY 161

int main ( int argc, char *argv[] );
float r8mat_rms ( int nx, int ny, float a[] );
void rhs ( int nx, int ny, float f[] );
#pragma omp declare target
void sweep ( int nx, int ny, float dx, float dy, float f[], 
  int itold, int itnew, float u[], float unew[] )

/******************************************************************************/
/*
  Purpose:

   SWEEP carries out one step of the Jacobi iteration.

  Discussion:

    Assuming DX = DY, we can approximate

      - ( d/dx d/dx + d/dy d/dy ) U(X,Y) 

    by

      ( U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1) - 4*U(i,j) ) / dx / dy

    The discretization employed below will not be correct in the general
    case where DX and DY are not equal.  It's only a little more complicated
    to allow DX and DY to be different, but we're not going to worry about 
    that right now.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    14 December 2011

  Author:

    John Burkardt

  Parameters:

    Input, int NX, NY, the X and Y grid dimensions.

    Input, float DX, DY, the spacing between grid points.

    Input, float F[], the right hand side data.

    Input, int ITOLD, the iteration index on input.

    Input, int ITNEW, the desired iteration index
    on output.

    Input, float U[, the solution estimate on 
    iteration ITNEW-1.

    Input/output, float UNEW[, on input, the solution 
    estimate on iteration ITOLD.  On output, the solution estimate on 
    iteration ITNEW.
*/
{
  int i;
  int it;
  int j;

# pragma omp parallel shared ( dx, dy, f, itnew, itold, nx, ny, u, unew ) private ( i, it, j )
  for ( it = itold + 1; it <= itnew; it++ )
  {
/*
  Save the current estimate.
*/
# pragma omp for
    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        u[i+j*nx] = unew[i+j*nx];
      }
    }
/*
  Compute a new estimate.
*/
# pragma omp for
    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        if ( i == 0 || j == 0 || i == nx - 1 || j == ny - 1 )
        {
          unew[i+j*nx] = f[i+j*nx];
        }
        else
        { 
          unew[i+j*nx] = 0.25 * ( 
            u[(i-1)+j*nx] + u[i+(j+1)*nx] + u[i+(j-1)*nx] + u[i+1+j*nx] + f[i+j*nx] * dx * dy );
        }
      }
    }

  }
  return;
}
#pragma omp end declare target
void timestamp ( void );
float u_exact ( float x, float y );
float uxxyy_exact ( float x, float y );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for POISSON_OPENMP.

  Discussion:

    POISSON_OPENMP is a program for solving the Poisson problem.

    This program uses OpenMP for parallel execution.

    The Poisson equation

      - DEL^2 U(X,Y) = F(X,Y)

    is solved on the unit square [0,1] x [0,1] using a grid of NX by
    NX evenly spaced points.  The first and last points in each direction
    are boundary points.

    The boundary conditions and F are set so that the exact solution is

      U(x,y) = sin ( pi * x * y )

    so that

      - DEL^2 U(x,y) = pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )

    The Jacobi iteration is repeatedly applied until convergence is detected.

    For convenience in writing the discretized equations, we assume that NX = NY.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    14 December 2011

  Author:

    John Burkardt
*/
{
  int converged;
  float diff;
  float dx;
  float dy;
  float error;
  float f[NX*NY];
  int i;
  int id;
  int itnew;
  int itold;
  int j;
  int jt;
  int jt_max = 20;
  int nx = NX;
  int ny = NY;
  float tolerance = 0.000001;
  float u[NX*NY];
  float u_norm;
  float udiff[NX*NY];
  float uexact[NX*NY];
  float unew[NX*NY];
  float unew_norm;
  float wtime;
  float x;
  float y;

  dx = 1.0 / ( float ) ( nx - 1 );
  dy = 1.0 / ( float ) ( ny - 1 );
/*
  Print a message.
*/
  timestamp ( );
  printf ( "\n" );
  printf ( "POISSON_OPENMP:\n" );
  printf ( "  C version\n" );
  printf ( "  A program for solving the Poisson equation.\n" );
  printf ( "\n" );
  printf ( "  Use OpenMP for parallel execution.\n" );
  printf ( "  The number of processors is %d\n", omp_get_num_procs ( ) );
# pragma omp parallel
{
  id = omp_get_thread_num ( );
  if ( id == 0 )
  {
    printf ( "  The maximum number of threads is %d\n", omp_get_num_threads ( ) ); 
  }
}
  printf ( "\n" );
  printf ( "  -DEL^2 U = F(X,Y)\n" );
  printf ( "\n" );
  printf ( "  on the rectangle 0 <= X <= 1, 0 <= Y <= 1.\n" );
  printf ( "\n" );
  printf ( "  F(X,Y) = pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )\n" );
  printf ( "\n" );
  printf ( "  The number of interior X grid points is %d\n", nx );
  printf ( "  The number of interior Y grid points is %d\n", ny );
  printf ( "  The X grid spacing is %f\n", dx );
  printf ( "  The Y grid spacing is %f\n", dy );
/*
  Set the right hand side array F.
*/
  rhs ( nx, ny, f );
/*
  Set the initial solution estimate UNEW.
  We are "allowed" to pick up the boundary conditions exactly.
*/
  for ( j = 0; j < ny; j++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
      {
        unew[i+j*nx] = f[i+j*nx];
      }
      else
      {
        unew[i+j*nx] = 0.0;
      }
    }
  }
  unew_norm = r8mat_rms ( nx, ny, unew );
/*
  Set up the exact solution UEXACT.
*/
  for ( j = 0; j < ny; j++ )
  {
    y = ( float ) ( j ) / ( float ) ( ny - 1 );
    for ( i = 0; i < nx; i++ )
    {
      x = ( float ) ( i ) / ( float ) ( nx - 1 );
      uexact[i+j*nx] = u_exact ( x, y );
    }
  }
  u_norm = r8mat_rms ( nx, ny, uexact );
  printf ( "  RMS of exact solution = %g\n", u_norm );
/*
  Do the iteration.
*/
  converged = 0;

  printf ( "\n" );
  printf ( "  Step    ||Unew||     ||Unew-U||     ||Unew-Exact||\n" );
  printf ( "\n" );

  for ( j = 0; j < ny; j++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      udiff[i+j*nx] = unew[i+j*nx] - uexact[i+j*nx];
    }
  }
  error = r8mat_rms ( nx, ny, udiff );
  printf ( "  %4d  %14g                  %14g\n", 0, unew_norm, error );

  wtime = omp_get_wtime ( );

  itnew = 0;

  for ( ; ; )
  {
    itold = itnew;
    itnew = itold + 500;
/*
  SWEEP carries out 500 Jacobi steps in parallel before we come
  back to check for convergence.
*/
#   pragma omp target map(to: itold, itnew, f[0:NX*NY], u[0:NX*NY]) map(tofrom: unew[0:NX*NY])
    sweep ( NX, NY, dx, dy, f, itold, itnew, u, unew );

/*
  Check for convergence.
*/
    u_norm = unew_norm;
    unew_norm = r8mat_rms ( nx, ny, unew );

    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        udiff[i+j*nx] = unew[i+j*nx] - u[i+j*nx];
      }
    }
    diff = r8mat_rms ( nx, ny, udiff );

    for ( j = 0; j < ny; j++ )
    {
      for ( i = 0; i < nx; i++ )
      {
        udiff[i+j*nx] = unew[i+j*nx] - uexact[i+j*nx];
      }
    }
    error = r8mat_rms ( nx, ny, udiff );

    printf ( "  %4d  %14g  %14g  %14g\n", itnew, unew_norm, diff, error );

    if ( diff <= tolerance )
    {
      converged = 1;
      break;
    }

  }

  if ( converged )
  {
    printf ( "  The iteration has converged.\n" );
  }
  else
  {
    printf ( "  The iteration has NOT converged.\n" );
  }

  wtime = omp_get_wtime ( ) - wtime;
  printf ( "\n" );
  printf ( "  Elapsed seconds = %g\n", wtime );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "POISSON_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

float r8mat_rms ( int nx, int ny, float a[] )

/******************************************************************************/
/*
  Purpose:

    R8MAT_RMS returns the RMS norm of a vector stored as a matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 March 2003

  Author:

    John Burkardt

  Parameters:

    Input, int NX, NY, the number of rows and columns in A.

    Input, float A[], the vector.

    Output, float R8MAT_RMS, the root mean square of the entries of A.
*/
{
  int i;
  int j;
  float v;

  v = 0.0;

  for ( j = 0; j < ny; j++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      v = v + a[i+j*nx] * a[i+j*nx];
    }
  }
  v = sqrt ( v / ( float ) ( nx * ny )  );

  return v;
}
/******************************************************************************/

void rhs ( int nx, int ny, float f[] )

/******************************************************************************/
/*
  Purpose:

    RHS initializes the right hand side "vector".

  Discussion:

    It is convenient for us to set up RHS as a 2D array.  However, each
    entry of RHS is really the right hand side of a linear system of the
    form

      A * U = F

    In cases where U(I,J) is a boundary value, then the equation is simply

      U(I,J) = F(i,j)

    and F(I,J) holds the boundary data.

    Otherwise, the equation has the form

      (1/DX^2) * ( U(I+1,J)+U(I-1,J)+U(I,J-1)+U(I,J+1)-4*U(I,J) ) = F(I,J)

    where DX is the spacing and F(I,J) is the value at X(I), Y(J) of

      pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    28 October 2011

  Author:

    John Burkardt

  Parameters:

    Input, int NX, NY, the X and Y grid dimensions.

    Output, float F[NX][NY], the initialized right hand side data.
*/
{
  float fnorm;
  int i;
  int j;
  float x;
  float y;
/*
  The "boundary" entries of F store the boundary values of the solution.
  The "interior" entries of F store the right hand sides of the Poisson equation.
*/
  for ( j = 0; j < ny; j++ )
  {
    y = ( float ) ( j ) / ( float ) ( ny - 1 );
    for ( i = 0; i < nx; i++ )
    {
      x = ( float ) ( i ) / ( float ) ( nx - 1 );
      if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
      {
        f[i+j*nx] = u_exact ( x, y );
      }
      else
      {
        f[i+j*nx] = - uxxyy_exact ( x, y );
      }
    }
  }

  fnorm = r8mat_rms ( nx, ny, f );

  printf ( "  RMS of F = %g\n", fnorm );

  return;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
/******************************************************************************/

float u_exact ( float x, float y )

/******************************************************************************/
/*
  Purpose:

    U_EXACT evaluates the exact solution.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    25 October 2011

  Author:

    John Burkardt

  Parameters:

    Input, float X, Y, the coordinates of a point.

    Output, float U_EXACT, the value of the exact solution 
    at (X,Y).
*/
{
  float pi = 3.141592653589793f;
  float value;

  value = sin ( pi * x * y );

  return value;
}
/******************************************************************************/

float uxxyy_exact ( float x, float y )

/******************************************************************************/
/*
  Purpose:

    UXXYY_EXACT evaluates ( d/dx d/dx + d/dy d/dy ) of the exact solution.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    25 October 2011

  Author:

    John Burkardt

  Parameters:

    Input, float X, Y, the coordinates of a point.

    Output, float UXXYY_EXACT, the value of 
    ( d/dx d/dx + d/dy d/dy ) of the exact solution at (X,Y).
*/
{
  float pi = 3.141592653589793f;
  float value;

  value = - pi * pi * ( x * x + y * y ) * sin ( pi * x * y );

  return value;
}
# undef NX
# undef NY
