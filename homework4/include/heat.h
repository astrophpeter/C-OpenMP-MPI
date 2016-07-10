#ifndef __homework_4_heat_h
#define __homework_4_heat_h

#include <mpi.h>


/*
  array_copy

  Copies the contents of one array to another. Note that this is "stronger" than
  just having ```toarray` point to the same location as `fromarray`.

  Parameters
  ----------
  toarray : double*
      Array to copy into.
  fromarray : double*
      Array to copy the contents from into `toarray`.
  length : size_t

  Returns
  -------

  None
      `toarray` is modified in-place.

*/
void array_copy(double* toarray, double* fromarray, size_t length);

/*
  heat_serial

  A serial version of solving the periodic heat equation on a uniform grid. Use
  for comparison with the MPI distributed version.

  The forward Euler method for numerical solving PDEs is unstable for large
  (even only moderately) values of `dt`. Make sure `dt` is small. Examples are
  given in the Python code that calls this function.

  Parameters
  ----------
  u : double*
      Input data array. Represents the heat / temperature across a 1d rod.
      Assumes that the data points are equally spaced in the space domain.
  Nx : size_t
      The length of `u`.
  dt : double
      The size of the time step to take with each iteration.
  Nt : size_t
      Number of time-steps to perform.
  dx : double 
      Spacial incrment used to compute the diffusion coefficient.

  Returns
  -------
  None
      The array `u` is modified in place.

*/
void heat_serial(double* u, double dx, size_t Nx, double dt, size_t Nt);
/*
   heat_parallel

   A paral version of solving the periodic heat eqaution on a uniform grid.


   Parameters
   ----------
   uk : double*
      Input data array. Represents the heat / temperature across a 1d rod.
      Assumes that the data points are equally spaced in the space domain.
   Nx : size_t
      The length of `uk' each process obtains.
   dt : double
      The size of the time step to take with each iteration.
   comm : MPI_Comm
      Allows method to be parallised between processes, a rank for each
      process and size can be defined.
   Nt : size_t
      Number of time-steps to perform.
   dx : double 
      Spacial incrment used to compute the diffusion coefficient.

   Returns
   -------
   None
      The array `uk` is modified in place.
*/

void heat_parallel(double* uk, double dx, size_t Nx, double dt, size_t Nt,
                   MPI_Comm comm);

#endif
