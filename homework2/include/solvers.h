#ifndef __homework2_solver_h
#define __homework2_solver_h

#include "linalg.h"

/*
  solve_lower_triangular

  Computes 'x' in Lx = b where 'L' is a lower triangular matrix.
  'x' and 'b' are vectors.

  Parameters
  ----------
  out : double*
    Storage for the resulting vector x.
  L : double*
    L lower triangular matrix in Lx=b problem.
  b : double*
    vector in the Lx=b problem.
  N : int
    dimensions of L ('N'x'N') and length of vectors 'out' and 'b'.

  Returns
  -------
  out : double*
    (Output by reference.) 'x' in the problem 'L''x' = 'b'

*/
void solve_lower_triangular(double* out, double* L, double* b, int N);
/*
  solve_upper_triangular

  Computes 'x' in Ux = 'b' where 'U' is a upper triangular matrix.
  'x' and 'b' are vectors.

  Parameters
  ----------
  out : double*
    Storage for the resulting vector x.
  L : double*
    U upper triangular matrix in Ux=b problem.
  b : double*
    vector in the Ux=b problem.
  N : int
    dimensions of U ('N'x'N') and length of vectors 'out' and 'b'.

  Returns
  -------
  out : double*
    (Output by reference.) 'x' in the problem 'U''x' = 'b'

*/
void solve_upper_triangular(double* out, double* U, double* b, int N);

/* 
  decompose

  Decomposes a matrix 'A' into its lower ('L'), upper ('U'), and
  diaginal ('D') elements.

  Parameters
  ----------
  L : double*
    Storage for the resulting lower decomposition.
  U : double*
    Storage for the resulting upper decomposition.
  D : double*
    Storage for the resulting diagonal decomposition.
  A : double*
    Matrix to be decomposed.
  N : int
    size of all matricies (NxN).

  Returns
  -------
  L U D : double*
    (Output by refernce) lower, upper and diagonal decompositions of 'A' 
    respectively.
*/
void decompose(double* L, double* U, double* D, double* A, int N);

/*
  jacobi_step

  Performs one iteration of the genereal 'jacobi' method. Takes an
  intial guess 'xk' and returns the next iteration using the jacobi
  step method.

  Parameters
  ----------
  xk1 : double*
    Storage for resulting vector iteration.
  D : double*
    Decompsed diagonal matrix.
  U : double*
    Decomposed Upper matrix.
  L : double*
    Decomposed lower matrix.
  xk : double*
    Intial vector guess.
  N : int
    Size of the vector 'xk' and 'xk1' and dimension of matricies 'D', 'L', 'U'.

  Returns
  -------
  xk1 : double*
    (Output by reference) results iteration from intial guess 'xk'.
*/
void jacobi_step(double* xk1, double* D, double* U, double* L, double* b, double* xk, int N);
/*
  jacobi
 
  Performs the jacobi iterative method to solve the linear system  'A'x= 'b'.
  Using an inital guess of a 'x' being a zero vector of size 'N'. Such that
  iterations converge within 'epsilon'.

  Parameters
  ----------
  out : double*
    Storage for the resulting solution.
  A : double*
    Matrix ('N'x'N") int he linear problem to be solved.
  b : double*
    Vector of length 'N' in the linear problem to be solved.
  N : int
    Size of vector 'b' and dimensions of matrix 'A'.
  epsilon : double
    required covergence radius of the guess iterations.

  Returns
  -------
  out : double*
   (Output by refernce) solution to 'A'x ='b' problem where iterations
   have converged within raduis 'epsilon'.  

*/
int jacobi(double* out, double* A, double* b, int N, double epsilon);

/*
  gauss_step

  Performs one iteration of the genereal 'gauss_seidel' method. Takes an
  intial guess 'xk' and returns the next iteration using the 'gauss_sedial'
  step method.

  Parameters
  ----------
  out : double*
    Storage for resulting vector iteration.
  DU : double*
    Decompsed diagonal matrix + Decomposed Uppermatrix.
  L : double*
    Decomposed lower matrix.
  xk : double*
    Intial vector guess.
  N : int
    Size of the vector 'xk' and 'xk1' and dimension of matricies 'DU', 'L'.

  Returns
  -------
  out : double*
    (Output by reference) results iteration from intial guess 'xk'.
*/
void gauss_step(double* out, double* DU, double* L, double* b, double* xk, int N);

/*
  gauss_seidel
 
  Performs the gaus_seidel iterative method to solve the linear system  'A'x= 'b'.
  Using an inital guess of a 'x' being a zero vector of size 'N'. Such that
  iterations converge within 'epsilon'.

  Parameters
  ----------
  out : double*
    Storage for the resulting solution.
  A : double*
    Matrix ('N'x'N") int he linear problem to be solved.
  b : double*
    Vector of length 'N' in the linear problem to be solved.
  N : int
    Size of vector 'b' and dimensions of matrix 'A'.
  epsilon : double
    required covergence radius of the guess iterations.

  Returns
  -------
  out : double*
   (Output by refernce) solution to 'A'x ='b' problem where iterations
   have converged within raduis 'epsilon'.  

*/
int gauss_seidel(double* out, double* A, double* b, int N, double epsilon);

#endif
