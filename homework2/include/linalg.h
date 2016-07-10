#ifndef __homework2_linalg_h
#define __homework2_linalg_h

/*
  vec_add

  Computes the sum of two vectors.

  Parameters
  ----------
  out : double*
    Storage for the resulting sum vector.
  v : double*
  w : double*
    The two vectors to sum.
  N : int
    The length of the vectors, `out`, `v`, and `w`.

  Returns
  -------
  out : double*
    (Output by reference.) The sum of `v` and `w`.
*/
void vec_add(double* out, double* v, double* w, int N);

/*
  vec_sub

  Computes the difference of two vectors.

  Parameters
  ----------
  out : double*
    Storage for the resulting differnce vector.
  v : double*
  w : double*
    The two vectors to take differnce 
  N : int
    The length of the vectors, `out`, `v`, and `w`.

  Returns
  -------
  out : double*
    (Output by reference.) The difference of `v` and `w`.
  
*/
void vec_sub(double* out, double* v, double* w, int N);

/*
  vec_norm

  Computes the the 2nd norm of a vector.

  Parameters
  ----------.
  v : double*
  N : int
    The length of the vector v 

  Returns
  -------
  norm : double
    Computes the second norm of the vector v'.
  
*/
double vec_norm(double* v, int N);

/*
  mat_add

  Computes the sum of two matracies of same size

  Parameters
  ----------
  out : double*
    Storage for the resulting sum matrix.
  A : double*
  B : double*
    The two matricies to take the sum.
  N : int
    The number colums of the matracies, `out`, `A`, and `B`. 
  M : int
    The number row of the matracies, `out`, `A`, and `B`.
  
  Returns
  -------
  out : double*
    (Output by reference.) The sum of `A` and `B`.
  
*/
void mat_add(double* out, double* A, double* B, int M, int N);

/*
  mat_vec

  Computes the inner product of a matrix and a vector.

  Parameters
  ----------
  out : double*
    Storage for the resulting vector.
  A : double*
    The matrix in the inner product.
  x : double*
    The vector in the inner product.
  N : int
    The number colums of the matrix 'A'.
  M : int
    The number rows in matrix 'A' and vector 'x'.
  
  Returns
  -------
  out : double*
    (Output by reference.) Inner product of 'A' anf 'x',
  
*/
void mat_vec(double* out, double* A, double* x, int M, int N);

/*
  mat_vec

  Computes the inner product of two matricies.

  Parameters
  ----------
  out : double*
    Storage for the resulting matrix
  A : double*
    The matrix in the inner product.
  B : double*
    The other matrix in the inner product.
  N : int
    The number colums of the matrix 'A', number of rows in 'B'.
  M : int
    The number rows in matrix 'A', and 'out'.
  K : int
    The number of columns of matrix 'B' and 'out'.
  Returns
  -------
  out : double*
    (Output by reference.) Inner product of 'A' anf 'B'
  
*/
void mat_mat(double* out, double* A, double* B, int M, int N, int K);

#endif
