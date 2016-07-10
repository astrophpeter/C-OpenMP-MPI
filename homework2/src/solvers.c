#include "linalg.h"
#include "solvers.h"
#include <stdlib.h>

// matrices L, U, and A are all long arrays of size NxN
// b is an array of length N

void solve_lower_triangular(double* out, double* L, double* b, int N)
{
   for (int i = 0; i < N; i++) {
      out[i] = b[i];
      for (int j = 0; j < i; j++) {
         out[i] -= L[i*N + j] * out[j];
      }
      out[i] /= L[i*N + i];
   }
}

void solve_upper_triangular(double* out, double* U, double* b, int N)
{
   for (int i = N - 1; i >= 0; i--) {
      out[i] = b[i];
      for (int j = i + 1; j < N; j++) {
         out[i] -= U[i*N + j] * out[j];
      }
      out[i] /= U[i*N + i];
   }
}

void decompose(double* L, double* U, double* D, double* A, int N)
{
  for (int i = 0; i < N; i++) {
     for(int j = 0; j < N; j++) {
        int index = i*N +j;
        //Lower matrix
        if (i > j) {    
           L[index] = A[index];
           D[index] = 0.0;
           U[index] = 0.0;
        //upper matrix 
        } else if (i < j) {
           U[index] = A[index];
           L[index] = 0.0;
           D[index] = 0.0;
        //diagonal
        } else {
           D[index] = A[index];
           L[index] = 0.0;
           U[index] = 0.0;
        }
     }  
  }
}

void jacobi_step(double* out, double* D, double* U, double* L, double* b, double* xk, int N)
{
  //initialize temp variables.
  double jmat[N*N];
  double jvec[N];  
  for( int i =0; i < N; i++) {
      jvec[i] = 0.0;
  }
  for (int i =0; i < N*N; i++) {
      jmat[i] = 0.0;
  }  
  
  //computer jacobi system.
  mat_add(jmat,L,U,N,N);
  mat_vec(jvec, jmat, xk, N, N);
  vec_sub(jvec,b,jvec,N);
  solve_lower_triangular(out,D,jvec,N);
}

int jacobi(double* out, double* A, double* b, int N, double epsilon)
{
  double * L = malloc(N*N*sizeof(double*));
  double * U = malloc(N*N*sizeof(double*));
  double * D = malloc(N*N*sizeof(double*));
  double * diff = malloc(N*N*sizeof(double*));
  // create L U D
  decompose(L, U, D, A, N);
  int count = 1;

  //initailize current.
  double current[N];
  for(int i = 0; i < N;++i) {
     current[i] = 0.0;
     out[i] = 0.0;
  }
   
  //compute intial guess.
  jacobi_step(out, D, U, L, b, current, N);
  vec_sub(diff, out, current, N);
  double norm = vec_norm(diff, N);
 
  while (norm > epsilon) {
     count++;
     for (int i = 0; i < N; i++) {
        current[i] = out[i];
     }
     jacobi_step(out, D, U, L, b, current, N);
     vec_sub(diff, out, current, N);
     norm = vec_norm(diff, N);
  } 
  
  free(L);
  free(U);
  free(D);
  free(diff);
  return count;
}


void gauss_step(double* out, double* DU, double* L, double* b, double* xk, int N)
{
  //initialize tempory variable.
  double jvec[N]; 
  for( int i =0; i < N; i++) {
      jvec[i] = 0.0;
  }

  //computer iteration 
  mat_vec(jvec, L, xk, N, N);
  vec_sub(jvec, b, jvec, N);
  solve_upper_triangular(out, DU, jvec, N);   
}

int gauss_seidel(double* out, double* A, double* b, int N, double epsilon)
{
  
  double * L = malloc(N*N*sizeof(double*));
  double * U = malloc(N*N*sizeof(double*));
  double * D = malloc(N*N*sizeof(double*));
  double * diff = malloc(N*sizeof(double*));
  // create L U D
  decompose(L, U, D, A, N);
  int count = 1;
   
  
  //initalize current
  double current[N];
  for(int i = 0; i < N;++i) {
     current[i] = 0.0;
     out[i] = 0.0;
  }
  
  //compute D + U
  double DU[N*N];
  for(int i = 0; i < N*N;++i) {
     DU[i] = 0.0;
  }
  mat_add(DU, D, U, N, N);

  //first iteration.
  gauss_step(out, DU, L, b, current, N);
  vec_sub(diff, out, current, N);
  double norm = vec_norm(diff, N);
  
  while (norm > epsilon) {
     count++;
     for (int i = 0; i < N; i++) {
        current[i] = out[i];
     }
     
     gauss_step(out, DU, L, b, current, N);
     vec_sub(diff, out, current, N);
     norm = vec_norm(diff, N);  
  }
  
  free(D);
  free(U);
  free(L);
  free(diff);
  return count;     
}

