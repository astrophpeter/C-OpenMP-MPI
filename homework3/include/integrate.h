/*
  integrate.h
  -----------

  Defines routines for numerically integrating (x_i,fval_i) data. That is, given
  a function f, some points along a domain x = [x_0, x_1, ..., x_{N-1}] and
  function values

  fvals = [f(x_0), f(x_1), ..., f(x_{N-1})]

  numerically approximate the integral of f using these function evaluations.
  These are meant to replecate the work done by `scipy.integrate.trapz` and
  `scipy.integrate.simps`, but written in C and using OpenMP.



  trapz_seriel

  computes the value of an Integral numerically using the trapaziod
  rule. for a function over domain 'x' with corresponding function
  values 'fvals'. 

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.

  Returns
  ------- 
  out : double
    numerical approximation to the function represented by 'fvals' over
    the domian 'x'.
         
*/ 
double trapz_serial(double* fvals, double* x, int N);

/*  
  trapz_parrallel

  computes the value of an Integral numerically using the trapaziod
  rule. for a function over domain 'x' with corresponding function
  values 'fvals'. Uses open MP to parallised computations. Must be 
  compiled with open mp flag for parallisation. 

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.
  num_threads : int 
    number of threads to be used by open mp when computing in
    parallel.
  
  Returns
  ------- 
  out : double
    numerical approximation to the function represented by 'fvals' over
    the domian 'x'.
         
*/

double trapz_parallel(double* fvals, double* x, int N, int num_threads);

/*  
  time_trapz_parrallel

  times how long the 'trapz_parallel' functions takes to approximate
  a function defined by 'fvals' over the domain 'x' whith a specified
  number of threads for the parallel computation, using the trapaziod
  approximation.

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.
  num_threads : int 
    number of threads to be used by open mp when computing in
    parallel.
  
  Returns
  ------- 
  out : double
    time taked to compute a numerical approximation to the function 
    represented by 'fvals' over the domian 'x', number of threads
    'num_threads'.
         
*/
double time_trapz_parallel(double* fvals, double* x, int N, int num_threads);

/*  
  simps_seriel

  computes the value of an Integral numerically using the simpson
  rule. for a function over domain 'x' with corresponding function
  values 'fvals'. 

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.

  Returns
  ------- 
  out : double
    numerical approximation to the function represented by 'fvals' over
    the domian 'x'.
         
*/
double simps_serial(double* fvals, double* x, int N);
/*  
  simps_parrallel

  computes the value of an Integral numerically using the simpson
  rule. for a function over domain 'x' with corresponding function
  values 'fvals'. Uses open MP to parallised computations. Must be 
  compiled with open mp flag for parallisation. 

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.
  num_threads : int 
    number of threads to be used by open mp when computing in
    parallel.
  
  Returns
  ------- 
  out : double
    numerical approximation to the function represented by 'fvals' over
    the domian 'x'.
         
*/
double simps_parallel(double* fvals, double* x, int N, int num_threads);

/*  
  time_simps_parrallel

  times how long the 'simps_parallel' functions takes to approximate
  a function defined by 'fvals' over the domain 'x' whith a specified
  number of threads for the parallel computation, using the simpson
  approximation.

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.
  num_threads : int 
    number of threads to be used by open mp when computing in
    parallel.
  repeat: int
    number of times to measure timing before average is taken
  
  Returns
  ------- 
  out : double
    time taked to compute a numerical approximation to the function 
    represented by 'fvals' over the domian 'x', number of threads
    'num_threads'.
*/
double time_simps_parallel(double* fvals, double* x, int N, int num_threads,
                           int repeat);
/*  
  simps_parrallel_chunked

  computes the value of an Integral numerically using the simpson
  rule. for a function over domain 'x' with corresponding function
  values 'fvals'. Uses open MP to parallised computations. Must be 
  compiled with open mp flag for parallisation. 

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.
  num_threads : int 
    number of threads to be used by open mp when computing in
    parallel.
  chunk_size : int
    size of chunk in thread is assigned in the parrallel computation

  Returns
  ------- 
  out : double
    numerical approximation to the function represented by 'fvals' over
    the domian 'x'.
         
*/
double simps_parallel_chunked(double* fvals, double* x, int N, int num_threads,
                              int chunk_size);

/*  
  time_simps_parrallel_chunked

  times how long the 'simps_parallel' functions takes to approximate
  a function defined by 'fvals' over the domain 'x' whith a specified
  number of threads for the parallel computation, using the simpson
  approximation.

  Parameters
  ----------
  fvals : double*
    values of the function whose integral it being approximated at
    corresponding 'x' values.
  x : double*
    values of the domian of the function whose integral is being 
    approximanted, corresponding to elements of 'fvals'.
  N : int
    Size of arrays 'fvals' and 'x'. Number of members in the domain
    of the function whose integral is being approximated.
  num_threads : int 
    number of threads to be used by open mp when computing in
    parallel.
  chunk_size : int
    size of chunk in thread is assigned in the parrallel computation.
  repeat: int
    number of times to measure timing before average is taken
  
  Returns
  ------- 
  out : double
    time taked to compute a numerical approximation to the function 
    represented by 'fvals' over the domian 'x', number of threads
    'num_threads'.
*/
double time_simps_parallel_chunked(double* fvals, double* x, int N,
                                   int num_threads, int chunk_size, int repeat);
