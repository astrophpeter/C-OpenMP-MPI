# Hint: you should only need the following functions from numpy and scipy
from numpy import diag, tril, triu, dot, array
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def decompose(A):
    """Creates 3 matrices from A. Diagonal elements, Sub Diagonal Elements, Super Diagonal Elements.
     
    Returns 3 numpy arrays with same shape as A:
         
    One containing lead diagonal elements of A, with zeros otherwise.
    One containing sub diagonal elements of A, zeros otherwise.
    One containing super diagonal elements of A, zeros otherwise.
    
    Parameters
    ----------
    A : numpy.ndarray
       2D matrix.

    Returns
    -------
    D : numpy.ndarray
        array containing lead diagonal of A, zeros everywhere else.
    L : numpy.ndarray
        array containing sub diagonal elements of A, zeros everywhere else.      
    U : numpy.ndarray
        array containing super diagonal elements of A, zeros everywhere else. 
    """
  
    L = tril(A,-1)
    D = diag(diag(A))
    U = triu(A,1)
    
    return D, L, U 


def is_sdd(A):
    """Returns true is matrix A is strictly diagonally dominant, false otherwise.
    
    Returns true if A satisfies |A_{ii}| > sum_{i}(|A{ij}|) for i /= j.
    
    Namely if, the absolute value of every diagonal element of A are  greater than the
    sum of the absolute values of the rest of the elements on its row. Returns
    false otheriwse. 

    Parameters
    ----------
    A : numpy.ndarray
        2D numpy array which is to be test for strick diagonal domanance.

    Returns
    -------
    bool: true if A is stictly diagonally dominant, false otherwise.      
    """
   
    
    for i in range(0,len(A)):
        
        #initialize of diagonal sum to zero.
        sum = 0
        #compute off diagonal sum. 
        for j in range(0, len(A)):
            
	    if i != j:
               sum += abs(A[i,j])
        #check if its bigger than diagonal element.
        if abs(A[i,i]) <= sum:
            return False 
    else: 
        return True       
             

def jacobi_step(D, L, U, b, xk):
    """Reuturns the the next iteration xk1 of jacobi Method when passed previous guess xk.
    
    Impleneted as a step in the jacobi_iteration function.
    Solves for array xk1 for the iterative system of equations:  (D)*xk1 = b - (L+U)*xk.
    Where * denotes vector multiplication.

    Parameters
    ----------
    D : numpy.ndarray
        array containing only lead diagonal elements, zeros everywhere else.
    L : numpy.ndarray
        array containing only sub diagonal elements, zeros everywhere else.      
    U : numpy.ndarray
        array containing only super diagonal elements, zeros everywhere else. 


    Returns
    -------
    xk1 : numpy.ndarry 
        the next iteration of the joacobi method.
    """
    sxk1 = b - dot((L+U),xk)
    kx1 = []
    
    # caluclate xk1 from vector sxk1, by cycling through D
    # and dividing each element in sxk1 by the corresponding
    # diagonal elemnt in D.
    for i in range(0, len(D)):
    
        for j in range(0,len(D)):
            if i == j:
                kx1.append(sxk1[i].astype(float) / D[i,i]) #avoid in division prob
    
    return array(kx1)


def jacobi_iteration(A, b, x0, epsilon=1e-8):
    '''Computes x for Ax=b using the jacobi iteration method.

    Decomposes A into sub (L), super (U) and Lead (D) diagonal matracies using the iteration
    D*xk+1= b - (L+U)xk for initial guess x0. Returns x when ||xk1-xk|| < epsilon (converges
    within epsilon.)

    Raises value error if A is not strictly diagonally dominant.

    Parameters
    ----------
    A : numpy.ndarray
        2D matrix in the problem Ax = b
    B : numpy.ndarray
        1D matrix in the problem Ax = b
    x0 : numpy.ndaarry
        1D matrix, intial guess for x.
    epsilon : float, convergene requirement for interation algorithm. 

    Returns
    -------
    xNext : numpy.ndarray
        x solution to the Ax = b problem   
    
    '''
    #test if A is diagonally dominant
    if not is_sdd(A):
        raise ValueError("A not diagonally dominant, please enter Diagonally dominant matrix")

    D, L, U = decompose(A)
    
    #compute unital iteration
    xCurrent = x0
    xNext = jacobi_step(D, L, U, b, xCurrent)
     
    #compute iterations until convergence within epsilon.
    while (norm(xCurrent - xNext,2) > epsilon):
        xCurrent = xNext
        xNext = jacobi_step(D, L, U, b, xCurrent)
    
    return xNext

def gauss_seidel_step(D, L, U, b, xk):
    """Reuturns the the next iteration xk1 of gauss seidel when passed previous guess xk.
    
    Impleneted as a step in the jacobi_iteration function.
    Solves for array xk1 for the iterative system of equations:  (D+U)*xk1 = b - (L)*xk.
    Where * denotes vector multiplication.

    Parameters
    ----------
    D : numpy.ndarray
        array containing only lead diagonal elements, zeros everywhere else.
    L : numpy.ndarray
        array containing only sub diagonal elements, zeros everywhere else.      
    U : numpy.ndarray
        array containing only super diagonal elements, zeros everywhere else. 


    Returns
    -------
    xk1 : numpy.ndarry 
        the next iteration of the gauss seidel method.

    """
    sxk1 = b - dot(L.astype(float),xk)
    xk1 = solve_triangular(D+U,sxk1)

    return xk1

def gauss_seidel_iteration(A, b, x0, epsilon=1e-8):
    '''Computes x for Ax=b using the gauss seidel  method.

    Decomposes A into sub (L), super (U) and Lead (D) diagonal matracies using the iteration
    (D+U)*xk+1= b - (L)xk for initial guess x0. Returns x when ||xk1-xk|| < epsilon (converges
    within epsilon.)

    Raises value error if A is not strictly diagonally dominant.

    Parameters
    ----------
    A : numpy.ndarray
        2D matrix in the problem Ax = b
    B : numpy.ndarray
        1D matrix in the problem Ax = b
    x0 : numpy.ndaarry
        1D matrix, intial guess for x.
    epsilon : float, convergene requirement for interation algorithm. 

    Returns
    -------
    xNext : numpy.ndarray
        x solution to the Ax = b problem   
    
    '''
    #test if A is diagonally dominant
    if not is_sdd(A):
        raise ValueError("A not diagonally dominant, please enter Diagonally dominant matrix")

    D, L, U = decompose(A)

    #compute unital iteration
    xCurrent = x0
    xNext = gauss_seidel_step(D, L, U, b, xCurrent)

    #compute iterations until convergence within epsilon.
    while (norm(xCurrent - xNext,2) > epsilon):
        xCurrent = xNext
        xNext = jacobi_step(D, L, U, b, xCurrent)

    return xNext

