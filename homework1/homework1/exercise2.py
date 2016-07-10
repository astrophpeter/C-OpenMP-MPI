
def gradient_step(xk, df, sigma=0.5):
    """Returns one iteration of the gradient Descent method.
    
    This functions computes and returns one gradient descent with scaling
    factor sigma: `xk` -`sigma` * f'(`xk`).
    
    It is used in the implentation the `gradient_descent` method.
    
    Parameters
    ----------
    xk : Float
        Starting x value for the iteration.
    df : Lambda
        Gradient of the function that's being descended.
    signma : float
	Default: 0.5. Scaling factor of the gradient descent.
     
    Returns
    -------
    _ : Float
        returns the updated value of x after one iteration.
    """
     
    return xk - sigma*df(xk)

def gradient_descent(f, df, x, sigma=0.5, epsilon=1e-8):
    """Returns a minima of `f` using the Gradient Descent method.

    A local minima, x*, is such that `f(x*) <= f(x)` for all `x` near `x*`.
    This function returns a local minima which is accurate to within `epsilon`.

    `gradient_descent` raises a ValueError if `sigma` is not strictly between
    zero and one.

    Parameters
    ----------
    f : Lambda
        The function that the gradient desecent method is being used on.
    df : Lambda
        Gradient of the function that's being descended.
    x  : Float
        Inital guess of the local minima.
    sigma : Float
        Default: 0.5. Scaling factor of the gradient descent.
    epsilon : Float
        Deafault: 1e-8. Precision to which the value is the local minima is 
                        computed.
     
    Returns
    -------
    xk1 : Float
        The value of x for which the local minima of f occurs. 
    """
    # sigma range test
    if sigma < 0 or sigma > 1:
        raise ValueError("Enter a sigma between 0 and 1") 
    
    #initialize
    xk1 = x
    xk = x + 1
  
    #update the guest of local minima until with in precison 
    # (epsilon).
    while (abs(xk1 - xk) > epsilon) :
	
        # if function is diverging to -infinty
        # has no local minina.
        if (f(xk1) <-10000) :
            raise ValueError("No local minima")
        xk = xk1
	xk1 = gradient_step(xk, df, sigma)


    #test if point is actual minima
    if (df(xk1 - 0.01) < 0 and df(xk1 + 0.01) > 0) :
        return xk1;
    #if not minima, perturb and recursively call self.
    elif (df(xk1 - 0.01) > 0) :
        return gradient_descent(f, df, xk1 - 0.01)
    else :
        return gradient_descent(f, df, xk1 + 0.01)
	
