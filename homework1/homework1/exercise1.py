# the documenation has been written for you in this exercise

def collatz_step(n):
    """Returns the result of the Collatz function.

    The Collatz function C : N -> N is used in `collatz` to generate collatz
    sequences. Raises an error if n < 1.

    Parameters
    ----------
    n : int

    Returns
    -------
    int
        The result of C(n).

    """
    #test for negative and non interger input.
    if n <= 0:
       raise ValueError("negative value entered - enter a positive value!")
    if type(n) != int:
        raise TypeError("integer not entered - please enter integer")
    elif n % 2 == 0:
       return n / 2
    else:
       if n == 1:
          return 1
       else:
          return 3*n + 1

def collatz(n):
    """Returns the Collatz sequence beginning with `n`.

    It is conjectured that Collatz sequences all end with `1`. Calls
    `collatz_step` at each iteration.

    Parameters
    ----------
    n : int

    Returns
    -------
    sequence : list
        A Collatz sequence.

    """
    if n == 1:
       return [1]
    listn = [n]
    current = n
    #genreate colatz sequence while terms are not 1.
    while (current != 1):
       nextval = collatz_step(current)
       listn.append(nextval)
       current = nextval
       
    return listn











