import numpy as np
import matplotlib.pyplot as plt

def lagrange_poly(p,xhat,n,x,tol):
    """
    Parameters
    ----------
    p : positive integer 
        Number of nodal points. 
    xhat : numpy.ndarray of shape (p+1,)
        The array containing the distinct nodal points. 
    n : positive integer 
        Number of evaluation points. 
    x : numpy.ndarray of shape (n,)
        The array containing the evaluation points. 
    tol : real number 
        Error number. 

    Returns
    -------
    lagrange_matrix : numpy.ndarray of shape(p+1,n)
        A matrix where the ijth entry of the matrix equals Li(xj).
    error_flag : integer
        A integer which is either 0 or 1 which tells you if the nodal points are distinct. 
    """
    lagrange_matrix = np.zeros((p+1,n))
    for i in range(p+1):
        xlist = np.delete(xhat,i)
        z = np.zeros(p)
        for q in range(p):
            z[q] = xhat[i] - xlist[q]
            diff = abs(xhat[i] - xlist[q])
            if diff < tol:
                error_flag = 1
            else:
                error_flag = 0 
        denom = np.prod(z)
        for j in range(n):
            v = np.zeros(p)
            for k in range(p):
                v[k] = x[j] - xlist[k]
            numer = np.prod(v)
            L = numer / denom 
            lagrange_matrix[i,j] = L 
    return lagrange_matrix, error_flag

def uniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    """
    Parameters
    ----------
    a : real number 
        Lower bound of the interval which the nodal points are uniformly spaced over.  
    b : real number
        Upper bound of the interval which the nodal points are uniformly spaced over. 
    p : positive integer 
        The order of the polynomial interpolant. 
    n : positive integer
        The number of evaluation points. 
    x : numpy.ndarray of shape (n,)
        The array containing the evaluation points. 
    f : function 
        The function of the polynomial interpolant.
    produce_fig : bool 
        Determines whether the figure showing the plot of the interpolant and function is shown.

    Returns
    -------
    interpolant : numpy.ndarray of shape (n,)
        A array containing the value of the pth order polynomial interpolant evaluated at the points contained in the array x. 
    fig : matplotlib.figure.Figure 
        A figure showing (if produce_fig is true) the plot of the function and interpolant evaluated at the evaluation points. 
    """
    xhat = np.linspace(a,b,p+1)
    tol = 1.0e-10
    lag = lagrange_poly(p,xhat,n,x,tol)[0]
    lis2 = np.zeros(n)
    for s in range(n):
        lis = np.zeros(p+1)
        for t in range(p+1):
            lis[t] = f(xhat[t])*lag[t,s]
        lis2[s] = np.sum(lis)
    interpolant = lis2     
    if produce_fig == True:       
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,f(x),label="Function",color="r")
        ax.plot(x,interpolant,label="Interpolant (uniform nodal points)",color="b")
        ax.set_title("Plot of function and interpolant")
        ax.legend()
        plt.show()
    else:
        fig = None 
    return interpolant, fig

def nonuniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    """
    Parameters
    ----------
    a : real number 
        Lower bound of the interval which the nodal points are nonuniformly spaced over.
    b : real number
        Upper bound of the interval which the nodal points are nonuniformly spaced over.
    p : positive integer 
        The order of the polynomial interpolant.
    n : positive integer
        The number of evaluation points. 
    x : numpy.ndarray of shape (n,)
        The array containing the evaluation points. 
    f : function 
        The function of the polynomial interpolant.
    produce_fig : bool 
        Determines whether the figure showing the plot of the interpolant and function is shown.

    Returns
    -------
    nu_interpolant :  numpy.ndarray of shape (n,)
        A array containing the value of the pth order polynomial interpolant evaluated at the points contained in the array x.
    fig : matplotlib.figure.Figure 
        A figure showing (if produce_fig is true) the plot of the function and interpolant evaluated at the evaluation points.
    """
    xhat = np.zeros(p+1)
    for i in range(p+1):
        xhat[i] = np.cos(((2*i+1)/(2*(p+1)))*np.pi)
    for q in range(p+1):
        if xhat[q] >= 0:
            dif1 = 1 - xhat[q] 
            poc = dif1 / 2 
            poc2 = poc*(b-a)
            xhat[q] = b - poc2
        elif xhat[q] < 0:
            dif1 = xhat[q] + 1
            poc = dif1 / 2 
            poc2 = poc*(b-a)
            xhat[q] = a + poc2 
    tol = 1.0e-10
    lag = lagrange_poly(p,xhat,n,x,tol)[0]
    lis2 = np.zeros(n)
    for s in range(n):
        lis = np.zeros(p+1)
        for t in range(p+1):
            lis[t] = f(xhat[t])*lag[t,s]
        lis2[s] = np.sum(lis)
        nu_interpolant = lis2
    if produce_fig == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,f(x),label="Function",color="r")
        ax.plot(x,nu_interpolant,label="Interpolant (nonuniform nodal points)",color="b")
        ax.set_title("Plot of function and interpolant")
        ax.legend()
        plt.show()
    else:
        fig = None 
    return nu_interpolant, fig


def compute_errors(a,b,n,P,f):
    """To begin with, the maximum error is given by the following 
inequality. 
max(|pₚ(x) - f(x)|) ≤ max|f⁽ᵖ⁺¹⁾(ξ)/(p+1)!|max|(x-x₀)...(x-xₚ)| , 
where a ≤ x,ξ ≤ b. 
Here we call wₚ(x) = (x-x₀)(x-x₁)...(x-xₚ). 
In both questions a) and b), 1/(p+1)! is determined by the order of 
the polynomial interpolant.
Furthermore, in both questions, max(|wₚ(x)|) is the same and can be
determined by the following code where ans = max(|wₚ(x)|).

p = "order/degree of interpolant"
n = p + 1  
evapoints = np.linspace(a,b,2000)
xhat = np.linspace(-1,1,n)
tlist1 = np.zeros(2000)
tlist2 = np.zeros(n)
for i in range(2000):
    for j in range(n):
        brac = evapoints[i] - xhat[j]
        tlist2[j] = brac 
    tlist1[i] = np.prod(tlist2)
tlist1 = abs(tlist1)
ans = max(tlist1)

This code shows the term max(|wₚ(x)|) decreases as p increases. 
Therefore, both 1/(p+1)! and max(|wₚ(x)|) decreases as p 
(the order) increases in both a) and b). 
Here we call Ψₚ = max(|wₚ(x)|) / (p+1)!
    
In a) max(|f⁽ᵖ⁺¹⁾(ξ)|) = (2π)ᵖ⁺¹. As p increases, the rate at which 
this term increases is always less than the rate at which the term 
Ψₚ (max(|wₚ(x)|) / (p+1)!) decreases. Therefore, the error term 
decreases as p increases from 1 to 40. Furthermore, the error 
tends to 0 as p tends to infinity. 

In b) max(|f⁽ᵖ⁺¹⁾(ξ)|) = (2π)ᵖ⁺¹ + 0.01((10π)ᵖ⁺¹). Here, the rate 
at which this term increases is greater than the rate at which 
Ψₚ decreases from p = 1 to approximately p = 20. 
Therefore from 1 ≤ p ≤ 20, the error term is increasing. After 
p = 20, the rate at which max(|f⁽ᵖ⁺¹⁾(ξ)|) is increasing begins
to become less than the rate at which the term Ψₚ is decreasing 
causing the error term (which is still greater than 1, 
implying max(|f⁽ᵖ⁺¹⁾(ξ)|) > Ψₚ as long as the error term remains 
greater than 1) to also decrease. Therefore, the error term 
reaches its maximum and peaks around p = 20, and begins to 
fall from 20 ≤ p ≤ 40. 
    """
    x = np.linspace(a,b,2000)
    error_matrix = np.zeros((2,n))
    for j in range(2):
        if j == 0:
            for i in range(n):
                p = P[i]
                lis,fig = uniform_poly_interpolation(a,b,p,2000,x,f,False)
                error = np.zeros(2000)
                for k in range(2000):
                    error[k] = abs(lis[k] - f(x[k]))
                maxerror = max(error)
                error_matrix[j,i] = maxerror 
        elif j == 1:
            for i in range(n):
                p = P[i]
                lis,fig = nonuniform_poly_interpolation(a,b,p,2000,x,f,False)
                error = np.zeros(2000)
                for k in range(2000):
                    error[k] = abs(lis[k] - f(x[k]))
                maxerror = max(error)
                error_matrix[j,i] = maxerror
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.semilogy(P,error_matrix[0],label="Uniform errors",color="b")
    plt.semilogy(P,error_matrix[1],label="Nonuniform errors",color="r")
    plt.title("Error values for different polynomial degrees")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Errors")
    if len(P) > 10:
        None
    else:
        plt.xticks(P)
    ax.legend()
    plt.show() 
    return error_matrix, fig

#%%
def piecewise_interpolation(a,b,p,m,n,x,f,produce_fig):
    """
    Parameters
    ----------
    a : real number 
        Lower bound of the interval which the nodal points are uniformly spaced over.
    b : real number
        Upper bound of the interval which the nodal points are uniformly spaced over.
    p : positive integer 
        The order of the polynomial interpolant.
    m : positive integer 
        The number of uniformly spaced subintervals of width (b-a)/m. 
    n : positive integer
        The number of evaluation points.
    x : numpy.ndarray of shape (n,)
        The array containing the evaluation points. 
    f : function 
        The function of the polynomial interpolant.
    produce_fig : bool 
        Determines whether the figure showing the plot of the interpolant and function is shown.

    Returns
    -------
    p_interpolant : numpy.ndarray of shape (n,)
        A array containing the value of the pth order piecewise polynomial interpolant evaluated at the points contained in the array x.
    fig : matplotlib.figure.Figure 
        A figure showing (if produce_fig is true) the plot of the function and interpolant evaluated at the evaluation points.
    """
    
    xhat2 = np.linspace(a,b,m+1)
    p_interpolant = np.zeros(n)
    for i in range(m):
        a = xhat2[i]
        b = xhat2[i+1]
        for j in range(n):
            if x[j] <= b and x[j] >= a:
                n2 = 1
                x2 = np.array([x[j]])
                eva = nonuniform_poly_interpolation(a,b,p,n2,x2,f,False)[0]
                p_interpolant[j] = eva  

    if produce_fig == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,f(x),label="Function",color="r")
        ax.plot(x,p_interpolant,label="Piecewise interpolant",color="b")
        ax.set_title("Plot of function and piecewise interpolant")
        ax.legend()
        plt.show()
    else:
        fig = None
    return p_interpolant, fig
