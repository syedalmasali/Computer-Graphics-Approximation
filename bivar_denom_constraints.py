import numpy as np
import matplotlib.pyplot as plt
import qpsolvers

######################################################################
# gives all pairs (deg_x, deg_y) such that deg_x + deg_y <= max_degree
def bivariate_exponents(max_degree):
    exponents = []
    for deg_x in range(max_degree+1):
        for deg_y in range(max_degree+1):
            if deg_x + deg_y <= max_degree:
                exponents.append((deg_x, deg_y))
    return np.array(exponents)


######################################################################
# make a "vandermonde-like" matrix for bivariate (x, y) data
def bivariate_vandermonde(x_samples_flat,
                          y_samples_flat,
                          exponents):
    # get each column to isolate x and y exponents
    x_exponents = exponents[:, 0]
    y_exponents = exponents[:, 1]
    V = (x_samples_flat[:,None] ** x_exponents[None,:] * 
         y_samples_flat[:,None] ** y_exponents[None,:])
    return V
    

def main():

    lo = -3.0
    hi = 3.0
    samples_per_axis = 25
    epsilon= 0.1
    
    x_samples = np.linspace(lo, hi, samples_per_axis)
    y_samples = np.linspace(lo, hi, samples_per_axis)

    # meshgrid gives all combinations of x & y
    # input arrays are flat (1-D) and output arrays are (2-D)
    x_samples, y_samples = np.meshgrid(x_samples, y_samples)

    # z_samples has the same shape as x_samples and y_samples
    z_samples = np.exp(-x_samples**2 - y_samples**2)

    #transform to 1D
    x_samples_flat = x_samples.flatten()
    y_samples_flat = y_samples.flatten()
    z_samples_flat = z_samples.flatten()
    m = len(x_samples_flat)


    #Get mesh of exponents
    numer_exponents = bivariate_exponents(3)
    denom_exponents = bivariate_exponents(3)

    print(numer_exponents)

    #get the coeff matrices
    V_numer = bivariate_vandermonde(x_samples_flat, y_samples_flat,
                                    numer_exponents)
    V_denom = bivariate_vandermonde(x_samples_flat, y_samples_flat,
                                    denom_exponents)

    # make sure that denom starts with the constant term
    assert np.all(denom_exponents[0] == 0)
    #stack them into 1 matrix but ignore the denom constant
    A = np.hstack((V_numer, V_denom[:, 1:] * -z_samples_flat[:, None]))
    #get length of numerator exponents for slicing purposes
    k = len(numer_exponents)
    print(k)


    # make our linear inequality matrix & vector
    H_left = np.zeros((m, k)) # no constraints on numerator, just denominator
    H_right = x_samples_flat[:, None] ** denom_exponents[1:,0] *y_samples_flat[:, None] ** denom_exponents[1:, 1]# Vandermonde matrix for denominator
    H = np.hstack((H_left, H_right))
    g = np.full(m, epsilon - 1.0)

    #get coeffs
    print("helft", H_left.shape)
    print("hright", H_right.shape)
    print("Atrans", A.T.shape)
    print("Atrans", A.shape)
    print("A", z_samples_flat.shape)
    print("h", H.shape)
    print("g", g.shape)
    q = qpsolvers.solve_qp(A.T @ A, -A.T @ z_samples_flat, -H, -g, solver='daqp')
    print(q)

    # evaluate numerator and denominator of our rational function
    f_numer = V_numer @ q[:k]
    f_denom = V_denom @ np.hstack([[1.0], q[k:]])

    #divide numer and denom, then reshape
    recons_samples_flat = f_numer / f_denom
    recons_samples = recons_samples_flat.reshape(z_samples.shape)

    print("original", z_samples.shape)
    print("fitted", z_samples_flat.shape)


    #creating the original plot
    plt.subplot(2, 1, 1)
    plt.pcolormesh(x_samples, y_samples, z_samples)
    plt.title('original')
    plt.colorbar()

    #creating the fitted plot
    plt.subplot(2, 1, 2)
    plt.pcolormesh(x_samples, y_samples, recons_samples)
    plt.title('fit')
    plt.colorbar()

    #shows graph
    plt.show()
    


main()