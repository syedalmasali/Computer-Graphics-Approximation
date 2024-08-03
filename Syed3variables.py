import numpy as np
import matplotlib.pyplot as plt
import qpsolvers


######################################################################
# gives all pairs (deg_x, deg_y) such that deg_x + deg_y <= max_degree

def trivariate_exponents(max_degree):
    exponents = []

    for deg_x in range(max_degree+1):
        for deg_y in range(max_degree+1):
            for deg_z in range(max_degree+1):
                if deg_x + deg_y + deg_z <= max_degree:
                    exponents.append((deg_x, deg_y, deg_z))

    return np.array(exponents)

######################################################################
# make a "vandermonde-like" matrix for trivariate (x, y, z) data

def trivariate_vandermonde(x_samples_flat,
                          y_samples_flat,
                          z_samples_flat,
                          exponents):

    # get each column to isolate x and y exponents
    x_exponents = exponents[:, 0]
    y_exponents = exponents[:, 1]
    z_exponents = exponents[:, 2]
    
    V = (x_samples_flat[:,None] ** x_exponents[None,:] * 
         y_samples_flat[:,None] ** y_exponents[None,:] * z_samples_flat[:, None]** z_exponents[None,:])

    return V
    

def main():
    #load in samples
    lo = -3.0
    hi = 3.0
    samples_per_axis = 25
    x_samples = np.linspace(lo, hi, samples_per_axis)
    y_samples = np.linspace(lo, hi, samples_per_axis)
    z_samples = np.linspace(lo, hi, samples_per_axis)
    epsilon =0.1

    # meshgrid gives all combinations of x & y
    # input arrays are flat (1-D) and output arrays are (2-D)
    x_samples, y_samples, z_samples = np.meshgrid(x_samples, y_samples, z_samples)

    # z_samples has the same shape as x_samples and y_samples
    # which came from meshgrid, so z_samples.shape is (25, 25)
    output = np.exp(-x_samples**2 - y_samples**2 - z_samples**2)

    #flatten
    x_samples_flat = x_samples.flatten()
    y_samples_flat = y_samples.flatten()
    z_samples_flat = z_samples.flatten()
    output_flat = output.flatten()
    m = len(x_samples_flat)

    #het all combinations for exponents
    numer_exponents = trivariate_exponents(3)
    denom_exponents = trivariate_exponents(3)
    
    #get the coeffs matrix
    V_numer = trivariate_vandermonde(x_samples_flat, y_samples_flat, z_samples_flat,
                                    numer_exponents)
    V_denom = trivariate_vandermonde(x_samples_flat, y_samples_flat, z_samples_flat,
                                    numer_exponents)
    k = len(numer_exponents)

    # make sure that denom starts with the constant term
    assert np.all(denom_exponents[0] == 0)

    #skip the constant in denom but make vander matrix
    A = np.hstack((V_numer, V_denom[:, 1:] * -output_flat[:, None]))

    # make our linear inequality matrix & vector
    H_left = np.zeros((m, k)) # no constraints on numerator, just denominator
    H_right = x_samples_flat[:, None] ** denom_exponents[1:,0] *y_samples_flat[:, None] ** denom_exponents[1:, 1] *z_samples_flat[:, None] ** denom_exponents[1:, 2]# Vandermonde matrix for denominator
    H = np.hstack((H_left, H_right))
    #print(H.shape)
    g = np.full(m, epsilon - 1.0)

    #get coeffs
    #q, _, _, _ = np.linalg.lstsq(A, output_flat, rcond=None)
    #print("helft", H_left.shape)
    #print("hright", H_right.shape)
    #print("Atrans", A.T.shape)
    #print("Atrans", A.shape)
    #print("A", z_samples_flat.shape)
    #print("h", H.shape)
    #print("g", g.shape)

    #All solvers are ['daqp', 'ecos', 'osqp', 'scs']
    q = qpsolvers.solve_qp(A.T @ A, -A.T @ output_flat, -H, -g, solver='ecos')
    print("my coeffs", q)

    # evaluate numerator and denominator of our rational function
    f_numer = V_numer @ q[:k]
    f_denom = V_denom @ np.hstack([[1.0], q[k:]])

    #flat array
    recons_samples_flat = A @ q
    # recons_samples.shape is (25, 25)
    #recons_samples = recons_samples_flat.reshape(z_samples.shape)

    # evaluate numerator and denominator of our rational function
    f_numer = V_numer @ q[:k]
    f_denom = V_denom @ np.hstack([[1.0], q[k:]])
    recons_samples_flat = f_numer / f_denom
    recons_samples = recons_samples_flat.reshape(output.shape) #reshape

    #HOW TO PLOT OR CHECK???

main()