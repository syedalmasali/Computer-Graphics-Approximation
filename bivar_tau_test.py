import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
#import quadprog
import scipy.optimize

# Bivariate rational approximation with QP
# min 1/2 ||Aq - y||^2 
#  = (Aq - y)^T (Aq - y)
#  = (1/2)q ^T A^T A q - q^T A^T y

# Constraint on denominator q(xi) to avoid asymptote 
# subject to q(xi) >= epsilon for all xi

# Test on a sweep of epsilons to check the effect of epsilon on geometric error
# compare the error for the entire dataset and LOO-CV

def bivariate_exponents(max_degree):
    exponents = []
    for deg_x in range(max_degree+1):
        for deg_y in range(max_degree+1):
            if deg_x + deg_y <= max_degree:
                exponents.append((deg_x, deg_y))
    exponents = np.array(exponents)

    return np.array(exponents)

# make a "vandermonde-like" matrix for bivariate (x, y) data
def bivariate_vandermonde(x_samples,
                          y_samples,
                          exponents):
    # get each column to isolate x and y exponents
    x_exponents = exponents[:, 0]
    y_exponents = exponents[:, 1]
    V = (x_samples[:,None] ** x_exponents[None,:] * 
         y_samples[:,None] ** y_exponents[None,:])

    return V

# return coefficients from quadratic programming
def quad_prog(num, den, tau, x_samples, y_samples, z_samples, constant_is_one):
    # transform to 1D
    x_samples = x_samples.flatten()
    y_samples = y_samples.flatten()
    z_samples = z_samples.flatten()

    m = len(x_samples)    # number of datapoints
    
    # pairs of exponents from max degrees 
    num_exp = bivariate_exponents(num)
    den_exp = bivariate_exponents(den)

    # number of coeffs of numerator
    # max num degree ≠ number of coeffs
    num_coeff_count = len(num_exp)  

    #get the coeff matrices
    V_num = bivariate_vandermonde(x_samples, y_samples, num_exp)
    V_den = bivariate_vandermonde(x_samples, y_samples, den_exp)

    # if true: use constant term b0 = 1
    if constant_is_one:
        # stack them into 1 matrix but ignore the denom constant
        A = np.hstack((V_num, V_den[:, 1:] * -z_samples[:, None]))

        # no constraints on numerator
        H_left = np.zeros((m, num_coeff_count)) 
        H_right = V_den[:, 1:]
        H = np.hstack((H_left, H_right))
        # boundary values for inequality -- m-1*1 vector of epsilon - 1
        g = np.full(m, tau - 1.0)

    # if not: no constraint on constant term
    else:
        # construct including the b_0 coeffs for denominator
        A = np.hstack((V_num, V_den * -z_samples[:, None])) 

        # include b_0 coeffs
        H_left = np.zeros((m, num_coeff_count))
        H_right = V_den
        H = np.hstack((H_left, H_right))
        g = np.full(m, tau)

    # option1. qpsolver.solve_qp
    # available solver: ['daqp', 'ecos', 'osqp', 'scs']
    q = solve_qp(A.T @ A, -A.T @ z_samples, -H, -g, solver='daqp')

    # option2. quadprog
    # q, _, _, _, _, _ = quadprog.solve_qp(A.T @  A, A.T @ z_samples, H.T, g)

    # Check if constraint of q(x,y)>=tau is met
    max_constraint_violation = constraint_violation(H, g, q)
    if max_constraint_violation > 1e-5:
        print(f"QP cv violated: {max_constraint_violation} at tau = {tau}")
        # raise RuntimeError(f'constraints not met, violation={max_constraint_violation}')

    # numerator coeffs
    p_num = q[:num_coeff_count]
    # denominator coeffs
    if constant_is_one:
        p_den = np.hstack(([1.0], q[num_coeff_count:])) # denominator coeffs with added 1 as the 1st entry
    else:
        p_den = np.hstack((q[num_coeff_count:]))

    return V_num, V_den, p_num, p_den, H

def evaluate_error(p_num, p_den, V_num, V_den, z):
    # rational fitting
    fit_num = V_num @ p_num
    fit_den = V_den @ p_den

    # convert flat z_fit to original 2D shape
    z_fit = (fit_num / fit_den).reshape(z.shape)
    error = np.sum(np.square(np.subtract(z, z_fit)))

    return error, z_fit

def constraint_violation(H, g, q):

    return np.maximum(0, (g - H @ q).max())

def optimize_more(tau, x, y, z, p_num, p_den, V_num, V_den, H, constant_is_one):
    num_coeff_count = len(p_num)    # column size of numerator Vandermonde matrix
    m = len(x.flatten())    # row size of Vandermonde matrix 

    if constant_is_one:
        q0 = np.hstack((p_num, p_den[1:]))
        g = np.full(m, tau - 1.0)
    else:
        q0 = np.hstack((p_num, p_den))
        g = np.full(m, tau)

    def split_q(q):
        p_num = q[:num_coeff_count]
        if constant_is_one:
            p_den = np.hstack(([1.0], q[num_coeff_count:]))
        else:
            p_den = q[num_coeff_count:]

        return p_num, p_den

    # minimize error with initial guess q0
    def objective(q):
        p_num, p_den = split_q(q)
        error, _ = evaluate_error(p_num, p_den, V_num, V_den, z)

        return error


    def grad(q):
        Vn = V_num
        a, b = split_q(q)
        g = Vn @ a

        if constant_is_one:
            Vd = V_den[:, 1:]
            b = b[1:]
            h = Vd @ b + 1.0
        else:
            Vd = V_den
            h = V_den @ b

        f = g / h
        d = f - z.flatten()

        # de_da = [de/dd] [dd/df] [df/dg] [dg/da]
        #       = [2.0 * d] [1.0] [1/h] [Vn]

        # de_db = [de/dd] [dd/df] [df/dh] [dh/db]
        #       = [2.0 * d] [1.0] [-g/h^2] [Vd]
        
        de_da = 2.0 * Vn.T @ (d / h)

        de_db = 2.0 * Vd.T @ (-d * g / h**2)

        return np.hstack((de_da, de_db))

    constraints = scipy.optimize.LinearConstraint(H, lb = g)
    # more precision minimizing function and step size
    # raise iterations limit
    options=dict(ftol=1e-5, eps=1e-3, disp=False, maxiter=1000)

    res = scipy.optimize.minimize(objective, q0, method = "SLSQP",
                                  jac = grad,
                                  constraints=constraints,
                                  options=options)

    # options=dict(disp=True, maxiter=1000)
    # res = scipy.optimize.minimize(objective, q0, method = "trust-constr",
    #                               jac = grad,
    #                               constraints=constraints,
    #                               options=options)

    max_constraint_violation = constraint_violation(H, g, res.x)
    if max_constraint_violation > 1e-5:
        print(f"GD cv violated: {max_constraint_violation} at tau = {tau}")
        # raise RuntimeError(f'constraint not satisfied after optimization, violation={max_constraint_violation}')

    p_num, p_den = split_q(res.x)
    error, z_fit = evaluate_error(p_num, p_den, V_num, V_den, z)

    return p_num, p_den, error, z_fit


def tau_test(num, den, x, y, z):

    # plot error data for each tau
    figure1, axis1 = plt.subplots(2) 
    
    # plot sum of squared coeffs
    figure2, axis2 = plt.subplots()

    # plot denominator normalized by b0 
    figure3, axis3 = plt.subplots(2)

    pre_fits = []
    cur_fits = []
    indices_tau = []

    for constant_is_one in [True, False]:

        cur_qp_errors = []
        cur_gd_errors = []
        cur_den_coeffs = []
        cur_sum_sq_coeffs = []

        # log spaced sweeps of taus
        if constant_is_one:
            taus = np.logspace(-2, -0.001, num=10)  # For b0=1, tau>1 is not mathematically feasible
        else:
            taus = np.logspace(-2, 4, num=10)   # b0 variable can take any tau

        for tau in taus:
            if constant_is_one:
                which_axis = 0
                title = "b0=1 constant"
            else:
                which_axis = 1
                title = "b0 as variable"

            # trained on all datapoints
            # True: setting b_0 as constant 1
            V_num, V_den, p_num, p_den, H = quad_prog(num, den, tau, x, y, z, constant_is_one)

            # print("constant", constant_is_one)
            # print("tau", tau)

            error, pre_z_fit = evaluate_error(p_num, p_den, V_num, V_den, z)
            cur_qp_errors.append(error)    # error between z_fit and true z-values
            cur_den_coeffs.append(p_den)    # den coeffs values
            cur_sum_sq_coeffs.append(np.sum(p_num**2) + np.sum(p_den**2))  # sum of squared coeffs of num and den
            pre_fits.append(pre_z_fit)

            # optimize more
            p_num, p_den, error, post_z_fit = optimize_more(tau, x, y, z, p_num, p_den, V_num, V_den, H, constant_is_one)
            cur_gd_errors.append(error) # error after gradient descent
            cur_fits.append(post_z_fit)

            # plot denominator normalized by b0
            axis3[which_axis].plot(p_den / p_den[0], label=tau)
            axis3[which_axis].set_title("denominator/b0 for " + title)

        # plot QP errors for each tau
        axis1[0].plot(taus, cur_qp_errors, 'o', label=title)
        # plot errors for each tau
        axis1[1].plot(taus, cur_gd_errors, "o", label=title)
        # plot sum of squared coeffs
        axis2.plot(taus, cur_sum_sq_coeffs, 'o', label=title)

        # identify tau with the smallest tau values
        index_tau = np.argmin(cur_gd_errors)
        indices_tau.append(index_tau)        
    print(f"For {constant_is_one}, tau where GD error minimized: {taus[index_tau]} at error={cur_gd_errors[index_tau]}")

    # fits
    pre_fits = np.array(pre_fits)
    cur_fits = np.array(cur_fits)

    # indices tau: first entry is for b0=1, the second entry is for b0 variable
    # Out of 10 fits, the first 5 entries is for b0=1. The index in the 5 entries is the best fit
    # The next 5 entries is for b0 variable
    const_index = indices_tau[0]
    var_index = indices_tau[1]

    best_fits_const = cur_fits[const_index, :, :]
    best_fits_var = cur_fits[var_index + 5, :, :]

    # plot errors for each tau
    axis1[0].set_title("error of QP")
    axis1[0].set_xscale("log")
    axis1[0].set_yscale("log")
    axis1[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
    axis1[1].set_title("GD error")
    axis1[1].set_xscale("log")
    axis1[1].set_yscale("log")
    axis1[1].legend(bbox_to_anchor=(1, 1), loc='upper left')
    figure1.supxlabel("taus")
    figure1.supylabel("errors")

    # plot sum of squared coeffs
    axis2.set_title('sum of squared coeffs')
    axis2.set_xscale("log")
    axis2.set_yscale("log")
    figure2.supxlabel("taus")
    figure2.supylabel("sum of squared coeffs")
    axis2.legend()

    # plot denominator normalized by b0 
    figure3.supylabel("den coeffs normalized with b0")
    axis3[0].legend(bbox_to_anchor=(1, 1), loc='upper left')

    return  best_fits_const, best_fits_var

def main():

    rng = np.linspace(-2, 2, 17)
    x, y = np.meshgrid(rng, rng)
    z = np.exp(-x**2 - y**2)

    num = 5
    den = 3
    best_fits_const, best_fits_var = tau_test(num, den, x, y, z)

    fig, ax = plt.subplots(3)
    ax[0].pcolormesh(x, y, best_fits_const)
    ax[0].set_title("b0=1")
    ax[1].pcolormesh(x, y, best_fits_var)
    ax[1].set_title("b0 var")
    ax[2].pcolormesh(x, y, z)
    ax[2].set_title("original")
    plt.show()

    
if __name__ == '__main__':
    main()

