import os
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
#import quadprog
import scipy.optimize
#import cvxpy as cp

# Solve quadratic programming for univariate rational approximation
# min 1/2 ||Aq - y||^2 
#  = (Aq - y)^T (Aq - y)
#  = (1/2)q^T A^TA q - (A^T y)^T q

# Constraint on denominator q(xi) to avoid asymptote 
# subject to q(xi) >= epsilon for all xi

# Test on a sweep of epsilons to check the effect of epsilon on geometric error
# compare the error for the entire dataset and LOO-CV

# OPEN QUESTIONS:
#
# 1) If we choose the tau that gets minimum error for each method,
#    which one has the lower sum of squared coeffs? 
#    for this example, b0=1 has lower sum coeff. than b0 variable
#    Bigger coefficients = less robust?
#
# 2) Why do we have infeasible QPs sometimes? Is it a numerical issue
#    or a data issue? Or are some taus really impossible to fit?

def data_points():

    data = np.genfromtxt('data/hpluv.txt')

    y1, y2, y3 = data.transpose()

    x = np.linspace(0, 1, len(y1))

    return x, y1, y2, y3

# return coefficients from quadratic programming
def quad_prog(tau, x_samples, y_samples, constant_is_one):
    num = 5  # numerator degree - n+1 coeffs for degree n
    den = 3 # denominator degree

    # number of datapoints
    m = x_samples.size

    # if true: use constant term b0 = 1
    # if not: no constraint on constant term
    if constant_is_one:
        # num 1, x_i, x_i^2, ...
        # dem    x_i, x_i^2,...
        num_exp = np.arange(0, num+1) 
        den_exp = np.arange(1, den+1)
        # boundary values for inequality -- m-1*1 vector of epsilon - 1
        g = np.full(m, tau - 1.0)

    else:
        num_exp = np.arange(0, num+1)
        den_exp = np.arange(0, den+1)   # includes power of 0 term as variable
        g = np.full(m, tau)

    # m*n+1 coeffs matrix [1, ... x_i^n]
    A_num = x_samples[:, None] ** num_exp[None, :]
    # print("A num shape", A_num.shape)

    # m*d [[-y_1 * x_1, -y_1 * x_1^2, ...]]
    A_den = -y_samples[:, None] * x_samples[:, None] ** den_exp[None, :]    
    # print("A den shape", A_den.shape)
    A = np.hstack((A_num, A_den))

    H_left = np.zeros((m, num+1)) # no constraints on numerator
    H_right = x_samples[:, None] ** den_exp[None, :]
    # print("H right shape", H_right.shape)
    H = np.hstack((H_left, H_right))

    # check if matrix A^T@A is positive definite. Otherwise, QP is unsolvable
    # G = A.T @ A
    # w, v = np.linalg.eigh(G)
    # print('w min:', w.min())
    # u, s, v = np.linalg.svd(A)
    # print('s min:', s.min())

    # option1. qpsolver.solve_qp
    # available: ['daqp', 'ecos', 'osqp', 'scs']
    q = solve_qp(A.T @ A, -A.T @ y_samples, -H, -g, solver='daqp')
    
    # option2. quad prog
    # q, _, _, _, _, _ = quadprog.solve_qp(A.T @  A, 
    #                                     A.T @ y_samples, 
    #                                     H.T, g)
    
    # check if constraint q(x)>=tau is satisfied
    cv = constraint_violation(H, g, q)
    # print("QP cv: ", cv)
    if cv > 1e-5:
        raise RuntimeError(f'constraints not met, violation={cv}')

    # construct vector with numerator and denominator coeffs from q
    p_num = q[:num+1] # stack the first n+1 coeffs as numerator
    if constant_is_one:
        p_den = np.hstack(([1.0], q[num+1:])) # denominator coeffs with added 1 as the 1st entry
    else:
        p_den = np.hstack((q[num+1:]))

    return p_num, p_den, H

def convex(tau, x_samples, y_samples, constant_is_one):
    num = 5
    den = 3

    # number of datapoints
    m = x_samples.size

    if constant_is_one:
        # num 1, x_i, x_i^2, ...
        # dem    x_i, x_i^2,...
        num_exp = np.arange(0, num+1) 
        den_exp = np.arange(1, den+1)
        # boundary values for inequality -- m-1*1 vector of epsilon - 1
        g = np.full(m, tau - 1.0)
        q = cp.Variable(num + den + 1)
    else:
        num_exp = np.arange(0, num+1)
        den_exp = np.arange(0, den+1)   # includes power of 0 term as variable
        g = np.full(m, tau)
        q = cp.Variable(num + den + 2)

    # m*n+1 coeffs matrix [1, ... x_i^n]
    A_num = x_samples[:, None] ** num_exp[None, :]
    # m*d [[-y_1 * x_1, -y_1 * x_1^2, ...]]
    A_den = -y_samples[:, None] * x_samples[:, None] ** den_exp[None, :]    
    A = np.hstack((A_num, A_den))

    H_left = np.zeros((m, num+1)) # no constraints on numerator
    H_right = x_samples[:, None] ** den_exp[None, :]
    H = np.hstack((H_left, H_right))

    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(q, A.T @ A) - (A.T @ y_samples) @ q),
                      [- H @ q <= -g])
    
    # available solvers
    # ['CBC', 'CVXOPT', 'MOSEK', 'GLPK', 'GLPK_MI', 'ECOS', 'SCS', 'SDPA'
    # 'SCIPY', 'GUROBI', 'OSQP', 'CPLEX', 'NAG', 'SCIP', 'XPRESS', 'PROXQP']
    prob.solve(solver = "OSQP", verbose = True )
    q = q.value

    cv = constraint_violation(H, g, q)
    print("Convex: ", cv)
    if cv > 1e-5:
        raise RuntimeError(f'constraints not met, violation={cv}')

    p_num = q[:num+1]
    if constant_is_one:
        p_den = np.hstack(([1.0], q[num+1:])) # denominator coeffs with added 1 as the 1st entry
    else:
        p_den = np.hstack((q[num+1:]))

    return p_num, p_den, H


def evaluate_error(p_num, p_den, x, y):

    y_fit_top = np.polyval(p_num[::-1], x)
    y_fit_bottom = np.polyval(p_den[::-1], x)
    y_fit = y_fit_top / y_fit_bottom

    # squared error of fitting
    error = np.sum(np.square(np.subtract(y, y_fit)))

    return error, y_fit

def constraint_violation(H, g, q):

    return np.maximum(0, (g - H @ q).max())

def optimize_more(tau, x, y, p_num, p_den, H, constant_is_one):

    num_coeff_count = len(p_num)
    m = len(x)

    # reshape p_num and p_den into a single vector
    if constant_is_one:
        assert p_den[0] == 1
        q0 = np.hstack((p_num, p_den[1:]))
        den_exp = np.arange(1, len(p_den))
        g = np.full(m, tau - 1.0)
    else:
        q0 = np.hstack((p_num, p_den))
        # normalize q0 to have unit length
        # q0 /= np.linalg.norm(q0)
        den_exp = np.arange(0, len(p_den))
        g = np.full(m, tau)

    def split_q(q):
        pn = q[:num_coeff_count]
        if constant_is_one:
            pd = np.hstack(([1], q[num_coeff_count:]))
        else:
            pd = q[num_coeff_count:]
        return pn, pd

    def objective(q):
        pn, pd = split_q(q)
        error, _ = evaluate_error(pn, pd, x, y)
        return error

    num_exp = np.arange(0, num_coeff_count)
    Vn = x[:, None] ** num_exp
    Vd = x[:, None] ** den_exp

    def grad(q):

        a, b = split_q(q)

        if constant_is_one:
            b = b[1:]

        g = Vn @ a

        if constant_is_one:
            h = Vd @ b + 1.0
        else:
            h = Vd @ b

        f = g / h

        d = f - y

        # de_da = [de/dd] [dd/df] [df/dg] [dg/da]
        #       = [2.0 * d] [1.0] [1/h] [Vn]

        # de_db = [de/dd] [dd/df] [df/dh] [dh/db]
        #       = [2.0 * d] [1.0] [-g/h^2] [Vd]
        
        de_da = 2.0 * Vn.T @ (d / h)

        de_db = 2.0 * Vd.T @ (-d * g / h**2)

        return np.hstack((de_da, de_db))

    '''
    # check my gradient function

    
    # evaluate the gradient analytically
    J = grad(q0)

    # evaluate the gradient using central difference approximation to derivative
    h = 1e-7
    J_num = np.zeros_like(J)

    # for each different coefficient
    for i in range(len(q0)):
        delta = np.zeros(q0.shape)
        delta[i] = h
        # perturb the coefficent vector along a single dimension
        e0 = objective(q0 - delta)
        e1 = objective(q0 + delta)
        # evaluate the central difference
        J_num[i] = 0.5 * (e1 - e0) / h

    print('J is:', J)
    print('J_num is:', J_num)

    assert np.allclose(J, J_num)

    print('PASS!')

    sys.exit(0)
    '''
        
    # object will enforce that Hq >= g element-wise
    constraints = scipy.optimize.LinearConstraint(H, lb=g)
    options=dict(ftol=1e-15, eps=1e-9, disp=True, maxiter=1000)

    res = scipy.optimize.minimize(objective, q0, method='SLSQP',
                                  jac=grad,
                                  constraints=constraints,
                                  options=options)

    # check if GD satisfies the constraint
    cv = constraint_violation(H, g, res.x)
    print("GD cv: ", cv)
    if cv > 1e-5:
        raise RuntimeError(f'constraint not satisfied after optimization, violation={cv}')

    pn, pd = split_q(res.x)

    error, y_fit = evaluate_error(pn, pd, x, y)

    return pn, pd, error, y_fit

def tau_test(x, y):
    # obtain the x values and the first col of y values

    # check1. LOO-CV
    # compare the geometric error of each single test data from LOO-CV

    # check2. Consistency check
    # compare LOO-CV error to nonlinear least-squares error on the entire dataset
    # 1. train the regressor on the entire dataset
    # 2. evaluate the non-CV data error
    # 3. should be lower than the LOO-CV error or roughly equal

    # plot original data to compare with y_fit
    figure1, axis1 = plt.subplots(2)
    axis1[0].plot(x, y, label = "original", linewidth = 3)
    axis1[1].plot(x, y, label = "original", linewidth = 3)
    figure1.supxlabel("x value")
    figure1.supylabel("y value")
    figure1.suptitle('original vs y_fit data')

    # plot error data for each tau
    figure2, axis2 = plt.subplots(3)

    # plot sum of squared coeffs
    figure3, axis3 = plt.subplots()

    # plot denominator normalized by b0 
    figure4, axis4 = plt.subplots(2)

    for constant_is_one in [True, False]:

        cur_all_errors = []
        cur_gd_errors = []
        cur_sample_errors = []
        cur_den_coeffs = []
        cur_sum_sq_coeffs = []

        # log spaced sweeps of taus
        if constant_is_one: # b0 = 1 cannot have a constraint of tau>1
            taus = np.logspace(-2, -0.001, num=50)
        else:
            taus = np.logspace(-2, 4, num=20)
    
        for tau in taus:
            # conditions to plot
            if constant_is_one:
                which_axis = 0
                title = "b0=1 constant"
            else:
                which_axis = 1
                title = "b0 as variable"

            # train with all x,y datapoints
            # quad prog
            p_num, p_den, H = quad_prog(tau, x, y, constant_is_one)
            # cvxpy
            # p_num, p_den, H = convex(tau, x, y, constant_is_one)

            error, y_fit = evaluate_error(p_num, p_den, x, y)
            cur_all_errors.append(error)
            cur_den_coeffs.append(p_den)  # denominator coeffs for a given tau
            cur_sum_sq_coeffs.append(np.sum(p_num**2) + np.sum(p_den**2))  # sum of squared coeffs of num and den

            print("tau: ", tau)
            print("constant: ", constant_is_one)
            # optimize the initial guess from QP by GD
            p_num, p_den, error, y_fit = optimize_more(tau, x, y, p_num, p_den, H, constant_is_one)
            cur_gd_errors.append(error)
            
            # plot y_fit
            axis1[which_axis].plot(x, y_fit, label = tau)
            axis1[which_axis].set_title(title)

            # plot coeffs.
            axis4[which_axis].plot(p_den / p_den[0], label=tau)
            axis4[which_axis].set_title('denominator/b0 for ' + title)

            # LOO-CV
            cv_error = 0
            for i in range(x.size):
                # create train dataset by eliminating one point
                x_samples = np.delete(x, i)
                y_samples = np.delete(y, i)

                # train
                p_num, p_den, H = quad_prog(tau, x_samples, y_samples, constant_is_one)
                tmp, _ = evaluate_error(p_num, p_den, x[i], y[i])
                cv_error += tmp

            # accumulate errors for LOO-CV
            cur_sample_errors.append(cv_error)

        # global error all data 
        axis2[0].plot(taus, cur_all_errors, 'o', label=title)

        # LOO-CV error
        axis2[1].plot(taus, cur_sample_errors, 'o', label=title)

        # post-gradient descent error
        axis2[2].plot(taus, cur_gd_errors, 'o', label=title)

        # post-gradient descent sum of squared coeffs
        axis3.plot(taus, cur_sum_sq_coeffs, 'o', label=title)


    axis1[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
    axis2[0].set_title("error trained on all data")
    axis2[0].set_xscale("log")
    axis2[0].set_yscale("log")
    axis2[0].legend()

    axis2[1].set_title("error trained with LOO-CV")
    axis2[1].set_xscale("log")
    axis2[1].set_yscale("log")
    axis2[1].legend()

    axis2[2].set_title("post-GD error")
    axis2[2].set_xscale("log")
    axis2[2].set_yscale("log")
    axis2[2].legend()

    figure2.supxlabel("tau value")
    figure2.supylabel("error")

    axis3.set_xscale("log")
    axis3.set_yscale("log")
    axis3.set_title('sum of squared coeffs')
    axis3.legend()

    axis4[0].legend()
    axis4[1].legend()

    plt.show()

def main():
    # create data points from hpluv.txt
    x, y, _, _ = data_points()
    tau_test(x, y)

main()
