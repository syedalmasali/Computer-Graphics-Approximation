import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import qpsolvers

def make_vandermonde_matrix(degree, x):

    exponents = np.arange(degree+1)
    
    return x[:,None] ** exponents[None,:]

def main():

    # red channel data from viridis colormap
    # https://bids.github.io/colormap/
    y = np.array([
        0.267004, 0.26851 , 0.269944, 0.271305, 0.272594, 0.273809,
        0.274952, 0.276022, 0.277018, 0.277941, 0.278791, 0.279566,
        0.280267, 0.280894, 0.281446, 0.281924, 0.282327, 0.282656,
        0.28291 , 0.283091, 0.283197, 0.283229, 0.283187, 0.283072,
        0.282884, 0.282623, 0.28229 , 0.281887, 0.281412, 0.280868,
        0.280255, 0.279574, 0.278826, 0.278012, 0.277134, 0.276194,
        0.275191, 0.274128, 0.273006, 0.271828, 0.270595, 0.269308,
        0.267968, 0.26658 , 0.265145, 0.263663, 0.262138, 0.260571,
        0.258965, 0.257322, 0.255645, 0.253935, 0.252194, 0.250425,
        0.248629, 0.246811, 0.244972, 0.243113, 0.241237, 0.239346,
        0.237441, 0.235526, 0.233603, 0.231674, 0.229739, 0.227802,
        0.225863, 0.223925, 0.221989, 0.220057, 0.21813 , 0.21621 ,
        0.214298, 0.212395, 0.210503, 0.208623, 0.206756, 0.204903,
        0.203063, 0.201239, 0.19943 , 0.197636, 0.19586 , 0.1941  ,
        0.192357, 0.190631, 0.188923, 0.187231, 0.185556, 0.183898,
        0.182256, 0.180629, 0.179019, 0.177423, 0.175841, 0.174274,
        0.172719, 0.171176, 0.169646, 0.168126, 0.166617, 0.165117,
        0.163625, 0.162142, 0.160665, 0.159194, 0.157729, 0.15627 ,
        0.154815, 0.153364, 0.151918, 0.150476, 0.149039, 0.147607,
        0.14618 , 0.144759, 0.143343, 0.141935, 0.140536, 0.139147,
        0.13777 , 0.136408, 0.135066, 0.133743, 0.132444, 0.131172,
        0.129933, 0.128729, 0.127568, 0.126453, 0.125394, 0.124395,
        0.123463, 0.122606, 0.121831, 0.121148, 0.120565, 0.120092,
        0.119738, 0.119512, 0.119423, 0.119483, 0.119699, 0.120081,
        0.120638, 0.12138 , 0.122312, 0.123444, 0.12478 , 0.126326,
        0.128087, 0.130067, 0.132268, 0.134692, 0.137339, 0.14021 ,
        0.143303, 0.146616, 0.150148, 0.153894, 0.157851, 0.162016,
        0.166383, 0.170948, 0.175707, 0.180653, 0.185783, 0.19109 ,
        0.196571, 0.202219, 0.20803 , 0.214   , 0.220124, 0.226397,
        0.232815, 0.239374, 0.24607 , 0.252899, 0.259857, 0.266941,
        0.274149, 0.281477, 0.288921, 0.296479, 0.304148, 0.311925,
        0.319809, 0.327796, 0.335885, 0.344074, 0.35236 , 0.360741,
        0.369214, 0.377779, 0.386433, 0.395174, 0.404001, 0.412913,
        0.421908, 0.430983, 0.440137, 0.449368, 0.458674, 0.468053,
        0.477504, 0.487026, 0.496615, 0.506271, 0.515992, 0.525776,
        0.535621, 0.545524, 0.555484, 0.565498, 0.575563, 0.585678,
        0.595839, 0.606045, 0.616293, 0.626579, 0.636902, 0.647257,
        0.657642, 0.668054, 0.678489, 0.688944, 0.699415, 0.709898,
        0.720391, 0.730889, 0.741388, 0.751884, 0.762373, 0.772852,
        0.783315, 0.79376 , 0.804182, 0.814576, 0.82494 , 0.83527 ,
        0.845561, 0.85581 , 0.866013, 0.876168, 0.886271, 0.89632 ,
        0.906311, 0.916242, 0.926106, 0.935904, 0.945636, 0.9553  ,
        0.964894, 0.974417, 0.983868, 0.993248
    ])

    #y = 100.0 # y goes from 100.1 to 101

    m = len(y)

    x = np.linspace(0, 1, m)

    n = 9 # numerator degree
    d = 5 # denom degree

    Vn = make_vandermonde_matrix(n, x)
    Vd = make_vandermonde_matrix(d, x)

    # still cross-multiplying, we're just not assuming that 
    # constant term in denominator is zero
    A = np.hstack((Vn, -y[:,None] * Vd))

    ATA = A.T @ A

    print('ATA.shape is', ATA.shape)
    print('n + d + 2 =', n + d + 2)

    # TODO: don't use eigh, use svd instead
    #evals, evecs = np.linalg.eigh(ATA)
    #print('evals:', evals)
    #q_alg = evecs[:,0]

    epsilon = 0.1
    
    H = np.hstack((0*Vn, Vd))
    g = np.full(m, epsilon )

    # quadratic programming is rejecting the trivial solution
    # BUT it is not guaranteeing that ||q|| = 1
    #
    # open question: can I transform the QP to add a constraint
    # that ||q|| = 1?
    q_alg = qpsolvers.solve_qp(A.T @ A, -A.T @ y, -H, -g, solver='ecos')

    #q_alg *= 100

    def eval_rational(q):

        pn = q[:n+1]
        pd = q[n+1:] # no longer assuming const. coeff is 1
        
        fn = np.polyval(pn[::-1], x)
        fd = np.polyval(pd[::-1], x)
        f = fn/fd

        return f

    def algebraic_error(q): # <-- proxy for the thing we want to minimize

        # minimizing fn - y*fd

        residual = A @ q - y

        return np.sum(residual ** 2)

    def geometric_error(q): # <--- actually want to minimize

        # literally minimize squared differences between input y and
        # the rational function
        # want: y = fn/fd
        # cross-multiplied: fd*y = fn
        residual = y - eval_rational(q)

        return np.sum(residual ** 2)

    # black box: put in the function to minimize - in this case
    # geometric_error, and we also put in the initial guess - in this
    # case the result of least squares, and it spits out a better solution
    res = scipy.optimize.minimize(geometric_error, q_alg, method='Powell')

    q_geom = res.x

    for q, qtype in [(q_alg, 'alg'), (q_geom, 'geom')]:
        print(f'algebraic error for q_{qtype} is', algebraic_error(q))
        print(f'geometric error for q_{qtype} is', geometric_error(q))
        print()
        
    plt.plot(x, y, label='orig')
    plt.plot(x, eval_rational(q_alg), label='alg')
    plt.plot(x, eval_rational(q_geom), label='geom')

    plt.legend(loc='upper left')
    
    plt.show()

    

main()