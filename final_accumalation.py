####################################################################################
###################### FINAL PROJECT E19 ###########################################
###################### SYED ALI AND MANUEL ESTRADA #################################
###################### COMPUTER GRAPHICS APPRX #####################################
####################################################################################

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import qpsolvers

# sliders from shader - feel free to edit these
ROUGHNESS = .5
METALNESS = 1.0
REFLECTANCE = 1


######################################################################
# normalize a vector or array of vectors
def normalize(a):
    n = np.linalg.norm(a, axis=-1)
    return a / n.reshape(a.shape[:-1] + (1,))


######################################################################
# raytracing function
# determine sphere visibility and distance along ray given ray origin and direction
# ro should be flat array of length 3
# rd should be array whose final dimension is 3 e.g. m-by-3 or m-by-n-by-3
def intersect_sphere(ro, rd):

    assert ro.shape == (3,)
    assert rd.shape[-1] == 3

    orig_shape = rd.shape

    rd = rd.reshape(-1, 3)

    n = rd.shape[0]

    a = np.sum(rd * rd, axis=-1)
    b = 2 * np.sum(rd * ro, axis=-1)
    c = np.sum(ro * ro) - 1

    discriminant = b*b - 4*a*c

    t = np.full(n, np.nan)

    mask = discriminant > 0
    
    tidx, = np.nonzero(mask)

    t[tidx] = (-b[tidx] - np.sqrt(discriminant[tidx])) / (2*a[tidx])

    # return a mask indicating where sphere was visible
    # and an array indicating distance along ray for each pixel
    return mask.reshape(orig_shape[:-1]), t.reshape(orig_shape[:-1])


######################################################################
# NDF from shader
# this is the isotropic variant of the  GGX distribution
# in the book, the equation is undistrubtued but the programmer,
# distributed the the denominator and cancelled with the numerator
def ggx_isotropic_ndf(NoH, alpha):

    a = NoH * alpha
    k = alpha / (1.0 - NoH * NoH + a * a)
    return k * k / np.pi


######################################################################
# visibility function from shader
def ggx_isotropic_visibility(NoV, NoL, alpha):

    a2 = alpha * alpha
    GV = NoL * np.sqrt(NoV * NoV * (1.0 - a2) + a2)
    GL = NoV * np.sqrt(NoL * NoL * (1.0 - a2) + a2)
    
    return 0.5 / (GV + GL)


######################################################################
# schlick approx. to fresnel reflectivity function
# the schlockk apprx we learned from the book
def fresnel_schlick(cosTheta, F0):

    assert len(cosTheta.shape) == len(F0.shape)
    assert F0.shape[-1] == 3

    return F0 + (1.0 - F0) * (1.0 - cosTheta)**5


######################################################################
# BRDF for ggx 
def ggx_brdf(N, V, L):

    assert len(N.shape) == len(V.shape)
    assert len(N.shape) == len(L.shape)

    assert N.shape[-1] == 3
    assert V.shape[-1] == 3
    assert L.shape[-1] == 3
    
    alpha = ROUGHNESS * ROUGHNESS
    albedo = np.array([1, 0.85, 0.57]) # F0 of gold 
    linear_albedo = albedo ** 2.2

    H = normalize(L + V)

    NoH = np.sum(N*H, axis=-1)
    NoV = np.sum(N*V, axis=-1)
    NoL = np.sum(N*L, axis=-1)
    #print('N*H is', N*H.shape)

    D = ggx_isotropic_ndf(NoH, alpha)
    #print('D is', D.shape)
    
    Vis = ggx_isotropic_visibility(NoV, NoL, alpha)
    #print('Vis is', Vis.shape)

    F0 = 0.16*REFLECTANCE*REFLECTANCE*(1.0-METALNESS) + linear_albedo*METALNESS
    #print('F0 is', F0)

    VoH = np.sum(V*H, axis=-1)
    #print("my costheta getting passed in", VoH.reshape(VoH.shape + (1,)))
    prefix = (1,) * len(VoH.shape)

    VoH_reshaped = np.expand_dims(VoH, -1)
    cosTheta_clamped = np.maximum(VoH_reshaped, 0.0)
    F = fresnel_schlick(cosTheta_clamped, F0.reshape(prefix + (3,)))

    #print('F is', F)

    specular_microfacet = D.reshape(D.shape + (1,)) * Vis.reshape(Vis.shape + (1,)) * F

    #print('specular_microfacet is', specular_microfacet.shape)

    diffuse_lambert = (1.0 - METALNESS) * linear_albedo / np.pi
    #print('diffuse_lambert is', diffuse_lambert.shape)

    diffuse_factor = 1.0 - F
    #print('diffuse_factor is', diffuse_factor.shape)

    return diffuse_factor * diffuse_lambert + specular_microfacet


######################################################################
# computes BRDF and adds tonemapping
def ggx_shading(N, V, L):

    NoL = np.sum(N*L, axis=-1)
    
    light_intensity = 2.0

    color = np.maximum(NoL.reshape(NoL.shape + (1,)), 0.0) * light_intensity * ggx_brdf(N, V, L)

    color = color / (color + 1.0) # reinhard tonemapping
    color = color ** 0.4545

    return color
    

######################################################################
# non-photorealistic shading model from chapter 5
def gooch_shading(N, V, L):

    csurface = np.array([0.7, 0.0, 0.7])
    ccool = np.array([0, 0, 0.55]) + 0.25*csurface
    cwarm = np.array([0.3, 0.3, 0]) + 0.25*csurface
    chighlight = np.array[1.0,1.0,1.0]

    t = (np.dot(N, L) + 1.0) / 2.0
    
    r = 2.0 * np.dot(N, L)[:,None] * N - L

    s = np.clip(100.0 * np.sum(r*V, axis=-1) - 97.0, 0.0, 1.0)

    return ( s[:,None] * chighlight[None,:] + 
             (1.0 - s[:,None]) *
             (t[:,None] * cwarm[None,:] + (1.0 - t[:,None])*ccool[None,:]) )


######################################################################
# from shader, for computing ray directions
def compute_ray(x, y, ro):

    xyshape = x.shape
    #print("ori x", xyshape)
    assert y.shape == xyshape

    prefix = (1,) * len(xyshape)

    cw = normalize(-ro)
    cp = np.array([0, 1.0, 0])
    cu = normalize(np.cross(cw, cp))
    cv = normalize(np.cross(cu, cw))

    #print("before ", cu.shape)

    xyz = np.dstack((x, y, np.full(x.shape, 2.0)))
    xyz = normalize(xyz)
    #print("match", xyz)
    #print("original l",xyz)
    cshape = (1,) * len(xyshape) + (3,)
    xyzshape = xyshape + (1,)
    x = xyz[:,:,0].reshape(xyzshape)
    #print("origianl l", x)
    y = xyz[:,:,1].reshape(xyzshape)
    #print("orignal ")
    z = xyz[:,:,2].reshape(xyzshape)

    cu = cu.reshape(cshape)
    cv = cv.reshape(cshape)
    cw = cw.reshape(cshape)

    rd = cu * x + cv * y + cw * z

    return rd 


######################################################################
# 3-vector from lat/lon - note that elevation=0 aligns with positive Z
def from_spherical(elevation, azimuth):

    x = np.cos(azimuth) * np.sin(elevation)
    y = np.sin(azimuth) * np.sin(elevation)
    z = np.cos(elevation)

    return np.stack((x, y, z), axis=-1)
    

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


######################################################################
############# Change of Basis ########################################
def hd_from_lv(L,V):

    h = normalize(L + V)
    a = normalize(np.cross([0,1,0], h))
    b = normalize(np.cross(h, a))

    a_dot_L = np.sum(a*L, axis=-1) #work along the last axis
    b_dot_L = np.sum(b*L, axis=-1)
    h_dot_L = np.sum(h*L, axis=-1)

    d = np.stack([a_dot_L, b_dot_L, h_dot_L], axis=-1) #we only care about the last axis

    return h, d

######################################################################
############# Change of Basis ########################################
def lv_from_hd(H,D):

    h= H
    a = normalize(np.cross([0,1,0], h))
    b = normalize(np.cross(h, a))
    x, y, z = np.split(D, 3, -1)
    allL = x*a + y*b + z*h

    allV= np.zeros(H.shape)

    h_dot_L = np.sum(h*allL, axis=-1)
    q = h * np.expand_dims(h_dot_L, axis=-1)
    p = allL - q
    allV = q-p

    return allL, allV

#########################################################################
########### Function to reproduce the shader toy ########################
def reproduce_shadertoy_plot():
    ######################################################################
    # part 1: make sure we can reproduce the plot from the shadertoy

    # image dimensions
    w = 640
    h = 480

    # normalized pixel coordinates
    xrng = (2*np.arange(w) - w + 0.5) / h
    yrng = (2*np.arange(h) - h + 0.5) / h

    x, y = np.meshgrid(xrng, yrng)

    # ray origin
    ro = np.array([3.0, 0.5, 0])

    # ray directions per pixel
    rd= compute_ray(x, y, ro)

    # raytrace sphere to determine hit points
    mask, t = intersect_sphere(ro, rd)

    pos = ro + rd * t[:,:,None]
    
    # lighting direction
    L = normalize(np.array([-0.5, 1.5, 0.1]))

    # normals for unit sphere are the same as positions (but only for pixels that intersect)
    N = pos[mask]

    # view vector is negative of ray direction
    V = -rd[mask]

    # image buffer of h rows by w cols by 3 (for RGB color)
    color = np.full((h, w, 3), 0.0)

    # set the color for the pixels that intersect
    color[mask] = ggx_shading(N, V, L[None, :])

    # plot it
    plt.figure()
    plt.imshow(color, origin='lower')
    plt.title('Port of https://www.shadertoy.com/view/flsyWX')
    plt.axis('off')


##############################################################
#################### Bivar fitting for F######################
def syed_bivar_fitting(elevation_h, elevation_d, ftemp):

    epsilon= 0.1
    x_samples = elevation_h
    y_samples = elevation_d
    z_samples = ftemp

    #transform to 1D
    x_samples_flat = x_samples.flatten()
    y_samples_flat = y_samples.flatten()
    z_samples_flat = z_samples.flatten()
    m = len(x_samples_flat)

    numer_exponents = bivariate_exponents(7)
    denom_exponents = bivariate_exponents(5)

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

    # make our linear inequality matrix & vector
    H_left = np.zeros((m, k)) # no constraints on numerator, just denominator
    H_right = x_samples_flat[:, None] ** denom_exponents[1:,0] *y_samples_flat[:, None] ** denom_exponents[1:, 1]# Vandermonde matrix for denominator
    H = np.hstack((H_left, H_right))
    g = np.full(m, epsilon - 1.0)

    q = qpsolvers.solve_qp(A.T @ A, -A.T @ z_samples_flat, -H, -g, solver='daqp')

    # evaluate numerator and denominator of our rational function
    f_numer = V_numer @ q[:k]
    f_denom = V_denom @ np.hstack([[1.0], q[k:]])

    #divide numer and denom, then reshape
    recons_samples_flat = f_numer / f_denom
    recons_samples = recons_samples_flat.reshape(z_samples.shape)

    return recons_samples
    
##############################################################
#################### Define the X Range#######################
def my_linspace(lo, hi, count, specificity):
    u = (np.arange(count) + specificity) / count
    return lo + u * (hi - lo)

######################################################################
######Below is what the original ggx port for bisector################
######################################################################
def plot_brdf_slice_hd_spherical():

    # specificity has to be under 1 [1 being at pi/2 = more error]
    specificity= .5
    elevation_h_rng = my_linspace(0, np.pi/2, 256, specificity)
    elevation_d_rng = my_linspace(0, np.pi/2, 256, specificity)

    elevation_h, elevation_d = np.meshgrid(elevation_h_rng, elevation_d_rng)

    azimuth_h = np.full_like(elevation_h, 0.0)

    azimuth_d = np.full_like(elevation_d, np.pi/2)

    h = from_spherical(elevation_h, azimuth_h)

    d = from_spherical(elevation_d, azimuth_d)

    l, v = lv_from_hd(h, d)

    n = np.array([0, 0, 1.0])
    
    f = ggx_brdf(n[None, None, :], v, l)

    fred = syed_bivar_fitting(elevation_h, elevation_d, f[:,:,0]).reshape(256,256)
    fgreen = syed_bivar_fitting(elevation_h, elevation_d, f[:,:,1]).reshape(256,256)
    fblue = syed_bivar_fitting(elevation_h, elevation_d, f[:,:,2]).reshape(256,256)

    f_approx = np.stack((fred, fgreen, fblue),axis=-1)
    f_approx_display = np.maximum(f_approx, 0)
    f_approx_display = f_approx_display / (1 + f_approx_display)
    f_approx_display= f_approx_display ** 0.4545
    #print("the size", f_approx_display.shape)

    ########################################################################
    ####################### Contour Mapping ##################################    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('APPRX GGX BRDF')
    plt.imshow(f_approx_display, origin='lower')
    # apply reinhard operator #We also tried hard clipping form [0,1]
    f_tonemapped = f / (1 + f)
    f_tonemapped = f_tonemapped ** 0.4545
    plt.subplot(1, 2, 2)
    plt.imshow(f_tonemapped, origin='lower')
    plt.title('GGX BRDF')

    ########################################################################
    #######################PLOTTING ERRORS##################################
    ########################################################################
    #Plot comparing the red
    f_red = f[-1, :, 0]  
    f_approx_red = f_approx[-1, :, 0]
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Red Channel')
    plt.plot(f_red, label='f')
    plt.plot(f_approx_red, label='f_approx')
    plt.legend()

    #Plot comparing the green
    f_green = f[-1, :, 1]  
    f_approx_green = f_approx[-1, :, 1]
    #plt.figure()
    plt.subplot(1, 3, 2)
    plt.title('Green Channel')
    plt.plot(f_green, label='f')
    plt.plot(f_approx_green, label='f_approx')
    plt.legend()

    #Plot comparing the blue
    f_blue = f[-1, :, 2]  
    f_approx_blue = f_approx[-1, :, 1]
    #plt.figure()
    plt.subplot(1, 3, 3)
    plt.title('Blue Channel')
    plt.plot(f_blue, label='f')
    plt.plot(f_approx_blue, label='f_approx')
    plt.legend()

    #Plot the MEAN SUMMED error
    f_error = np.sum((f - f_approx)**2, axis=-1)
    f_mserror = f_error/(256*256*3)
    i, j = np.unravel_index(f_mserror.argmax(axis=None), f_mserror.shape)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(f_mserror[4:-4, 4:-4], origin='lower', interpolation='nearest')
    plt.title(f'total MS error is {f_mserror.sum()}')
    plt.colorbar()

    #Plot the SUMMED error
    f_error = np.sum((f - f_approx)**2, axis=-1)
    #print("copy", f_error.shape)
    i, j = np.unravel_index(f_error.argmax(axis=None), f_error.shape)
    
    #plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(f_error[4:-4, 4:-4], origin='lower', interpolation='nearest')
    plt.title(f'total error is {f_error.sum()}')
    plt.colorbar()


#################################################################
############## coeff matrix #####################################
def make_vandermonde_matrix(degree, x):

    exponents = np.arange(degree+1)
    return x[:,None] ** exponents[None,:]


##################################################################
################ compare different types of apprx ################
def compare_methods():

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

    m = len(y)
    x = np.linspace(0, 1, m)

    n = 3 # numerator degree
    d = 2 # denom degree

    Vn = make_vandermonde_matrix(n, x)
    Vd = make_vandermonde_matrix(d, x)

    A = np.hstack((Vn, -y[:,None] * Vd))
    ATA = A.T @ A

    epsilon = 0.1
    H = np.hstack((0*Vn, Vd))
    g = np.full(m, epsilon )

    q_alg = qpsolvers.solve_qp(A.T @ A, -A.T @ y, -H, -g, solver='ecos')

    #####################################################################
    ####################### Insert the coeffcients correctly ############
    def eval_rational(q):
        pn = q[:n+1]
        pd = q[n+1:] # no longer assuming const. coeff is 1
        fn = np.polyval(pn[::-1], x)
        fd = np.polyval(pd[::-1], x)
        f = fn/fd
        return f

    #####################################################################
    ####################### Insert the coeffcients for qp ##############
    def eval_rational2(q):
        pn = q[:n+1]
        pd = np.hstack(([1.0], q[n+1:]))
        fn = np.polyval(pn[::-1], x)
        fd = np.polyval(pd[::-1], x)
        f = fn/fd
        return f

    ######################################################################
    ###################### minimizing fn - y*fd ##########################
    def algebraic_error(q): # <-- proxy for the thing we want to minimize
        residual = A @ q - y
        return np.sum(residual ** 2)

    ################################################################
    ###################### fd*y = fn ###############################
    def geometric_error(q): # <--- actually want to minimize
        residual = y - eval_rational(q)
        return np.sum(residual ** 2)

    res = scipy.optimize.minimize(geometric_error, q_alg, method='Powell')
    q_geom = res.x

    A = np.hstack((Vn, -y[:,None] * Vd[:,1:]))
    q_lstq, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    denom_exponents = np.arange(1, d+1)

    # make our linear inequality matrix & vector
    H_left = np.zeros((m, n+1)) # no constraints on numerator, just denominator
    H_right = x[:, None]**denom_exponents[None,:]
    H = np.hstack((H_left, H_right))
    g = np.full(m, epsilon - 1.0)
    A = np.hstack((Vn, -y[:,None] * Vd[:,1:]))
    ATA = A.T @ A
    q_alg2 = qpsolvers.solve_qp(A.T @ A, -A.T @ y, -H, -g, solver='ecos')

    plt.figure()
    plt.plot(x, y, label='orig')
    plt.plot(x, eval_rational(q_geom), label='gradient descent')
    plt.plot(x, eval_rational2(q_alg2), label='qp with 1 set')
    plt.legend(loc='upper left')


def main():

    reproduce_shadertoy_plot() #to reproduce shadertoy

    plot_brdf_slice_hd_spherical() #to produce approximations

    compare_methods() #to compare gd and qp

    plt.show() #show the graphs

main()