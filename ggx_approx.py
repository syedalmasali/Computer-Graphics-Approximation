import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import bivar_tau_test

# port of https://www.shadertoy.com/view/flsyWX to Python

# sliders from shader - feel free to edit these
ROUGHNESS = 0.5
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
    #print("what is the shape", cosTheta)

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

    D = ggx_isotropic_ndf(NoH, alpha)
    
    Vis = ggx_isotropic_visibility(NoV, NoL, alpha)

    F0 = 0.16*REFLECTANCE*REFLECTANCE*(1.0-METALNESS) + linear_albedo*METALNESS

    VoH = np.sum(V*H, axis=-1)
    prefix = (1,) * len(VoH.shape)

    VoH_reshaped = np.expand_dims(VoH, -1)
    cosTheta_clamped = np.maximum(VoH_reshaped, 0.0)
    F = fresnel_schlick(cosTheta_clamped, F0.reshape(prefix + (3,)))

    specular_microfacet = D.reshape(D.shape + (1,)) * Vis.reshape(Vis.shape + (1,)) * F

    diffuse_lambert = (1.0 - METALNESS) * linear_albedo / np.pi

    diffuse_factor = 1.0 - F

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
    chighlight = np.array([1.0, 1.0, 1.0])

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
    assert y.shape == xyshape

    prefix = (1,) * len(xyshape)

    cw = normalize(-ro)
    cp = np.array([0, 1.0, 0])
    cu = normalize(np.cross(cw, cp))
    cv = normalize(np.cross(cu, cw))

    xyz = np.dstack((x, y, np.full(x.shape, 2.0)))
    xyz = normalize(xyz)
    cshape = (1,) * len(xyshape) + (3,)
    xyzshape = xyshape + (1,)
    x = xyz[:,:,0].reshape(xyzshape)
    y = xyz[:,:,1].reshape(xyzshape)
    z = xyz[:,:,2].reshape(xyzshape)

    cu = cu.reshape(cshape)
    cv = cv.reshape(cshape)
    cw = cw.reshape(cshape)

    # the d is rd 
    # the R is the matrix with cu, cv, cw
    # the h vector is cw
    # the l vector is xyz
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
# change to fit the C++ transformation code

def hd_from_lv(L,V):

    h = normalize(L + V)
    # brdf dataset method to convert l,v to h, d    
    elevation_h = np.arccos(h[2])
    azimuth_h = np.arctan2(h[1], h[0])

    # rotation
    a = np.array([[np.cos(azimuth_h) * np.cos(elevation_h)],
                 [np.sin(azimuth_h) * np.cos(elevation_h)],
                 [-np.sin(elevation_h)]])
    
    b = np.array([[-np.sin(azimuth_h)],
                 [np.cos(azimuth_h)],
                 [0]])
    
    c = np.array([[np.cos(azimuth_h) * np.sin(elevation_h)],
                  [np.sin(azimuth_h) * np.sin(elevation_h)],
                  [np.cos(elevation_h)]])  
    
    R = np.hstack((a, b, c))

    d = np.linalg.inv(R) @ L
    

    # a = normalize(np.cross([0,1,0], h))
    # b = normalize(np.cross(h, a))

    # a_dot_L = np.sum(a*L, axis=-1) #work along the last axis
    # b_dot_L = np.sum(b*L, axis=-1)
    # h_dot_L = np.sum(h*L, axis=-1)

    # d = np.stack([a_dot_L, b_dot_L, h_dot_L], axis=-1) #we only care about the last axis

    return h, d

######################################################################
def lv_from_hd(H,D):

    h = H
    a = normalize(np.cross([0,1,0], h))
    b = normalize(np.cross(h, a))
    #h,a,b are 256,256,3

    x, y, z = np.split(D, 3, -1)
    allL = x*a + y*b + z*h

    allV= np.zeros(H.shape)

    h_dot_L = np.sum(h*allL, axis=-1)
    q = h * np.expand_dims(h_dot_L, axis=-1)
    p = allL - q
    allV = q-p
    
    return allL, allV

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

    # plot the shadertoy version
    plt.figure()
    plt.imshow(color, origin='lower')
    plt.title('Port of https://www.shadertoy.com/view/flsyWX')
    plt.axis('off')

def plot_brdf_slice_lv_spherical():

    ######################################################################
        #Below is what the original ggx port had modifief by Zucker
    ######################################################################

    # part 2: plot a slice of the BRDF 
    # we will hold the L vector constant at 45 degrees above the x axis
    azimuth_l = 0.0
    elevation_l = np.pi/4

    # we will vary the view vector direction using linearly spaced vectors
    azimuth_v_rng = np.linspace(0, np.pi, 256)
    elevation_v_rng = np.linspace(0, np.pi/2, 256)

    # make a meshgrid to sample all pairs of cartesian product 
    # of two linearly spaced arrays
    azimuth_v, elevation_v = np.meshgrid(azimuth_v_rng, elevation_v_rng)

    L = from_spherical(elevation_l, azimuth_l)    
    V = from_spherical(elevation_v, azimuth_v)
    
    # N normal vector points straight up along Z axis
    N = np.array([0, 0, 1.0])

    # compute the BRDF for our sampled points
    f = ggx_brdf(N[None, None, :], V, L[None, None, :])

    # brdf has samples in RGB, we want to take average across all 3 channels
    f_avg = f.mean(axis=-1)

    taus = bivar_tau_test.tau_test(azimuth_v, elevation_v, f_avg)

    # plot l,v version of f_avg
    plt.figure()
    plt.pcolormesh(azimuth_v_rng, elevation_v_rng, f_avg)
    plt.title('Average of all channels of GGX brdf')
    plt.xlabel('azimuth v')
    plt.ylabel('elevation v')
    plt.colorbar()

def debug_disney_figure():

    cases= np.array([
        [0,0], 
        [np.pi/4,0], 
        [np.pi/2,0], 
        [0, np.pi/4], 
        [0,np.pi/2], 
        [np.pi/2, np.pi/4], 
        [np.pi/2, np.pi/2]
    ])

    azimuth_h = 0
    azimuth_d = np.pi/2

    for elevation_h, elevation_d in cases:
        print(elevation_h,elevation_d )
        h = from_spherical(elevation_h, azimuth_h)
        d = from_spherical(elevation_d, azimuth_d)
        print('h', h)
        print('d', d)

        l,v = lv_from_hd(h, d)

        print("l", l)
        print("v", v)

        h2, d2 = hd_from_lv(l, v)

        print('h2', h2)
        print('d2', d2)

        assert np.allclose(h, h2)
        assert np.allclose(d, d2)

        print('pass!')


def my_linspace(lo, hi, count):
    u = (np.arange(count) + 0.5) / count
    return lo + u * (hi - lo)

def plot_brdf_slice_hd_spherical():

    elevation_h_rng = my_linspace(0, np.pi/2, 256)
    elevation_d_rng = my_linspace(0, np.pi/2, 256)
    elevation_h, elevation_d = np.meshgrid(elevation_h_rng, elevation_d_rng)

    azimuth_h = np.full_like(elevation_h, 0.0) 
    azimuth_d = np.full_like(elevation_d, np.pi/2)

    h = from_spherical(elevation_h, azimuth_h)
    d = from_spherical(elevation_d, azimuth_d)

    l, v = lv_from_hd(h, d)
    n = np.array([0, 0, 1.0])
    f = ggx_brdf(n[None, None, :], v, l)    # f shape: 256, 256, 3

    f_avg = f.mean(axis=-1)

    # note: we want to fit each channel of f
    # individually, not the tonemapped version
    taus_red = bivar_tau_test.tau_test(elevation_h, elevation_d, f[:, :, 0])
    taus_green = bivar_tau_test.tau_test(elevation_h, elevation_d, f[:, :, 1])
    taus_blue = bivar_tau_test.tau_test(elevation_h, elevation_d, f[:, :, 2])

    # #creating the fitted plot
    # plt.figure()
    # plt.pcolormesh(elevation_h_rng, elevation_d_rng, f_avg)
    # plt.title('Fitted average of all channels of GGX brdf')
    # plt.xlabel('elevation h')
    # plt.ylabel('elevation d')
    # plt.colorbar()

    plt.figure()
    plt.title('GGX brdf')
    # apply reinhard operator
    f_tonemap = f / (1 + f)
    f_tonemap = f_tonemap ** 0.4545
    plt.imshow(f_tonemap, origin='lower')

    # f_green = f[-8, :, 1]
    f_red = f[0, :, 0]
    f_green = f[0, :, 1]
    f_blue = f[0, :, 2]

    f_red = f_red / (1 + f_red)
    f_green = f_green / (1 + f_green)
    f_blue = f_blue / (1 + f_blue)

    # f_red = f_red ** 0.4545
    # f_green = f_green ** 0.4545
    # f_blue = f_blue ** 0.4545

    plt.figure()
    plt.plot(elevation_h_rng, f_green, label = "green")
    plt.plot(elevation_h_rng, f_red, label = "red")
    plt.plot(elevation_h_rng, f_blue, label = "blue")
    plt.legend()


def main():

    # show shadertoy image
    #reproduce_shadertoy_plot()

    # plot f_avg data with l,v axis
    # plot_brdf_slice_lv_spherical()

    # debug_disney_figure()

    # plot original theta_h, theta_d slice image
    plot_brdf_slice_hd_spherical()

    return

    
main()
