import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import sys
import bivar_tau_test

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0

RGB_SCALE = np.array([RED_SCALE, GREEN_SCALE, BLUE_SCALE])

######################################################################
# returns "dot product" over last dimension of two arrays
# TODO: steal me for ggx_approx program!

def vdot(a, b):
    return np.sum(a*b, axis=-1)

######################################################################
# returns a copy of input array with extra dimension e.g. (m,n) -> (m,n,1)
# TODO: steal me for ggx_approx program!

def expand_last(a):
    return np.expand_dims(a, axis=-1)

######################################################################
# if input is shape (..., k) returns array where each k-vector is
# normalized
# TODO: steal me for ggx_approx program!

def normalize(v):
    return v / expand_last(np.linalg.norm(v, axis=-1))

######################################################################
# normalize or replace with default vector
# TODO: steal me for ggx_approx program!

def normalize_or_else(v, default, tol=1e-15):

    l = np.linalg.norm(v, axis=-1)

    ok = l > tol

    n = np.zeros_like(v)

    n[ok] = v[ok] / expand_last(l[ok])
    
    n[~ok] = default

    return n



######################################################################
# for array of shape baseshape + (k,) returns an array of shape
# basehsape with the last index (up to k-1) provided
#
# e.g. get all z-components of m-by-n-by-3 array of mn vectors by
# calling lastidx(v, 2)
#
# TODO: steal me for ggx_approx program!

def lastidx(v, idx):
    inds = (slice(None),) * (len(v.shape) - 1) + (idx,)
    return v[inds]

######################################################################
# compute a and b basis vectors of rotation matrix from h vector
#
# TODO: steal me for ggx_approx program!

def ab_from_h(h):

    newshape = (1,) * (len(h.shape) - 1) + (3,)

    z = np.array([0, 0, 1.0]).reshape(newshape)
    y = np.array([0, 1.0, 0]).reshape(newshape)

    b = normalize_or_else(np.cross(z, h, axis=-1), y)

    a = normalize(np.cross(b, h, axis=-1))

    return a, b
    

######################################################################
# compute h and d from l and v
#
# TODO: steal me for ggx_approx program!

def hd_from_lv(l, v):

    h = normalize(l + v)

    a, b = ab_from_h(h)

    d = normalize(np.stack((vdot(a, l), vdot(b, l), vdot(h, l)), axis=-1))

    return h, d

######################################################################
# compute l and v from h and d
#
# TODO: steal me for ggx_approx program!

def lv_from_hd(h, d):

    a, b = ab_from_h(h)

    dx = expand_last(lastidx(d, 0))
    dy = expand_last(lastidx(d, 1))
    dz = expand_last(lastidx(d, 2))

    l = normalize(dx*a + dy*b + dz*h)

    h_dot_l = vdot(h, l)
    q = h * expand_last(h_dot_l)
    v = normalize(2*q - l)

    return l, v

######################################################################

def wrap_angle(angle):
    """Wraps a given angle to the [-PI, PI] interval."""
    return np.remainder(angle + np.pi, 2*np.pi) - np.pi

def angle_diff(b, a):
    """Returns the smallest (magnitude) angle c such that
       wrap_angle(a + c) = wrap_angle(b)"""
    return wrap_angle(b-a)

def from_spherical(theta, fi):

    z = np.cos(theta)
    p = np.sin(theta)
    
    x = p * np.cos(fi)
    y = p * np.sin(fi)

    return np.stack((x, y, z), axis=-1)

def rotate_vector(vector, axis, angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return (expand_last(c) * vector +
            expand_last((1  - c) * vdot(axis, vector)) * axis +
            expand_last(s) * np.cross(axis, vector))

def to_spherical(v, normalize):

    if normalize:
        v = v / expand_last(np.linalg.norm(v, axis=-1))

    theta = np.arccos(lastidx(v, 2))
    fi = np.arctan2(lastidx(v, 1), lastidx(v, 0))

    return theta, fi

def half_diff_coords_from_std_coords(theta_in, fi_in, theta_out, fi_out):

    assert len(theta_in.shape) == len(fi_in.shape)
    assert len(theta_in.shape) == len(theta_out.shape)
    assert len(theta_in.shape) == len(fi_out.shape)

    v_in = from_spherical(theta_in, fi_in)
    v_out = from_spherical(theta_out, fi_out)

    v_half = normalize(v_in + v_out)

    theta_half, fi_half = to_spherical(v_half, normalize=False)

    newshape = (1,) * (len(v_half.shape) - 1) + (3,)

    normal = np.array([0, 0, 1.0]).reshape(newshape)
    bi_normal = np.array([0, 1.0, 0]).reshape(newshape)

    v_temp = rotate_vector(v_in, normal, -fi_half)
    v_diff = rotate_vector(v_temp, bi_normal, -theta_half)

    theta_diff, fi_diff = to_spherical(v_diff, normalize=True)

    return theta_half, fi_half, theta_diff, fi_diff

######################################################################

def read_brdf(filename):

    f = open(filename, 'rb')

    buf = f.read(12)

    print('buf:', buf)

    dims = np.frombuffer(buf, dtype=np.int32)

    print('dims:', dims)
    
    expected_dims = (BRDF_SAMPLING_RES_THETA_H, 
                     BRDF_SAMPLING_RES_THETA_D, 
                     BRDF_SAMPLING_RES_PHI_D // 2)

    if np.any(dims != expected_dims):
        raise RuntimeError(f'bad dims in brdf file: expected {expected_dims} but got {tuple(dims)}')

    n = dims.prod()

    brdf_buf = f.read(8 * 3 * n)

    brdf = np.frombuffer(brdf_buf, dtype=np.float64)

    assert brdf.shape == (3*n,)


    return brdf.reshape((3,) + expected_dims)

######################################################################

def test_exhaustive():

    theta_rng = np.linspace(0, np.pi/2, 16, False)
    fi_rng = np.linspace(0, np.pi, 16)

    fi_in, theta_in, fi_out, theta_out = np.meshgrid(fi_rng, theta_rng, 
                                                     fi_rng, theta_rng)

    fi_out += fi_in

    th, fh, td, fd = half_diff_coords_from_std_coords(theta_in, fi_in, 
                                                      theta_out, fi_out)

    l = from_spherical(theta_in, fi_in)
    v = from_spherical(theta_out, fi_out)

    h, d = hd_from_lv(l, v)

    l2, v2 = lv_from_hd(h, d)

    h2 = from_spherical(th, fh)

    d2 = from_spherical(td, fd)
    
    print('l2 should be 0:', np.abs(l2 - l).max())
    print('v2 should be 0:', np.abs(v2 - v).max())

    ok = (th > 1e-12)
    ok2 = (td > 1e-12)

    print('h2 should be 0:', np.abs(h2 - h).max())
    print('d2 should be 0:', np.abs(d2[ok] - d[ok]).max())

    th2, fh2 = to_spherical(h, normalize=True)
    td2, fd2 = to_spherical(d, normalize=True)
    
    print('th2 should be 0:', np.abs(th - th2).max())
    print('fh2 should be 0:', np.abs(angle_diff(fh, fh2)).max())

    print('td2 should be 0:', np.abs(td[ok] - td2[ok]).max())
    print('fd2 should be 0:', np.abs(angle_diff(fd[ok & ok2], fd2[ok & ok2])).max())


######################################################################

def theta_half_index(theta_half):

    theta_half = np.where(theta_half < 0.0, 0.0, theta_half)

    theta_half_deg = (theta_half / (np.pi/2.0)) * BRDF_SAMPLING_RES_THETA_H
    temp = theta_half_deg * BRDF_SAMPLING_RES_THETA_H
    temp = np.sqrt(temp)

    rval = np.int32(temp)

    rval = np.clip(rval, 0, BRDF_SAMPLING_RES_THETA_H-1)

    return rval

######################################################################

def theta_half_from_index(i):

    i = np.clip(i, 0, BRDF_SAMPLING_RES_THETA_H-1)

    x = i + 0.5
    x *= x

    x /= BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_H

    return x * np.pi/2

######################################################################

def theta_diff_index(theta_diff):

    i = np.int32( (theta_diff / (np.pi * 0.5)) * BRDF_SAMPLING_RES_THETA_D )

    return np.clip(i, 0, BRDF_SAMPLING_RES_THETA_D-1)

######################################################################

def theta_diff_from_index(i):

    return (i + 0.5) * 0.5 * np.pi / BRDF_SAMPLING_RES_THETA_D

######################################################################

def test_indexing():

    thi = np.arange(90)
    th = theta_half_from_index(thi)
    thi2 = theta_half_index(th)

    tdi = np.arange(90)
    td = theta_diff_from_index(tdi)
    tdi2 = theta_diff_index(td)

    print('thi ok:', np.all(thi == thi2))
    print('tdi ok:', np.all(tdi == tdi2))

######################################################################
# returns 5 numpy arrays:
#
#   - th_samples is the true theta_half coordinates in radians (size matches rgb) 
#   - td_samples is the true theta_diff coordinates in radians
#   - th_display are theta_half coordinates for displaying a color plot (one extra row/col)
#   - td_display are theta_diff coordinates for displaying a color plot
#   - rgb is the actual BRDF data in rgb channels 
#
# where rgb[i,j,k] corresponds to measured data at td_samples[i], th_samples[j], 
# and k being 0=red, 1=green, 2=blue
#
# note data is not uniformly sampled along theta_half axis and hence
# td_samples and th_display are not linearly spaced.
#
# see disney paper "These correspond to 1 degree increments except for
# the θh axis which was warped to concentrate data samples near the
# specular peak"

def get_slice(filename):

    brdf = read_brdf(filename)

    rgb_unscaled = np.moveaxis(brdf[:, :, :, 90], (0, 1, 2), (2, 1, 0))

    rgb = rgb_unscaled * RGB_SCALE

    # radians
    th_samples = theta_half_from_index(np.arange(90))
    td_samples = theta_diff_from_index(np.arange(90))

    # degrees
    th_display = theta_half_from_index(np.arange(91) - 0.5) * 180 / np.pi
    td_display = theta_diff_from_index(np.arange(91) - 0.5) * 180 / np.pi

    #th_display, td_display = np.meshgrid(th_display, td_display)

    return th_samples, td_samples, th_display, td_display, rgb 


######################################################################

def main():

    if len(sys.argv) == 1:

        test_exhaustive()
        test_indexing()

    else:

        filename = sys.argv[1]

        th_samples, td_samples, th_display, td_display, rgb = get_slice(filename)

        # TODO: fit rgb_samples to meshgrid(th_samples, td_samples) using bivariate fit!
        # note you would either take the average across r,g,b channels
        # or perform independent fits for r, g, and b

        # plot using non-uniform scaling on x-axis
        #plt.pcolormesh(th_display, td_display, rgb_display)

        # plot without non-uniform scaling
        #plt.pcolormesh(rgb_display)
        #plt.gca().set_aspect('equal')
        #plt.show()

        # bivar_x, bivar_y = np.meshgrid(th_samples, td_samples)

        # rgb_fit = np.empty_like(rgb)

        # tau = 0.1
        # constant_is_one = True

        # for channel in range(3):

        #     Vn, Vd, pn, pd, H = bivar_tau_test.quad_prog(
        #         tau,
        #         bivar_x, 
        #         bivar_y, 
        #         rgb[:, :, channel],
        #         constant_is_one)

        #     # note that rgb should always be >= 0
        #     # (not always true for invalid slices in the data)
        #     # so we can take a log without domain errors
        #     pn, pd, _, rgb_chan_fit = bivar_tau_test.optimize_more(
        #         tau, bivar_x, bivar_y, np.log(np.maximum(rgb[:, :, channel], 0)),
        #         pn, pd, Vn, Vd, H, 
        #         constant_is_one)

        #     # since we use exp here, the output is guaranteed to be positive
        #     rgb_fit[:, :, channel] = np.exp(rgb_chan_fit)

        # rgb_fit_display = np.clip(rgb_fit, 0, 1) ** 0.4545

        # fig, ax = plt.subplots(2)

        # ax[0].pcolormesh(th_display, td_display, rgb_display)
        # ax[0].set_aspect('equal')

        # ax[1].pcolormesh(th_display, td_display, rgb_fit_display)
        # ax[1].set_aspect('equal')

        # plt.show()

        # original data
        # clip to [0, 1] and then gamma correct for display
        rgb_display = np.clip(rgb, 0, 1) ** 0.4545

        bivar_x, bivar_y = np.meshgrid(th_samples, td_samples)

        const_rgb = np.empty_like(rgb)
        var_rgb = np.empty_like(rgb)

        # fitting each channel
        for channel in range(3):
            num = 5
            den = 3
            print(f"Fitting channel {channel}")
            each_fits_const, each_fits_var = bivar_tau_test.tau_test(num, den, bivar_x, bivar_y, rgb[:, :, channel])
            const_rgb[:, :, channel] = each_fits_const
            var_rgb[:, :, channel] = each_fits_var

        const_rgb_display = np.clip(const_rgb, 0, 1) ** 0.4545
        var_rgb_display = np.clip(var_rgb, 0, 1) ** 0.4545



        num = 5
        den = 3
        # fitting average
        rgb_avg = rgb.mean(axis = -1)
        print("fitting the average")
        avg_fits_const, avg_fits_var = bivar_tau_test.tau_test(num, den, bivar_x, bivar_y, rgb_avg)

        # avg fitting display
        avg_const_rgb_display = np.clip(avg_fits_const, 0, 1) ** 0.4545
        avg_var_rgb_display = np.clip(avg_fits_var, 0, 1) ** 0.4545

        # plotting raw
        fig1, ax1 = plt.subplots(1, 2)

        th_raw, td_raw = np.meshgrid(np.arange(0, 91), np.arange(0, 91))

        plt.suptitle(f'slice for {filename} at phi_d = 90 deg')
        ax1[0].pcolormesh(th_raw, td_raw, rgb_display)
        ax1[0].set_aspect('equal')
        ax1[0].set_title('as measured/stored')

        ax1[1].pcolormesh(th_display, td_display, rgb_display)
        ax1[1].set_aspect('equal')
        ax1[1].set_title('unwarped (true angles)')

        # plotting each channel fitting
        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].pcolormesh(th_display, td_display, const_rgb_display)
        ax2[0].set_aspect("equal")
        ax2[0].set_title("b0=1 each channel")

        ax2[1].pcolormesh(th_display, td_display, var_rgb_display)
        ax2[1].set_aspect("equal")
        ax2[1].set_title("b0 variable each channel")

        # plotting avg channels
        fig3, ax3 = plt.subplots(1, 2)
        ax3[0].pcolormesh(th_display, td_display, avg_const_rgb_display)
        ax3[0].set_aspect("equal")
        ax3[0].set_title("b0=1 with f_avg")

        ax3[1].pcolormesh(th_display, td_display, avg_var_rgb_display)
        ax3[1].set_aspect("equal")
        ax3[1].set_title("b0 variable with f_avg")
        
        # plot 3d
        fig = plt.figure()
        ax4 = fig.add_subplot(1, 2, 1, projection = "3d")
        sc = ax4.scatter(bivar_x, bivar_y, rgb_avg, c=rgb_avg, s=3)
        ax4.set_title("original")
        ax4.set_xlabel("theta h")
        ax4.set_ylabel("theta d")

        ax5 = fig.add_subplot(1, 2, 2, project = "3d")
        sc = ax5.scatter(bivar_x, bivar_y, avg_fits_const, avg_fits_const)
        ax5.set_title("avg_const")
        ax5.set_xlabel("theta h")
        ax5.set_ylabel("theta d")

        plt.colorbar(sc)

        plt.show()

######################################################################

if __name__ == '__main__':

    main()
