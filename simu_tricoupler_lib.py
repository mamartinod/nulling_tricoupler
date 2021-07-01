#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:26:34 2021

@author: mam
"""
import numpy as np
import cupy as cp
from itertools import combinations
import h5py
import matplotlib.pyplot as plt


def atmo_screen(isz, ll, r0, L0, fc=19.5, correc=1.0, pdiam=None, seed=None):
    ''' -----------------------------------------------------------
    The Kolmogorov - Von Karman phase screen generation algorithm.
    Adapted from the work of Carbillet & Riccardi (2010).
    http://cdsads.u-strasbg.fr/abs/2010ApOpt..49G..47C
    Kolmogorov screen can be altered by an attenuation of the power
    by a correction factor *correc* up to a cut-off frequency *fc*
    expressed in number of cycles across the phase screen
    Parameters:
    ----------
    - isz    : the size of the array to be computed (in pixels)
    - ll     :  the physical extent of the phase screen (in meters)
    - r0     : the Fried parameter, measured at a given wavelength (in meters)
    - L0     : the outer scale parameter (in meters)
    - fc     : DM cutoff frequency (in lambda/D)
    - correc : correction of wavefront amplitude (factor 10, 100, ...)
    - pdiam  : pupil diameter (in meters)
    Returns: two independent phase screens, available in the real and
    imaginary part of the returned array.
    Remarks:
    -------
    If pdiam is not specified, the code assumes that the diameter of
    the pupil is equal to the extent of the phase screen "ll".
    ----------------------------------------------------------- '''

    if not seed is None:
        np.random.seed(seed)

    phs = 2*np.pi * (np.random.rand(isz, isz) - 0.5)

    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    rr = np.hypot(yy, xx)
    rr = np.fft.fftshift(rr)
    rr[0, 0] = 1.0

    modul = (rr**2 + (ll/L0)**2)**(-11/12.)

    if pdiam is not None:
        in_fc = (rr < fc * ll / pdiam)
    else:
        in_fc = (rr < fc)

    modul[in_fc] /= correc

    screen = np.fft.ifft2(modul * np.exp(1j*phs)) * isz**2
    screen *= np.sqrt(2*0.0228)*(ll/r0)**(5/6.)

    screen -= screen.mean()
    return(screen)


def generatePhaseScreen(wavel_r0, isz, ll, r0, L0, fc=19.5, correc=1., pdiam=None, seed=None):
    """
    Generate phase screen in meter.

    :param wavel_r0: DESCRIPTION
    :type wavel_r0: TYPE
    :param isz: DESCRIPTION
    :type isz: TYPE
    :param ll: DESCRIPTION
    :type ll: TYPE
    :param r0: DESCRIPTION
    :type r0: TYPE
    :param L0: DESCRIPTION
    :type L0: TYPE
    :param fc: DESCRIPTION, defaults to 19.5
    :type fc: TYPE, optional
    :param correc: DESCRIPTION, defaults to 1.
    :type correc: TYPE, optional
    :param pdiam: DESCRIPTION, defaults to None
    :type pdiam: TYPE, optional
    :param seed: DESCRIPTION, defaults to None
    :type seed: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    phs_screen = atmo_screen(isz, ll, r0, L0, fc=fc,
                             correc=correc, pdiam=pdiam, seed=seed)
    phs_screen = phs_screen.real * wavel_r0 / (2*np.pi)
    return phs_screen


def movePhaseScreen(phase_screens, v_wind, angle_wind, time, meter2pixel):
    '''
    Parameters
    ----------
    phase_screen : array
        phase screen to move
    v_wind : float
        Speed of the wind in m/s
    angle_wind : float
        Orientation of the wind in degree.
    time : float
        time spent since the creation of the phase screen
    meter2pixel : float
        conversion factor from meter to pixel

    Returns
    -------
    moved_phase_screen : array
        Moved phase screen.
    '''

    # phase_screens = cp.array(phase_screens, dtype=cp.float32)
    yshift_in_pix = int(
        np.around(v_wind * time * meter2pixel * np.sin(np.radians(angle_wind))))
    xshift_in_pix = int(
        np.around(v_wind * time * meter2pixel * np.cos(np.radians(angle_wind))))

    return cp.roll(phase_screens, (yshift_in_pix, xshift_in_pix), axis=(-2, -1)), (xshift_in_pix, yshift_in_pix)


def uniform_disk(ysz, xsz, radius, rebin, between_pix=False, norm=False):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a uniform disk of radius "radius".
    ---------------------------------------------------------  '''
    xsz2 = xsz * rebin
    ysz2 = ysz * rebin
    radius2 = radius * rebin

    if between_pix is False:
        xx, yy = np.meshgrid(np.arange(xsz2)-xsz2//2, np.arange(ysz2)-ysz2//2)
    else:
        xx, yy = np.meshgrid(np.arange(xsz2)-xsz2//2+0.5,
                             np.arange(ysz2)-ysz2//2+0.5)
    mydist = np.hypot(yy, xx)
    res = np.zeros_like(mydist)
    res[mydist <= radius2] = 1.0
    res = np.reshape(res, (ysz, rebin, xsz, rebin))
    res = res.mean(3).mean(1)
    if norm:
        res = res / (np.sum(res))**0.5

    return(res)


def createSubPupil(sz, sub_rad, baseline, rebin, between_pix=False, norm=False):
    global center1, center2, sub_pupil
    pupil = np.zeros((sz, sz))
    sub_pupil = uniform_disk(sub_rad*2, sub_rad*2,
                             sub_rad, rebin, between_pix, norm)
    center1 = int(sz // 2 - baseline // 2)
    center2 = int(sz // 2 + baseline // 2)
    pupil[sz//2-sub_rad:sz//2+sub_rad, center1 -
          sub_rad:center1+sub_rad] = sub_pupil
    pupil[sz//2-sub_rad:sz//2+sub_rad, center2 -
          sub_rad:center2+sub_rad] = sub_pupil
    return pupil


def calculateInjection(phs_screen, wl, geo_inj=0.8):
    rms = cp.std(phs_screen)
    strehl = cp.exp(-(2*np.pi/wl)**2 * rms**2)
    injections = geo_inj * strehl
    return injections


def binning(arr, binning, axis=0, avg=False):
    """
    Bin frames together

    :Parameters:
        **arr**: nd-array
            Array containing data to bin
        **binning**: int
            Number of frames to bin
        **axis**: int
            axis along which the frames are
        **avg**: bol
            If ``True``, the method returns the average of the binned frame.
            Otherwise, return its sum.

    :Attributes:
        Change the attributes

        **data**: ndarray
            datacube
    """
    if binning is None:
        binning = arr.shape[axis]

    shape = arr.shape
    # Number of frames which can be binned respect to the input value
    crop = shape[axis]//binning*binning
    arr = np.take(arr, np.arange(crop), axis=axis)
    shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    shape = shape[:axis] + (-1, binning) + shape[axis+1:]
    arr = arr.reshape(shape)
    if not avg:
        arr = arr.sum(axis=axis+1)
    else:
        arr = arr.mean(axis=axis+1)

    return arr


def addPhotonNoise(data, QE, enf=None):
    if enf is None:
        noisy_data = cp.random.poisson(
            data*QE, size=data.shape, dtype=cp.float32)
    else:
        noisy_data = cp.random.poisson(
            data*QE*enf, size=data.shape, dtype=cp.float32)

    return noisy_data


def _addRON(data, ron, offset):
    ron_noise = cp.random.normal(offset, ron, data.shape)
    noisy_data = data + ron_noise
    return noisy_data


def _addDark(ndark, fps, shape, enf=None):
    dark_electrons = ndark / fps
    if enf is None:
        noisy_data = cp.random.poisson(
            dark_electrons, size=shape, dtype=cp.float32)
    else:
        noisy_data = cp.random.poisson(
            dark_electrons*enf, size=shape, dtype=cp.float32)

    return noisy_data


def addDetectorNoise(data, ron, ndark, fps, enf=None):
    noisy = data + _addDark(ndark, fps, data.shape, enf)
    noisy = noisy + _addRON(noisy, ron, 0)
    return noisy


def addNoise(data, QE, read_noise, ndark, fps, activate_photon_noise, activate_detector_noise, enf=None):
    if activate_photon_noise:
        if activate_detector_noise:
            noisy = addPhotonNoise(data, QE, enf)
        else:
            noisy = addPhotonNoise(data, 1, enf)
    else:
        noisy = data

    if activate_detector_noise:
        if activate_photon_noise:
            noisy = addDetectorNoise(noisy, read_noise, ndark, fps, enf)
        else:
            noisy = addDetectorNoise(noisy*QE, read_noise, ndark, fps, enf)

    return noisy


def measurePhase3(data, wl):
    di = data[2] - data[0]
    denom = (data[0]+data[2]-2*data[1])
    di2 = di / denom
    dphi = cp.arctan(-3**0.5*di2)

    pij = cp.diff(dphi) / cp.array([(2*np.pi*(1/wl[i+1] - 1/wl[i]))
                                    for i in range(wl.size-1)], dtype=cp.float32)
    ddphi = cp.abs(cp.diff(dphi))

    weight = cp.zeros_like(ddphi)
    weight[ddphi < 2*np.pi/3] = 1.

    ddm = cp.sum(weight * pij, 0) / cp.sum(weight, 0)

    return cp.array([ddm], dtype=cp.float32)


def get_tricombiner(t_coeff, c_coeff, s1, s2):
    """Calculate the transfer matrix of the photonic combiner.

    DEPRECATED !!!

    :param t_coeff: transmission coefficient for several wavelengths
    :type t_coeff: array
    :param c_coeff: coupling coefficient for several wavelengths
    :type c_coeff: array
    :param s1: splitting coefficient of beam 1 for several wavelengths
    :type s1: array
    :param s2: splitting coefficient of beam 1 for several wavelengths
    :type s2: array
    :return combiner_tri: Spectral matrix of the combiner
    :rtype: 3D-array

    """
    tri_splitter = get_splitter(s1, s2)
    tricoupler = get_tricoupler(t_coeff, c_coeff)
    combiner_tri = cp.einsum('ijk,jlk->ilk', tricoupler, tri_splitter)

    return combiner_tri


def get_tricombiner2(t1, t2, c1, c2, s1, s2):
    """Calculate the transfer matrix of the photonic combiner.

    :param t1: transmission coefficient left to left for several wavelengths
    :type t1: array
    :param t2: transmission coefficient centre to centrefor several wavelengths
    :type t2: array
    :param c1: coupling coefficient left to centre for several wavelengths
    :type c1: array
    :param c2: coupling coefficient left to right for several wavelengths
    :type c2: array
    :param s1: splitting coefficient of beam 1 for several wavelengths
    :type s1: array
    :param s2: splitting coefficient of beam 1 for several wavelengths
    :type s2: array
    :return combiner_tri: Spectral matrix of the combiner
    :rtype: 3D-array


    """
    tri_splitter = get_splitter(s1, s2)
    tricoupler = get_tricoupler2(t1, t2, c1, c2)
    combiner_tri = cp.einsum('ijk,jlk->ilk', tricoupler, tri_splitter)

    return combiner_tri


def get_splitter(s1, s2):
    zeros = np.zeros_like(s1)
    splitter = np.array([[1-s1, zeros],
                         [zeros, 1-s2],
                         [s1, zeros],
                         [zeros,  s2]])

    splitter = splitter**0.5
    return cp.asarray(splitter, dtype=cp.float32)


def get_tricoupler(t_coeff, c_coeff):
    """Get transfer matrix of a tricoupler.

    DEPRECATED

    Transfer matrix of a tricoupler. Center row is the null output,
    phase and antinull is recovered from linear combinations of
    the three outputs (rows)

    Structure:
        1st row = left output
        2st row = null output
        3rd row = right output
        4th row = photometric output A
        5th row = photometric output B
    """
    phi_tc = np.arccos(-c_coeff / (2 * t_coeff))
    phi_c = 0.
    ones = np.ones_like(phi_tc)
    z = np.zeros_like(phi_tc)
    tricoupler = np.array([[t_coeff * np.exp(1j*phi_tc), c_coeff, z, z],
                           [c_coeff, c_coeff, z, z],
                           [c_coeff, t_coeff * np.exp(1j*phi_tc), z, z],
                           [z, z, ones, z],
                           [z, z, z, ones]], dtype=np.complex64)
    tricoupler[:3, :2] *= np.exp(1j * phi_c)
    tricoupler = cp.asarray(tricoupler)

    return tricoupler


def get_tricoupler2(t1, t2, c1, c2):
    """Get transfer matrix of a tricoupler.

    Transfer matrix of a tricoupler. Center row is the null output,
    phase and antinull is recovered from linear combinations of
    the three outputs (rows)

    Structure:
        1st row = left output
        2st row = null output
        3rd row = right output
        4th row = photometric output A
        5th row = photometric output B
    """
    phi_c1 = 0.
    z2 = 1/2 * (2*c2 - c1**2/c2 - 1j * (4*t1**2*c2**2 - c1**4)**0.5/(c2))
    dphi2 = np.arccos(c1/(2*abs(z2))) - np.angle(z2)
    dphi1 = dphi2 + np.arccos(-c1**2/(2*t1*c2))

    zeros = np.zeros_like(t1)
    ones = np.ones_like(t1)

    t1_coeff = t1*np.exp(1j*dphi1)
    c2_coeff = c2*np.exp(1j*dphi2)
    tricoupler = np.array([[t1_coeff, c2_coeff, zeros, zeros],
                           [c1, c1, zeros, zeros],
                           [c2_coeff, t1_coeff, zeros, zeros],
                           [zeros, zeros, ones, zeros],
                           [zeros, zeros, zeros, ones]],
                          dtype=np.complex64)
    tricoupler[:3, :2] *= np.exp(1j*phi_c1)
    tricoupler = cp.asarray(tricoupler)

    return tricoupler


def get_tricoupler_coeffs(path_left, path_center, path_right, wl):
    """Load coefficients of the tricoupler from 1-by-1 injection.

    DEPRECATED

    :param path_left: DESCRIPTION
    :type path_left: TYPE
    :param path_center: DESCRIPTION
    :type path_center: TYPE
    :param path_right: DESCRIPTION
    :type path_right: TYPE
    :param wl: DESCRIPTION
    :type wl: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    l_out = np.transpose(np.loadtxt(path_left))
    c_out = np.transpose(np.loadtxt(path_center))
    r_out = np.transpose(np.loadtxt(path_right))

    wl0 = np.array(l_out[0])*1e-6
    t_coeff = l_out[1]**0.5
    c_coeff_c = c_out[1]**0.5
    c_coeff_r = r_out[1]**0.5

    # taking an average FOR NOW; figure out what to do more specifically later
    c_coeff = np.mean(np.array([c_coeff_c, c_coeff_r]), axis=0)

    t_coeff_interp = np.interp(wl, wl0, t_coeff)
    c_coeff_interp = np.interp(wl, wl0, c_coeff)

    return t_coeff_interp, c_coeff_interp


def get_tricoupler_coeffs2(path_inj_left, path_inj_centre, path_inj_right, wl):
    """Load coefficients of the tricoupler from 1-by-1 injection.

    :param path_inj_left: tuple of files of coefficients
                            when injection in left input
    :type path_inj_left: 3-tuple
    :param path_inj_centre: tuple of files of coefficients
                            when injection in centre input
    :type path_inj_centre: 3-tuple
    :param path_inj_right: tuple of files of coefficients
                            when injection in right input
    :type path_inj_right: 3-tuple
    :param wl: DESCRIPTION
    :type wl: array
    :return t1_interp: interpolated values of the coefficient left-left
    :rtype: array
    :return t2_interp: interpolated values of the coefficient centre-centre
    :rtype: array
    :return c1_interp: interpolated values of the coefficient left-centre
    :rtype: array
    :return c2_interp: interpolated values of the coefficient left-right
    :rtype: array

    """
    inj_left = [np.transpose(np.loadtxt(elt)) for elt in path_inj_left]
    inj_centre = [np.transpose(np.loadtxt(elt)) for elt in path_inj_centre]
    inj_right = [np.transpose(np.loadtxt(elt)) for elt in path_inj_right]

    wl0 = np.array(inj_left[0][0])*1e-6
    t1 = inj_left[0][1]**0.5
    t2 = inj_centre[1][1]**0.5
    c1 = inj_left[1][1]**0.5
    c2 = inj_left[2][1]**0.5

    t1_interp = np.interp(wl, wl0, t1)
    t2_interp = np.interp(wl, wl0, t2)
    c1_interp = np.interp(wl, wl0, c1)
    c2_interp = np.interp(wl, wl0, c2)

    return t1_interp, t2_interp, c1_interp, c2_interp


def get_bicoupler_coeffs(zeta_path, wl):
    """Calculate chromatic coefficients for directional coupler from\
        real data.

    :param zeta_path: contains current data from GLINT (4 beams, 6 baselines)
    :type zeta_path: str
    :param wl: waelength to which interpolate the zeta coefficients
    :type wl: 1D-array
    :return: splitting coefficients of beams 1 and 2, coupling coefficients
        of the coupler
    :rtype: 4-tuple

    """
    zeta_file = h5py.File(zeta_path, 'r')

    # null/antinull outputs for beams 1 and 2 (zeta coefficients)
    # Conversion in numpy array is mandatory to do calculations with the values
    zeta_b1_n1 = np.array(zeta_file['b1null1'])
    zeta_b1_an1 = np.array(zeta_file['b1null7'])
    zeta_b1_n3 = np.array(zeta_file['b1null3'])
    zeta_b1_an3 = np.array(zeta_file['b1null9'])
    zeta_b1_n5 = np.array(zeta_file['b1null5'])
    zeta_b1_an5 = np.array(zeta_file['b1null11'])

    # Conversion in numpy array is mandatory to do calculations with the values
    zeta_b2_n1 = np.array(zeta_file['b2null1'])
    zeta_b2_an1 = np.array(zeta_file['b2null7'])
    zeta_b2_n2 = np.array(zeta_file['b2null2'])
    zeta_b2_an2 = np.array(zeta_file['b2null8'])
    zeta_b2_n6 = np.array(zeta_file['b2null6'])
    zeta_b2_an6 = np.array(zeta_file['b2null12'])

    wl0 = np.array(zeta_file['wl_scale'])  # wavelength scale
    zeta_file.close()

    # splitting ratio for beam 1 (into coupler 1)
    alpha_b1 = (zeta_b1_n1 + zeta_b1_an1) / (1 + (zeta_b1_n1 + zeta_b1_an1) +
                                             (zeta_b1_n3 + zeta_b1_an3) +
                                             (zeta_b1_n5 + zeta_b1_an5))

    # first splitting ratio for beam 2 (into coupler 1)
    alpha_b2 = (zeta_b2_n1 + zeta_b2_an1) / (1 + (zeta_b2_n1 + zeta_b2_an1) +
                                             (zeta_b2_n2 + zeta_b2_an2) +
                                             (zeta_b2_n6 + zeta_b2_an6))

    # Coupling coefficients inside the coupler 1
    kappa_12 = (zeta_b1_an1 / zeta_b1_n1) / (1 + (zeta_b1_an1 / zeta_b1_n1))
    kappa_21 = (zeta_b2_n1 / zeta_b2_an1) / (1 + (zeta_b2_n1 / zeta_b2_an1))

    """
    Wavelength scale; note we cut off the highest and lowest wavelengths
    as zeta coeffs become messy there
    """
    within = ((wl0 < 1650) & (wl0 > 1350)
              )  # central wavelength 1550 +/- 100 nm
    wl0 = np.array(wl0[within] * 1e-9)

    alpha_b1 = alpha_b1[within]
    alpha_b2 = alpha_b2[within]
    kappa_12 = kappa_12[within]
    kappa_21 = kappa_21[within]

    alpha_b1_interp = np.interp(wl, wl0, alpha_b1)
    alpha_b2_interp = np.interp(wl, wl0, alpha_b2)
    kappa_12_interp = np.interp(wl, wl0, kappa_12)
    kappa_21_interp = np.interp(wl, wl0, kappa_21)

    return alpha_b1_interp, alpha_b2_interp, kappa_12_interp, kappa_21_interp


def get_bicoupler(kappa_12, kappa_21):
    """Get transfer matrix of a directional coupler.

    Transfer matrix of a directional coupler.

    Structure:
        1st row = left output
        2st row = right output
        3rd row = fake output for compatibility in the simulation
        4th row = photometric output A
        5th row = photometric output B
    """
    z = np.zeros(kappa_12.shape)
    ones = np.ones(kappa_12.shape)
    bicoupler = np.array([[(1-kappa_12)**0.5,
                           kappa_12**0.5 * np.exp(-1j * np.pi/2), z, z],
                          [kappa_21**0.5 *
                           np.exp(-1j * np.pi/2), (1-kappa_21)**0.5, z, z],
                          [z, z, z, z],
                          [z, z, ones, z],
                          [z, z, z, ones]], dtype=np.complex64)
    bicoupler = cp.asarray(bicoupler)
    return bicoupler


def get_bicombiner(alpha_b1, alpha_b2, kappa_12, kappa_21):
    bi_splitter = get_splitter(alpha_b1, alpha_b2)
    bicoupler = get_bicoupler(kappa_12, kappa_21)
    bicombiner = cp.einsum('ijk,jlk->ilk', bicoupler, bi_splitter)

    return bicombiner


if __name__ == "__main__":
    path_inj_left = ['3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_1_last.dat',
                     '3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_2_last.dat',
                     '3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_3_last.dat']
    path_inj_centre = ['3DTriRatioCplLen1700Wvl14-17_Centre_bp_mon_1_last.dat',
                       '3DTriRatioCplLen1700Wvl14-17_Centre_bp_mon_2_last.dat',
                       '3DTriRatioCplLen1700Wvl14-17_Centre_bp_mon_3_last.dat']
    path_inj_right = ['3DTriRatioCplLen1700Wvl14-17_Right_bp_mon_1_last.dat',
                      '3DTriRatioCplLen1700Wvl14-17_Right_bp_mon_2_last.dat',
                      '3DTriRatioCplLen1700Wvl14-17_Right_bp_mon_3_last.dat']

    wavel = 1.6e-6  # Wavelength of observation (in meter)
    # Bandwidth around the wavelength of observation (in meter)
    bandwidth = 0.2e-6
    dwl = 5e-9  # Width of one spectral channel (in meter)

    wl = np.arange(wavel-bandwidth/2, wavel+bandwidth/2, dwl)
    coeff_tri = 1/4. * np.ones(wl.shape)

    t1, t2, c1, c2 = \
        get_tricoupler_coeffs2(
            path_inj_left, path_inj_centre, path_inj_right, wl)

    # t1 = t2 = c1 = c2 = 1/3**0.5 * np.ones(wl.shape)

    tricoupler = get_tricombiner(t1, c1, coeff_tri, coeff_tri)
    tricoupler2 = get_tricombiner2(t1, t2, c1, c2, coeff_tri, coeff_tri)

    phase = np.linspace(0, 2*np.pi, 1000, False)
    i_out = []
    i_out2 = []
    for p in phase:
        a_in = cp.array([np.exp(1j*p)*np.ones_like(wl), np.ones_like(wl)])

        a_out = cp.einsum('ijk,jk->ik', tricoupler, a_in)
        i_out.append(abs(cp.asnumpy(a_out))**2)

        a_out2 = cp.einsum('ijk,jk->ik', tricoupler2, a_in)
        i_out2.append(abs(cp.asnumpy(a_out2))**2)

    i_out = np.array(i_out)
    i_out2 = np.array(i_out2)

    plt.figure()
    plt.plot(phase, i_out[:, 0, 10])
    plt.plot(phase, i_out[:, 1, 10])
    plt.plot(phase, i_out[:, 2, 10])
    plt.grid()
    plt.figure()
    plt.plot(phase, i_out2[:, 0, 10])
    plt.plot(phase, i_out2[:, 1, 10])
    plt.plot(phase, i_out2[:, 2, 10])
    plt.grid()
