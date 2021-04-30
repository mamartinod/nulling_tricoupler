#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:26:34 2021

@author: mam
"""
import numpy as np
# import cupy as cp
from itertools import combinations

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
    phs_screen = atmo_screen(isz, ll, r0, L0, fc=fc, correc=correc, pdiam=pdiam, seed=seed)
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
    yshift_in_pix = int(np.around(v_wind * time * meter2pixel * np.sin(np.radians(angle_wind))))
    xshift_in_pix = int(np.around(v_wind * time * meter2pixel * np.cos(np.radians(angle_wind))))

    return np.roll(phase_screens, (yshift_in_pix,xshift_in_pix), axis=(-2,-1)), (xshift_in_pix, yshift_in_pix)

def uniform_disk(ysz, xsz, radius, rebin, between_pix=False, norm=False):
    ''' ---------------------------------------------------------
    returns an (ys x xs) array with a uniform disk of radius "radius".
    ---------------------------------------------------------  '''
    xsz2 = xsz * rebin
    ysz2 = ysz * rebin
    radius2 = radius * rebin

    if between_pix is False:
        xx,yy  = np.meshgrid(np.arange(xsz2)-xsz2//2, np.arange(ysz2)-ysz2//2)
    else:
        xx,yy  = np.meshgrid(np.arange(xsz2)-xsz2//2+0.5, np.arange(ysz2)-ysz2//2+0.5)
    mydist = np.hypot(yy,xx)
    res = np.zeros_like(mydist)
    res[mydist <= radius2] = 1.0
    res = np.reshape(res, (ysz, rebin, xsz, rebin))
    res = res.mean(3).mean(1)
    if norm:
        res = res / (np.sum(res))**0.5

    return(res)

def createSubPupil(sz, sub_rad, centres, rebin, between_pix=False, norm=False):
    pupil = np.zeros((sz, sz))
    sub_pupil = uniform_disk(sub_rad*2, sub_rad*2, sub_rad, rebin, between_pix, norm)
    for centre in centres:
        pupil[centre[0]-sub_rad:centre[0]+sub_rad, centre[1]-sub_rad:centre[1]+sub_rad] = sub_pupil
    return pupil

def calculateInjection(phs_pup, wl, centre, sub_rad, geo_inj=0.8):
    phs_screen = phs_pup[centre[0]-sub_rad:centre[0]+sub_rad, centre[1]-sub_rad:centre[1]+sub_rad]
    rms = np.std(phs_screen)
    strehl = np.exp(-(2*np.pi/wl)**2 * rms**2)
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
    crop = shape[axis]//binning*binning # Number of frames which can be binned respect to the input value
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
        noisy_data = np.random.poisson(data*QE, size=data.shape, dtype=np.float32)
    else:
        noisy_data = np.random.poisson(data*QE*enf, size=data.shape, dtype=np.float32)

    return noisy_data

def _addRON(data, ron, offset):
    ron_noise = np.random.normal(offset, ron, data.shape)
    noisy_data = data + ron_noise
    return noisy_data

def _addDark(ndark, fps, shape, enf=None):
    dark_electrons = ndark / fps
    if enf is None:
        noisy_data = np.random.poisson(dark_electrons, size=shape, dtype=np.float32)
    else:
        noisy_data = np.random.poisson(dark_electrons*enf, size=shape, dtype=np.float32)

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

def measurePhase(data, wl):
    di = data[0] - data[2]
    denom = (data[0]+data[2]-2*data[1])
    di2 = di / denom
    dphi = np.arctan(-3**0.5*di2)
    dphi = dphi / (2*np.pi) * wl
    return np.array([dphi], dtype=np.float32)

def measurePhase2(data, wl):
    di = data[0] - data[2]
    denom = (data[0]+data[2]-2*data[1])
    di2 = di / denom
    dphi = np.arctan(-3**0.5*di2)

    pij = np.array([(dphi[i+1] - dphi[i]) / (2*np.pi*(1/wl[i+1] - 1/wl[i])) for i in range(wl.size-1)], dtype=np.float32)
    ddphi = np.array([abs(dphi[i] - dphi[i+1]) for i in range(wl.size-1)], dtype=np.float32)

    weight = np.zeros_like(ddphi)
    weight[ddphi<2*np.pi/3] = 1.

    ddm = np.sum(weight * pij, 0) / np.sum(weight, 0)

    return np.array([ddm], dtype=np.float32)

def measurePhase3(data, wl):
    di = data[0] - data[2]
    denom = (data[0]+data[2]-2*data[1])
    di2 = di / denom
    dphi = np.arctan(-3**0.5*di2)

    pij = np.diff(dphi) / np.array([(2*np.pi*(1/wl[i+1] - 1/wl[i])) for i in range(wl.size-1)], dtype=np.float32)
    ddphi = np.abs(np.diff(dphi))

    weight = np.zeros_like(ddphi)
    weight[ddphi<2*np.pi/3] = 1.

    ddm = np.sum(weight * pij, 0) / np.sum(weight, 0)

    return np.array([ddm], dtype=np.float32)

def selectCoupler(name):
    if name == 'tricoupler':
        return 1/3**0.5 * np.array([[1.0                  , np.exp(1j* 2*np.pi/3)],
                                    [np.exp(1j* 2*np.pi/3), np.exp(1j* 2*np.pi/3)],
                                    [np.exp(1j* 2*np.pi/3), 1.]], dtype=np.complex64)
    else:
        return 1/2**0.5 * np.array([[1.0                  , np.exp(-1j* np.pi/2)],
                                    [np.exp(-1j* np.pi/2)  , 1.],
                                    [0.                   , 0.]], dtype=np.complex64)

def calculate_pistons(phs_pup, sub_pupil_centres, sub_rad):
    '''Calculates the piston of each sub pupil'''
    pistons = np.zeros(len(sub_pupil_centres))
    for i, centre in enumerate(sub_pupil_centres):
        sub_pupil = phs_pup[centre[0]-sub_rad:centre[0]+sub_rad, centre[1]-sub_rad:centre[1]+sub_rad]
        pistons[i] = np.mean(sub_pupil[sub_pupil != 0])
    return pistons


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wl0 = 1.6e-6
    bandwidth = 200e-9
    dwl = 5e-9
    wl = np.arange(wl0-bandwidth/2, wl0+bandwidth/2, dwl, dtype=np.float32)
    pistons = np.linspace(-10*wl0, 10*wl0, 1000, False)
    print('lc', wl0**2/bandwidth, wl0**2/dwl)

    coupler = selectCoupler('tricoupler')

    a_in = np.array([np.exp(1j*2*np.pi/wl[None,:]*pistons[:,None]), np.ones((pistons.size, wl.size))], dtype=np.complex64)
    a_in = np.transpose(a_in, (2, 0, 1))
    a_out = coupler@a_in
    a_out = np.transpose(a_out, (1, 2, 0))
    i_out = np.abs(a_out)**2

    ddm = np.array([measurePhase2(i_out[:,i], wl) for i in range(pistons.size)])
    ddm2 = np.array([measurePhase3(i_out[:,i], wl) for i in range(pistons.size)])

    plt.figure()
    plt.plot(pistons/wl0, ddm/wl0)
    plt.plot(pistons/wl0, ddm2/wl0)
    plt.grid()

    # plt.figure()
    # plt.plot(cp.asnumpy(pistons)/wl0, cp.asnumpy(i_out[0,:].mean(-1)), label='Left')
    # plt.plot(cp.asnumpy(pistons)/wl0, cp.asnumpy(i_out[1,:].mean(-1)), label='Center')
    # plt.plot(cp.asnumpy(pistons)/wl0, cp.asnumpy(i_out[2,:].mean(-1)), label='Right')
    # plt.legend(loc='best')
    # plt.grid()

    # Check correlator
    # num = i_out[0] + i_out[2] - 2*i_out[1]
    # num = i_out[0] - i_out[2]
    # denom = cp.sqrt(3/4 * ((i_out[0]-i_out[2])**2 + 1/3*(i_out[0] + i_out[2] - 2*i_out[1])**2))
    # model_fringes = cp.asnumpy(num / denom)
    # model_fringes = model_fringes.T

    # plt.figure()
    # plt.plot(cp.asnumpy(pistons)/wl0, model_fringes.T, alpha=0.5)
    # plt.grid()

    # piston_atm = wl/4
    # a_test = cp.array([cp.exp(1j*2*cp.pi/wl*piston_atm), cp.ones((wl.size,))], dtype=cp.complex64)
    # b_test = tricoupler@a_test
    # i_test = cp.abs(b_test)**2

    # num2 = i_test[0] + i_test[2] - 2*i_test[1]
    # num2 = i_test[0] - i_test[2]
    # denom2 = cp.sqrt(3/4 * ((i_test[0]-i_test[2])**2 + 1/3*(i_test[0] + i_test[2] - 2*i_test[1])**2))
    # fringe = cp.asnumpy(num2 / denom2)

    # fringe = cp.asnumpy((i_test[0] + i_test[2] - 2 * i_test[1]) / (i_test.sum(0)))

    # correlation = model_fringes@fringe

    # plt.figure()
    # plt.plot(cp.asnumpy(pistons)/wl0, correlation)
    # plt.grid()
