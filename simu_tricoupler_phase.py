#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

*** A copy of simu_tricoupler.py for measuring the output intensities over the (achromatic) input phase!! ***

This code runs the simulation of the tricoupler for a range of different achromatic input phases and plots the variation
of intensity with phase shift. Treats the wavelength the data is sampled at as 1.6e-6 micron (hard-coded). 

Note that this code was copied from simu_tricoupler.py but some parts have been removed/altered. d
"""

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import simu_tricoupler_lib as lib
from astropy.time import Time
from datetime import datetime

# Teresa's imports
import sys
from tqdm import tqdm
import time
import h5py
import addcopyfighandler # allows to ctrl+C matplotlib images

# =============================================================================
# Settings
# =============================================================================
# To save data
save = False
# To simulate photon noise
activate_photon_noise = False
# To simulate the detector noise
activate_detector_noise = False
# To simulate the flux of a star, otherwise the incoming fluxes are equal to 1
activate_flux = False
# Use an achromatic phase mask instead of air-delaying (chromatic) one beam with respect to the other
activate_achromatic_phase_shift = True
# To activate turbulence
activate_turbulence = False
# Use of photometric outputs
activate_photometric_output = True

# Set the seed to an integer for repeatable simulation, to ``None'' otherwise
seed = 1

# Size in pixels of the pupil of the telescope (which will be later cropped into subpupils)
sz = 256
# Oversampling the array of the telescope for various use (e.g. bigger phase screen to mimic turbulence without wrapping)
oversz = 4

# Set the values of the phase masks for both tricoupler and directional coupler for each beam
# achromatic_phasemask_tricoupler = np.array([np.pi, 0.])
achromatic_phasemask_tricoupler = np.array([0., 0.])
achromatic_phasemask_cocoupler = np.array([np.pi/2, 0.])

all_phases_tricoupler = np.linspace(0., 2*np.pi, 100)


# ============================================================================
# Importing tricoupler and bicoupler data 
# ============================================================================

chromaticity = False
chromaticity_tri = False

# Tricoupler: RSoft data
if chromaticity_tri:
    l_out = np.transpose(np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_1_last.dat"))
    c_out = np.transpose(np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_2_last.dat"))
    r_out = np.transpose(np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_3_last.dat"))
    
    wl = cp.array(l_out[0])*1e-6
    t_coeff = l_out[1]**0.5
    c_coeff_c = c_out[1]**0.5
    c_coeff_r = r_out[1]**0.5
    
    c_coeff = np.mean(np.array([c_coeff_c, c_coeff_r]), axis=0) # taking an average FOR NOW; figure out what to do more specifically later
    
    # Note: re-writing t and c coefficients for now!!!
    # c_coeff = np.linspace(0, 2/3,56) # 56 spectral channels for now
    c_coeff = np.full(56, 1/3)
    t_coeff = np.sqrt(1-2*(c_coeff**2))
    
    wl = cp.linspace(1.5e-06, 1.7e-06, len(c_coeff))    
    
    phi = np.arccos(-c_coeff / (2*t_coeff))
    ones = np.ones(phi.shape)
    z = np.zeros(phi.shape)
    
# Bicoupler: GLINT data
one_baseline_data = False # GLINT data that used only one baseline (two apertures)
if chromaticity:
    if one_baseline_data:
        in_file = h5py.File("one_baseline_data.hdf5", 'r')
    
        wl = cp.array(in_file['wl']) * 1e-9
        alpha_b1 = np.array(in_file['alpha_b1'])
        alpha_b2 = np.array(in_file['alpha_b2'])
        kappa_12 = np.array(in_file['kappa_12'])
        kappa_21 = np.array(in_file['kappa_21'])
        
    else:
        # Contains current data from GLINT (4 beams, 6 baselines) hence need to calculate 4 splitting coeffs per beam 
        zeta_file = h5py.File("20210322_zeta_coeff_raw.hdf5", 'r')
    
        # null/antinull outputs for beams 1 and 2 (zeta coefficients)
        zeta_b1_n1 = np.array(zeta_file['b1null1']) # Conversion in numpy array is mandatory to do calculations with the values
        zeta_b1_an1 = np.array(zeta_file['b1null7'])
        zeta_b1_n3 = np.array(zeta_file['b1null3'])
        zeta_b1_an3 = np.array(zeta_file['b1null9'])
        zeta_b1_n5 = np.array(zeta_file['b1null5'])
        zeta_b1_an5 = np.array(zeta_file['b1null11'])
    
        zeta_b2_n1 = np.array(zeta_file['b2null1']) # Conversion in numpy array is mandatory to do calculations with the values
        zeta_b2_an1 = np.array(zeta_file['b2null7'])
        zeta_b2_n2 = np.array(zeta_file['b2null2'])
        zeta_b2_an2 = np.array(zeta_file['b2null8'])
        zeta_b2_n6 = np.array(zeta_file['b2null6'])
        zeta_b2_an6 = np.array(zeta_file['b2null12'])
    
        # splitting ratio for beam 1 (into coupler 1)
        alpha_b1 = (zeta_b1_n1 + zeta_b1_an1) / (1 + (zeta_b1_n1 + zeta_b1_an1) + (zeta_b1_n3 + zeta_b1_an3) + (zeta_b1_n5 + zeta_b1_an5))
        # other splitting ratios for beam 1
        # beta_b1 = 1 / (1 + (zeta_b1_n1 + zeta_b1_an1) + (zeta_b1_n3 + zeta_b1_an3) + (zeta_b1_n5 + zeta_b1_an5))
        # gamma_b1 = (zeta_b1_n3 + zeta_b1_an3) / (1 + (zeta_b1_n1 + zeta_b1_an1) + (zeta_b1_n3 + zeta_b1_an3) + (zeta_b1_n5 + zeta_b1_an5))
        # delta_b1 = (zeta_b1_n5 + zeta_b1_an5) / (1 + (zeta_b1_n1 + zeta_b1_an1) + (zeta_b1_n3 + zeta_b1_an3) + (zeta_b1_n5 + zeta_b1_an5))
    
        # first splitting ratio for beam 2 (into coupler 1)
        alpha_b2 = (zeta_b2_n1 + zeta_b2_an1) / (1 + (zeta_b2_n1 + zeta_b2_an1) + (zeta_b2_n2 + zeta_b2_an2) + (zeta_b2_n6 + zeta_b2_an6))
    
        # Wavelength scale; note we cut off the highest and lowest wavelengths as zeta coeffs become messy there
        wl = np.array(zeta_file['wl_scale']) # wavelength scale
        within = ((wl < 1650) & (wl > 1350)) # central wavelength 1550 +/- 100 nm
        wl = cp.array(wl[within] * 1e-9)
        
        zeta_file.close()
    
    
        # Coupling and splitting coefficients
        kappa_12 = (zeta_b1_an1 / zeta_b1_n1) / (1 + (zeta_b1_an1 / zeta_b1_n1))
        kappa_21 = (zeta_b2_n1 / zeta_b2_an1) / (1 + (zeta_b2_n1 / zeta_b2_an1))
    
        kappa_12 = kappa_12[within]
        kappa_21 = kappa_21[within]
        alpha_b1 = alpha_b1[within]
        # beta_b1 = beta_b1[within]
        # gamma_b1 = gamma_b1[within]
        # delta_b1 = delta_b1[within]
        alpha_b2 = alpha_b2[within]


# =============================================================================
# Telescope and AO parameters
# =============================================================================
tdiam = 8.2 # Diameter of the telescope (in meter)
subpup_diam = 1. # Diameter of the sub-pupils (in meter)
baseline = 5.55 # Distance between two sub-pupils (in meter)
fc_scex = 19.5 # 19.5 l/D is the cutoff frequency of the DM for 1200 modes corrected (source: Vincent)
wavel_r0 = 0.5e-6 # wavelength where r0 is measured (in meters)
# wavel = 1.6e-6 # Wavelength of observation (in meter)
# wavel = float(np.mean(wl))
bandwidth = 0.2e-6 # Bandwidth around the wavelength of observation (in meter)
dwl = 5e-9 # Width of one spectral channel (in meter)

if not chromaticity and not chromaticity_tri:
    wavel = 1.6e-6 
    wl = cp.arange(wavel-bandwidth/2, wavel+bandwidth/2, dwl, dtype=cp.float32)
else:
    wavel = float(np.mean(wl))
    # wl should already be defined

if 1 == 0:
    wavel = 0
    bandwidth = 0
    dwl = 0

meter2pixel = sz / tdiam # scale factor converting the meter-size in pixel, in pix/m
ao_correc = 8. # How well the AO flattens the wavefront

# =============================================================================
# Atmo parameters
# =============================================================================
r0 = 0.16  # Fried parameter at wavelength wavel_r0 (in meters), the bigger, the better the seeing is
ll = tdiam * oversz # Physical extension of the wavefront (in meter)
L0 = 1e15 # Outer scale for the model of turbulence, keep it close to infinity for Kolmogorov turbulence (the simplest form) (in meter)
wind_speed = 9.8 # speed of the wind (in m/s)
angle = 45 # Direction of the wind

# =============================================================================
# Acquisition and detector parameters
# =============================================================================
fps = 2000 # frame rate (in Hz)
delay = 0.001 # delay of the servo loop (in second)
dit = 1 / fps # Detector Integration Time, time during which the detector collects photon (in second)
timestep = 1e-4 # time step of the simulation (in second)
time_obs = 0.01 #0.1 # duration of observation (in second)

# Let's define the axe of time on which any event will happened (turbulence, frame reading, servo loop)
timeline = np.around(np.arange(0, time_obs+delay, timestep, dtype=cp.float32), int(-np.log10(timestep)))

# Detector is CRED-1
read_noise = 0.7 # Read noise of the detector (in e-)
QE = 0.6 # Quantum efficiency (probability of detection of a photon by the detector)
ndark = 50 # Dark current (false event occured because of the temperature) (in e-/pix/second)
enf = 1. #1.25 # Excess noise factor due to the amplification process

# # Detector is CRED-2
# read_noise = 30 # Read noise of the detector (in e-)
# QE = 0.8 # Quantum efficiency (probability of detection of a photon by the detector)
# ndark = 1500 # Dark current (false event occured because of the temperature) (in e-/pix/second)

# =============================================================================
# Beam combiner
# =============================================================================
if activate_photometric_output:
    # Fractions of intensity sent to photometric output
    coeff_tri = 0.25
    coeff_bi = 1/3.
else:
    coeff_tri = 0.
    coeff_bi = 0.

'''
Transfer matrix of a tricoupler. Center row is the null output,
phase and antinull is recovered from linear combinations of the three outputs (rows)

Structure:
    1st row = left output
    2st row = null output
    3rd row = right output
    4th row = photometric output A
    5th row = photometric output B
'''

# Loading in RSoft data for chromatic tricoupler
if chromaticity_tri:
    
    tricoupler = cp.asarray(np.array([[t_coeff * np.exp(1j*phi) , c_coeff                  , z    , z   ],
                                      [c_coeff                  , c_coeff                  , z    , z   ],
                                      [c_coeff                  , t_coeff * np.exp(1j*phi) , z    , z   ],
                                      [z                        , z                        , ones , z   ],
                                      [z                        , z                        , z    , ones]], dtype=np.complex64))

else:
    # tricoupler = cp.asarray(np.array([[1/3**0.5                         , 1/3**0.5 * np.exp(1j* 2*np.pi/3)  , 0., 0.],
    #                                   [1/3**0.5 * np.exp(1j* 2*np.pi/3) , 1/3**0.5 * np.exp(1j* 2*np.pi/3)  , 0., 0.],
    #                                   [1/3**0.5 * np.exp(1j* 2*np.pi/3) , 1/3**0.5                          , 0., 0.],
    #                                   [0.                               , 0.                                , 1., 0.],
    #                                   [0.                               , 0.                                , 0., 1.]], dtype=np.complex64))

    # Changing the tricoupler convention so phase shifts are on diagonals
    tricoupler = cp.asarray(np.array([[1/3**0.5 * np.exp(1j* -2*np.pi/3) , 1/3**0.5                          , 0., 0.],
                                      [1/3**0.5                         , 1/3**0.5                          , 0., 0.],
                                      [1/3**0.5                         , 1/3**0.5 * np.exp(1j* -2*np.pi/3)  , 0., 0.],
                                      [0.                               , 0.                                , 1., 0.],
                                      [0.                               , 0.                                , 0., 1.]], dtype=np.complex64))


# Tri splitter

if chromaticity_tri:
    
    coeff_tri = 0.25 * ones # mock splitting coefficients; all set to the same value for now
    
    tri_splitter = cp.array([[1-coeff_tri , z           ],
                             [z           , 1-coeff_tri ],
                             [coeff_tri   , z           ],
                             [z           , coeff_tri   ]], dtype=cp.float32)
    tri_splitter = tri_splitter**0.5
    
    # Total combiner
    combiner_tri = cp.einsum('ijk,jlk->ilk', tricoupler, tri_splitter)

    
else:
    # Splitter before combining the beams to get the photometric tap
    tri_splitter = cp.array([[1-coeff_tri, 0.         ],
                             [0.         , 1-coeff_tri],
                             [coeff_tri  , 0.         ],
                             [0.         , coeff_tri  ]], dtype=cp.float32)
    # The coefficients below are given for intensities although we deal with amplitude of wavefront
    # so a square root must be applied to them
    tri_splitter = tri_splitter**0.5
    
    # Total combiner
    combiner_tri = tricoupler@tri_splitter

'''
Transfer matrix of a directional coupler.
One output is the nulled signal, the other is the antinulled.

Structure:
    1st row = null output
    2st row = antinull output
    3rd row = fake output (for compatibility with the tricoupler when plotting the results)
    4th row = photometric output A
    5th row = photometric output B
'''


# =============================================================================
# Coupling and splitting coefficients for chromatic case 
# =============================================================================


if chromaticity:
    z = np.zeros(len(kappa_12))
    ones = np.ones(len(kappa_12))

    bicoupler = cp.asarray(np.array([[(1-kappa_12)**0.5                    , kappa_12**0.5 * np.exp(-1j* np.pi/2) , z    , z   ],
                                     [kappa_21**0.5 * np.exp(-1j* np.pi/2) , (1-kappa_21)**0.5                    , z    , z   ],
                                     [z                                    , z                                    , z    , z   ],
                                     [z                                    , z                                    , ones , z   ],
                                     [z                                    , z                                    , z    , ones]], dtype=np.complex64))

else:
    bicoupler = cp.asarray(np.array([[1.0/2**0.5                         , 1.0/2**0.5 * np.exp(-1j* np.pi/2) , 0., 0.],
                                      [1.0/2**0.5 * np.exp(-1j* np.pi/2) , 1.0/2**0.5                        , 0., 0.],
                                      [0.0                               , 0.0                               , 0., 0.],
                                      [0.0                               , 0.                                , 1., 0.],
                                      [0.                                , 0.                                , 0., 1.]], dtype=np.complex64))


if chromaticity:
    bi_splitter = cp.array([[(1-alpha_b1) , z           ],
                            [z            , (1-alpha_b2)],
                            [(alpha_b1)   , z           ],
                            [z            , (alpha_b2)  ]], dtype=cp.complex64)

    # The coefficients below are given for intensities although we deal with amplitude of wavefront
    # so a square root must be applied to them
    bi_splitter = bi_splitter**0.5

    # Total combiner
    combiner_bi = cp.einsum('ijk,jlk->ilk', bicoupler, bi_splitter)
    # performs matrix multiplication while keeping indices in the right order

else:
    # Splitter before combining the beams to get the photometric tap
    bi_splitter = cp.array([[1-coeff_bi , 0.],
                              [0.        , 1-coeff_bi],
                              [coeff_bi  , 0.],
                              [0.        , coeff_bi]], dtype=cp.float32)

    # The coefficients below are given for intensities although we deal with amplitude of wavefront
    # so a square root must be applied to them
    bi_splitter = bi_splitter**0.5

    # Total combiner
    combiner_bi = bicoupler@bi_splitter

# =============================================================================
# Convert physical quantities in pixel
# =============================================================================
pupil_rad = sz // 2
subpup_diamp = subpup_diam / tdiam * sz
baselinep = baseline / tdiam * sz

# =============================================================================
# Servo loop parameters
# =============================================================================
servo_gain = 0.
servo_int = 0.# 1.#0.05 # 0: no fringe tracking, 1.: fringe trcking at high flux, 0.05: fringe tracking at low flux (SNR<=2)
err_int = cp.array([0.], dtype=cp.float32)
diff_piston_command = cp.array([0.], dtype=cp.float32)

if not activate_achromatic_phase_shift:
    DIFF_PISTON_REF = wavel / 2
    DIFF_PISTON_REFBI = wavel / 4
    achromatic_phasemask_tricoupler[:] = 0.
    achromatic_phasemask_cocoupler[:] = 0.
else:
    DIFF_PISTON_REF = 0.
    DIFF_PISTON_REFBI = 0.

# =============================================================================
# Flux parameters
# =============================================================================
# Magnitude of the star, the smaller, the brighter.
magnitude = -27

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
MAG0FLUX = 1e10 #ph/um/s/m^2
SCEXAO_THROUGHPUT = 0.2
pupil_area = np.pi / 4 * subpup_diam**2

star_photons = MAG0FLUX * 10**(-0.4*magnitude) * SCEXAO_THROUGHPUT * pupil_area * bandwidth*1e6 * dit
print('Star photo-electrons', star_photons*0.75*QE, (star_photons*0.75*QE)**0.5)

# =============================================================================
# Misc
# =============================================================================
"""
Phase mask may be completely shifted.
To prevent aliasing, a new mask is created and the time to calculate the shift
is offseted.
"""
TIME_OFFSET = 0. # time offset of the shift of the mask after creation of a new one
count_delay = 1
count_dit = 1
debug = []

# =============================================================================
# Run
# =============================================================================
start = timer()

# Create the sub-pupil
pupil = lib.createSubPupil(sz, int(subpup_diamp//2), baselinep, 5, norm=False)
pupil = cp.array(pupil , dtype=cp.float32)

# Create the phase screen.
if activate_turbulence:
    phs_screen = lib.generatePhaseScreen(wavel_r0, sz*oversz, ll, r0, L0, fc=fc_scex, correc=ao_correc, pdiam=tdiam, seed=seed)
    phs_screen = cp.array(phs_screen, dtype=cp.float32)
else:
    phs_screen = cp.zeros((sz*oversz, sz*oversz), dtype=cp.float32)

# =============================================================================
# Initiate storage lists
# =============================================================================
data = []
noisy_data = []
data_noft = []
noisy_data_noft = []
data_bi = []
noisy_data_bi = []

diff_pistons_atm = []
diff_pistons_ft = []
diff_pistons_fr = []
diff_pistons_bi = []
diff_pistons_measured = []
diff_pistons_measured_noft = []
injections = []
injections_ft = []
injections_fr = []
shifts = []
time_fr = []
time_ft = []


i_out = cp.zeros((5, wl.size), dtype=cp.float32)
i_out_noft = cp.zeros((5, wl.size), dtype=cp.float32)
i_out_bi = cp.zeros((5, wl.size), dtype=cp.float32)

# =============================================================================
# Loop over phases!!! and over simulated time
# =============================================================================

pups_A = np.linspace(-100.*wavel, 100*wavel, timeline.size)

all_left = []
all_null = []
all_right = []













for i in tqdm(range(0, 100)): # loop of length 100
    for index, t in enumerate(timeline):
        phs_screen_moved, xyshift = lib.movePhaseScreen(phs_screen, wind_speed, angle, t-TIME_OFFSET, meter2pixel)
        if xyshift[0] > phs_screen.shape[0] or xyshift[1] > phs_screen.shape[1]:
            if seed != None:
                seed += 20
            phs_screen = lib.generatePhaseScreen(wavel_r0, sz*oversz, ll, r0, L0, fc=fc_scex, correc=9, pdiam=tdiam, seed=None)
            phs_screen = cp.array(phs_screen, dtype=cp.float32)
            TIME_OFFSET = t
        
        # plt.imshow(cp.asnumpy(phs_screen_moved)) # checking this works for now
        # plt.show()
    
        shifts.append(xyshift)
        # We stay in phase space hence the simple multiplication below to crop the wavefront.
        phs_pup = pupil * phs_screen_moved[phs_screen_moved.shape[0]//2-sz//2:phs_screen_moved.shape[0]//2+sz//2,\
                                           phs_screen_moved.shape[1]//2-sz//2:phs_screen_moved.shape[1]//2+sz//2]
    
        # Measure the piston of the subpupils
        piston_pupA = cp.mean(phs_pup[:,:sz//2][pupil[:,:sz//2]!=0], keepdims=True)
        # piston_pupA = 3.*wavel # setting piston_pupA to a const value
        # piston_pupA = pups_A[index] # setting piston_pupA to a ramp
        piston_pupB = cp.mean(phs_pup[:,sz//2:][pupil[:,sz//2:]!=0], keepdims=True)
    
        # Measure the differential atmospheric piston
        diff_piston_atm = piston_pupA - piston_pupB
        diff_pistons_atm.append(diff_piston_atm)
    
        # Total differential piston, including the instrumental air-delay between the beams
        diff_piston = cp.array([DIFF_PISTON_REF + diff_piston_atm[0]], dtype=cp.float32) # pupil A - pupil B
        diff_piston_corrected = diff_piston - diff_piston_command[0]
    
        injection = cp.array([lib.calculateInjection(phs_pup[:,:sz//2][pupil[:,:sz//2]!=0], wl), \
                              lib.calculateInjection(phs_pup[:,sz//2:][pupil[:,sz//2:]!=0], wl)])
        injections.append(injection)
    
        # Input wavefronts
        a_in = cp.array([cp.exp(1j*2*cp.pi/wl*(piston_pupA + diff_piston_corrected) + 1j*achromatic_phasemask_tricoupler[0]),\
                         cp.exp(1j*2*cp.pi/wl*piston_pupB                           + 1j*achromatic_phasemask_tricoupler[1])],\
                        dtype=cp.complex64)
            
        # a_in[1] = cp.zeros(len(a_in[1])) # right input is 0; right input only
        # sys.exit()
    
        if activate_flux:
            a_in *= injection**0.5 * star_photons**0.5
            
        if chromaticity_tri:
            a_out = cp.einsum('ijk,jk->ik', combiner_tri, a_in)
        else:
            a_out = combiner_tri@a_in # Deduce the outcoming wavefront after the integrated-optics device
        i_out += cp.abs(a_out)**2
        # count += 1
        # if count==50:
        #     print(i_out)
        #     sys.exit()
        # print(diff_piston_command)
        
        # Same but with no fringe tracking
        a_in_noft = cp.array([cp.exp(1j*2*cp.pi/wl*(piston_pupA)    + 1j*achromatic_phasemask_tricoupler[0]),\
                              cp.exp(1j*2*cp.pi/wl*piston_pupB      + 1j*achromatic_phasemask_tricoupler[1])],\
                             dtype=cp.complex64)
        if activate_flux:
            a_in_noft *= injection**0.5 * star_photons**0.5
            
        if chromaticity_tri:
            a_out_noft = cp.einsum('ijk,jk->ik', combiner_tri, a_in_noft)
        else:
            a_out_noft = combiner_tri@a_in_noft
        i_out_noft += cp.abs(a_out_noft)**2
    
        # Same but with codirectional coupler
        diff_piston_bi = cp.array([DIFF_PISTON_REFBI + diff_piston_atm[0]], dtype=cp.float32) # pupil 1 - pupil 2
        diff_pistons_bi.append(diff_piston_bi)
        a_in_bi = cp.array([cp.exp(1j*2*cp.pi/wl*(piston_pupA + diff_piston_bi) + 1j*achromatic_phasemask_cocoupler[0]),\
                            cp.exp(1j*2*cp.pi/wl*(piston_pupB)                  + 1j*achromatic_phasemask_cocoupler[1])],\
                           dtype=cp.complex64)
    
    
    
        if activate_flux:
            a_in_bi *= injection**0.5 * star_photons**0.5
    
    
        if chromaticity:
            a_out_bi = cp.einsum('ijk,jk->ik', combiner_bi, a_in_bi)
            # multiplies the second, bzw. first axes and keeps the wavelength axes
    
        else:
            a_out_bi = combiner_bi@a_in_bi
    
        i_out_bi += cp.abs(a_out_bi)**2
    
        if count_dit < dit/timestep:
            count_dit += 1
        else:
            # print('Acq frame', count_dit, count_delay, t)
            i_out /= count_dit
            data.append(i_out)
            noisy_i_out = lib.addNoise(i_out, QE, read_noise, ndark, fps, activate_photon_noise, activate_detector_noise, enf)
            noisy_data.append(noisy_i_out)
    
            # No fringe tracking
            i_out_noft /= count_dit
            data_noft.append(i_out_noft)
            noisy_i_out_noft = lib.addNoise(i_out_noft, QE, read_noise, ndark, fps, activate_photon_noise, activate_detector_noise, enf)
            noisy_data_noft.append(noisy_i_out_noft)
    
            # Codirectional coupler
            i_out_bi /= count_dit
            data_bi.append(i_out_bi)
            noisy_i_out_bi = lib.addNoise(i_out_bi, QE, read_noise, ndark, fps, activate_photon_noise, activate_detector_noise, enf)
            noisy_data_bi.append(noisy_i_out_bi)
    
            # Store some data
            diff_pistons_fr.append(diff_piston_atm)
            injections_fr.append(injection)
            time_fr.append(t)
    
            if count_delay == count_dit: # Capture the first frame of the double cycles on DIT and Delay to send to fringe trakcer
                noisy_i_out_toft = cp.array(noisy_i_out, dtype=cp.float32)
                noisy_i_out_noft_toft = cp.array(noisy_i_out_noft, dtype=cp.float32)
                diff_piston_toft = cp.array(diff_piston_atm)
                injection_toft = cp.array(injection)
    
            # Reinit the cycle of integration of a frame
            i_out = cp.zeros_like(i_out)
            i_out_noft = cp.zeros_like(i_out_noft)
            i_out_bi = cp.zeros_like(i_out_bi)
            count_dit = 1
    
    
        # if couplertype == 'tricoupler':
        if count_delay < delay/timestep:
            count_delay += 1
        else:
            # print('Delay', count_dit, count_delay, t)
            # With fringe tracking and application of the measured piston
            diff_piston_meas = lib.measurePhase3(noisy_i_out_toft, wl)
            diff_pistons_measured.append(diff_piston_meas)
            diff_piston_error = diff_piston_meas - DIFF_PISTON_REF
            err_int += diff_piston_error
            diff_piston_command = servo_int * err_int + servo_gain * diff_piston_error
    
            diff_piston_meas_noft = lib.measurePhase3(noisy_i_out_noft_toft, wl)
            diff_pistons_measured_noft.append(diff_piston_meas_noft)
    
            diff_pistons_ft.append(diff_piston_toft)
            injections_ft.append(injection_toft)
            time_ft.append(t)
            count_delay = 1
            
    
    
    
    # =============================================================================
    # Format data
    # =============================================================================
    diff_pistons_atm = cp.asnumpy(cp.array(diff_pistons_atm))
    diff_pistons_bi = cp.asnumpy(cp.array(diff_pistons_bi))
    diff_pistons_ft = cp.asnumpy(cp.array(diff_pistons_ft))
    diff_pistons_fr = cp.asnumpy(cp.array(diff_pistons_fr))
    diff_pistons_measured = cp.asnumpy(cp.array(diff_pistons_measured))
    diff_pistons_measured_noft = cp.asnumpy(cp.array(diff_pistons_measured_noft))
    injections = cp.asnumpy(cp.array(injections))
    injections_ft = cp.asnumpy(cp.array(injections_ft))
    injections_fr = cp.asnumpy(cp.array(injections_fr))
    
    shifts = np.array(shifts)
    time_fr = np.array(time_fr)
    time_ft = np.array(time_ft)
    
    diff_pistons_atm = np.squeeze(diff_pistons_atm)
    diff_pistons_bi = np.squeeze(diff_pistons_bi)
    diff_pistons_ft = np.squeeze(diff_pistons_ft)
    diff_pistons_fr = np.squeeze(diff_pistons_fr)
    diff_pistons_measured = np.squeeze(diff_pistons_measured)
    diff_pistons_measured_noft = np.squeeze(diff_pistons_measured_noft)
    
    data = cp.asnumpy(cp.array(data))
    noisy_data = cp.asnumpy(cp.array(noisy_data))
    data_bi = cp.asnumpy(cp.array(data_bi))
    data_noft = cp.asnumpy(cp.array(data_noft))
    noisy_data_noft = cp.asnumpy(cp.array(noisy_data_noft))
    noisy_data_bi = cp.asnumpy(cp.array(noisy_data_bi))
    
    data = np.transpose(data, (1,0,2))
    data_noft = np.transpose(data_noft, (1,0,2))
    data_bi = np.transpose(data_bi, (1,0,2))
    
    noisy_data = np.transpose(noisy_data, (1,0,2))
    noisy_data_noft = np.transpose(noisy_data_noft, (1,0,2))
    noisy_data_bi = np.transpose(noisy_data_bi, (1,0,2))
    
    stop = timer()
    print('Total duration', stop - start, "seconds")
    
    # =============================================================================
    # Calculate other quantities
    # =============================================================================
    
    # temp - Teresa added this (suppresses RuntimeWarning: invalid value encountered in true_divide)
    np.seterr(divide='ignore', invalid='ignore')
    
    antinull = 2/3 * (noisy_data[0] + noisy_data[2]) - 1/3 * noisy_data[1]
    null_depth = noisy_data[1] / antinull
    
    antinull_noft = 2/3 * (noisy_data_noft[0] + noisy_data_noft[2]) - 1/3 * noisy_data_noft[1]
    null_depth_noft = noisy_data_noft[1] / antinull_noft
    
    antinull_bi = noisy_data_bi[1]
    null_depth_bi = noisy_data_bi[0] / antinull_bi
    
    # Timestamps: removed for simplicity
    # Save data: removed for simplicity
    
    # =============================================================================
    # Setting left, null and right channels 
    # =============================================================================
    left_out = np.median(noisy_data[0], axis=0) # output A
    null_out = np.median(noisy_data[1], axis=0) # output B
    right_out = np.median(noisy_data[2], axis=0) # output C
    # phot_left = np.median(noisy_data[3], axis=0) # photometric output 
    # phot_right = np.median(noisy_data[4], axis=0)
    
    # Indexing by the central wavelength, hard-coded to be index 20 for now
    left_response = left_out[20]
    null_response = null_out[20]
    right_response = right_out[20]
    
    all_left.append(left_response)
    all_null.append(null_response)
    all_right.append(right_response)
    
    # do something to the phase here 
    if i != 99:
        achromatic_phasemask_tricoupler = np.array([all_phases_tricoupler[i+1], 0.])
    
    # Restarting all the storage lists (this code is really ugly, I know)
    data = []
    noisy_data = []
    data_noft = []
    noisy_data_noft = []
    data_bi = []
    noisy_data_bi = []
    
    diff_pistons_atm = []
    diff_pistons_ft = []
    diff_pistons_fr = []
    diff_pistons_bi = []
    diff_pistons_measured = []
    diff_pistons_measured_noft = []
    injections = []
    injections_ft = []
    injections_fr = []
    shifts = []
    time_fr = []
    time_ft = []
    
    
    i_out = cp.zeros((5, wl.size), dtype=cp.float32)
    i_out_noft = cp.zeros((5, wl.size), dtype=cp.float32)
    i_out_bi = cp.zeros((5, wl.size), dtype=cp.float32)

# =============================================================================
# Display and plot
# =============================================================================
# Note: removed all the plots from simu_tricoupler.py

print("Went through the loop!")

all_left = np.array(all_left)
all_null = np.array(all_null)
all_right = np.array(all_right)
all_phases_tricoupler = np.degrees(all_phases_tricoupler)

plt.figure(1)
plt.plot(all_phases_tricoupler, all_left, label='Left')
plt.plot(all_phases_tricoupler, all_null, label='Null')
plt.plot(all_phases_tricoupler, all_right, label='Right')
plt.xlabel("Input phase (degrees)")
plt.ylabel("Intensity")
plt.grid(True)
plt.legend(loc='best')



