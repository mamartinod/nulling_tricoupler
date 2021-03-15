#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:05:38 2021

@author: mam

The simulation aims at evaluate the ceiling null depth wrt to the efficiency
of the fringe tracking for different regimes: photon noise or read-out noise.
For the latter, properties of C-Red 1 are considered (but the ENF which is easy
                                                      to activate)

The timeline of the simulation is the delay of the servo loop so that the correction
measured at iteration N is applied at N+1.

The outputs of the tricoupler allows to recover the null 
(direct reading of the central output), the phase and the antinull by 
combinations of the 3 outputs.
FYI: the cross-product I1*I2 can be recovered as well (not needed here)

Interferometric equations are simple:
    central output: classic interferometric equation with all terms divided by 3
    left: same as central with -2pi/3 shift
    right: same as central with +2pi/3 shift
"""

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import simu_tricoupler_lib as lib
from astropy.time import Time
from datetime import datetime

# =============================================================================
# Simu Parameters    
# =============================================================================
seed = 1
sz = 256
oversz = 4
# couplertype = 'tricoupler'
save = True

# =============================================================================
# Telescope and AO parameters
# =============================================================================
tdiam = 8.2
subpup_diam = 1.
baseline = 5.55
fc_scex = 19.5 # 19.5 l/D is the cutoff frequency of the DM for 1200 modes corrected (source: Vincent)
wavel_r0 = 0.5e-6 # wavelength where r0 is measured (in meters)
wavel = 1.6e-6
bandwidth = 0.2e-6 # microns
num_channels = 10
dwl = 5e-9

meter2pixel = sz / tdiam # pix/m
ao_correc = 8. # How well the AO flattens the wavefront

# =============================================================================
# Atmo parameters
# =============================================================================
r0 = 0.16  # Fried parameter at wavelength wavel_r0 (in meters)
ll = tdiam * oversz
L0 = 1e15
wind_speed = 9.8 # in m/s
angle = 45

# =============================================================================
# Acquisition and detector parameters
# =============================================================================
fps = 2000 # in Hz
delay = 0.001 # in second
dit = 1 / fps
timestep = 1e-4 # time step of the simulation in second
time_obs = 100. # duration of observation in second
timeline = np.around(np.arange(0, time_obs+delay, timestep, dtype=cp.float32), int(-np.log10(timestep)))

# Detector is CRED-1
read_noise = 0.7 # e-
QE = 0.6
ndark = 50 # e-/pix/second
activate_nonoise = False

# # Detector is CRED-2
# read_noise = 30 # e-
# QE = 0.8
# ndark = 1500 # e-/pix/second
# activate_nonoise = False

# =============================================================================
# Nuller: central element of the simulation: the transfer matrix of the tricoupler.
# Center row is the null output, side ones are for fringe tracking.
# =============================================================================
# coupler = lib.selectCoupler(couplertype)
tricoupler = 1/3**0.5 * cp.array([[1.0                  , cp.exp(1j* 2*cp.pi/3)],
                                    [cp.exp(1j* 2*cp.pi/3), cp.exp(1j* 2*cp.pi/3)],
                                    [cp.exp(1j* 2*cp.pi/3), 1.]], dtype=cp.complex64)
bicoupler = 1/2**0.5 * cp.array([[1.0                  , cp.exp(-1j* cp.pi/2)],
                                    [cp.exp(-1j* cp.pi/2)  , 1.],
                                    [0.                   , 0.]], dtype=cp.complex64)

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
servo_int = 1.#0.05
err_int = cp.array([0.], dtype=cp.float32)
piston_command = cp.array([0.], dtype=cp.float32)

# if couplertype == 'tricoupler':
#     PISTON_REF = wavel / 2
# else:
#     PISTON_REF = wavel / 4

PISTON_REF = wavel / 2
PISTON_REFBI = wavel / 4

# =============================================================================
# Flux parameters
# =============================================================================
magnitude = 6

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

wl = cp.arange(wavel-bandwidth/2, wavel+bandwidth/2, dwl, dtype=cp.float32)
# wl = cp.array([wavel], dtype=cp.float32)
pupil = lib.createSubPupil(sz, int(subpup_diamp//2), baselinep, 5, norm=False)
pupil = cp.array(pupil , dtype=cp.float32)

phs_screen = lib.generatePhaseScreen(wavel_r0, sz*oversz, ll, r0, L0, fc=fc_scex, correc=ao_correc, pdiam=tdiam, seed=seed)
phs_screen = cp.array(phs_screen, dtype=cp.float32)
phs_screen_moved_previter = cp.array(phs_screen)
phs_screen_correlated = cp.array([0.], dtype=cp.float32)

data = []
noisy_data = []
data_noft = []
noisy_data_noft = []
data_bi = []
noisy_data_bi = []

pistons_atm = []
pistons_ft = []
pistons_fr = []
pistons_bi = []
pistons_measured = []
pistons_measured_noft = []
injections = []
injections_ft = []
injections_fr = []
shifts = []
time_fr = []
time_ft = []


i_out = cp.zeros((3, wl.size), dtype=cp.float32)
i_out_noft = cp.zeros((3, wl.size), dtype=cp.float32)
i_out_bi = cp.zeros((3, wl.size), dtype=cp.float32)

for t in timeline:
    phs_screen_moved, xyshift = lib.movePhaseScreen(phs_screen, wind_speed, angle, t-TIME_OFFSET, meter2pixel)
    if xyshift[0] > phs_screen.shape[0] or xyshift[1] > phs_screen.shape[1]:
        if seed != None:
            seed += 20
        phs_screen = lib.generatePhaseScreen(wavel_r0, sz*oversz, ll, r0, L0, fc=fc_scex, correc=9, pdiam=tdiam, seed=None)
        phs_screen = cp.array(phs_screen, dtype=cp.float32)
        TIME_OFFSET = t

    shifts.append(xyshift)
    # We stay in phase space hence the simple multiplication below to crop the wavefront.
    phs_pup = pupil * phs_screen_moved[phs_screen_moved.shape[0]//2-sz//2:phs_screen_moved.shape[0]//2+sz//2, phs_screen_moved.shape[1]//2-sz//2:phs_screen_moved.shape[1]//2+sz//2]
    
    piston_atm = cp.mean(phs_pup[:,:sz//2][pupil[:,:sz//2]!=0], keepdims=True) - cp.mean(phs_pup[:,sz//2:][pupil[:,sz//2:]!=0], keepdims=True)
    # piston_atm = cp.array([0.], dtype=cp.float32)
    pistons_atm.append(piston_atm)

    piston = cp.array([PISTON_REF + piston_atm[0]], dtype=cp.float32) # pupil 1 - pupil 2
    piston_corrected = piston - piston_command[0]


    injection = cp.array([lib.calculateInjection(phs_pup[:,:sz//2][pupil[:,:sz//2]!=0], wl), \
                          lib.calculateInjection(phs_pup[:,sz//2:][pupil[:,sz//2:]!=0], wl)])
    injections.append(injection)

    # Input wavefront, pupil 1 is shifted from pupil 2 by piston
    a_in = cp.array([cp.exp(1j*2*cp.pi/wl*piston_corrected), cp.ones_like(wl)], dtype=cp.complex64)
    if not activate_nonoise:
        a_in *= injection**0.5 * star_photons**0.5
    a_out = tricoupler@a_in # Deduce the outcoming wavefront from the incoming wavefront
    i_out += cp.abs(a_out)**2
    
    # No fringe tracking
    a_in_noft = cp.array([cp.exp(1j*2*cp.pi/wl*piston), cp.ones_like(wl)], dtype=cp.complex64)
    if not activate_nonoise:
        a_in_noft *= injection**0.5 * star_photons**0.5 
    a_out_noft = tricoupler@a_in_noft
    i_out_noft += cp.abs(a_out_noft)**2  

    # Coridrectional coupler
    piston_bi = cp.array([PISTON_REFBI + piston_atm[0]], dtype=cp.float32) # pupil 1 - pupil 2
    pistons_bi.append(piston_bi)
    a_in_bi = cp.array([cp.exp(1j*2*cp.pi/wl*(piston_bi)), cp.ones_like(wl)], dtype=cp.complex64)    
    if not activate_nonoise:
        a_in_bi *= injection**0.5 * star_photons**0.5 
    a_out_bi = bicoupler@a_in_bi
    i_out_bi += cp.abs(a_out_bi)**2
    
    if count_dit < dit/timestep:
        count_dit += 1
    else:
        # print('Acq frame', count_dit, count_delay, t)
        i_out /= count_dit 
        data.append(i_out)
        if not activate_nonoise:
            noisy_i_out = lib.addNoise(i_out, QE, read_noise, ndark, fps, enf=None)
        else:
            noisy_i_out = i_out
        noisy_data.append(noisy_i_out)
    
        # No fringe tracking
        i_out_noft /= count_dit
        data_noft.append(i_out_noft)
        if not activate_nonoise:
            noisy_i_out_noft = lib.addNoise(i_out_noft, QE, read_noise, ndark, fps, enf=None)
        else:
            noisy_i_out_noft = i_out_noft
        noisy_data_noft.append(noisy_i_out_noft)
        i_out = cp.zeros((3, wl.size), dtype=cp.float32)
        i_out_noft = cp.zeros((3, wl.size), dtype=cp.float32)
        
        # Codirectional coupler
        i_out_bi /= count_dit
        data_bi.append(i_out_bi)
        if not activate_nonoise:
            noisy_i_out_bi = lib.addNoise(i_out_bi, QE, read_noise, ndark, fps, enf=None)
        else:
            noisy_i_out_bi = i_out_bi
        noisy_data_bi.append(noisy_i_out_bi)
            
        # Store some data
        pistons_fr.append(piston_atm)
        injections_fr.append(injection)
        time_fr.append(t)
        
        if count_delay == count_dit: # Capture the first frame of the double cycles on DIT and Delay to send to fringe trakcer
            noisy_i_out_toft = cp.array(noisy_i_out, dtype=cp.float32)
            noisy_i_out_noft_toft = cp.array(noisy_i_out_noft, dtype=cp.float32)
            piston_toft = cp.array(piston_atm)
            injection_toft = cp.array(injection)

        # Reinit the cycle
        i_out = cp.zeros((3, wl.size), dtype=cp.float32)
        i_out_noft = cp.zeros((3, wl.size), dtype=cp.float32)
        i_out_bi = cp.zeros((3, wl.size), dtype=cp.float32)        
        count_dit = 1
        
        
    # if couplertype == 'tricoupler':
    if count_delay < delay/timestep:
        count_delay += 1
    else:
        # print('Delay', count_dit, count_delay, t)
        # With fringe tracking and application of the measured piston
        # piston_meas = lib.measurePhase(noisy_i_out.mean(1), wl.mean())
        piston_meas = lib.measurePhase3(noisy_i_out_toft, wl)
        pistons_measured.append(piston_meas - PISTON_REF)
        piston_error = piston_meas - PISTON_REF
        err_int += piston_error
        piston_command = servo_int * err_int + servo_gain * piston_error
        
        piston_meas_noft = lib.measurePhase3(noisy_i_out_noft_toft, wl)
        pistons_measured_noft.append(piston_meas_noft - PISTON_REF)
        
        pistons_ft.append(piston_toft)
        injections_ft.append(injection_toft)
        time_ft.append(t)
        count_delay = 1
            



pistons_atm = cp.asnumpy(cp.array(pistons_atm))
pistons_bi = cp.asnumpy(cp.array(pistons_bi))
pistons_ft = cp.asnumpy(cp.array(pistons_ft))
pistons_fr = cp.asnumpy(cp.array(pistons_fr))
pistons_measured = cp.asnumpy(cp.array(pistons_measured))
pistons_measured_noft = cp.asnumpy(cp.array(pistons_measured_noft))
injections = cp.asnumpy(cp.array(injections))
injections_ft = cp.asnumpy(cp.array(injections_ft))
injections_fr = cp.asnumpy(cp.array(injections_fr))

shifts = np.array(shifts)
time_fr = np.array(time_fr)
time_ft = np.array(time_ft)

pistons_atm = np.squeeze(pistons_atm)
pistons_bi = np.squeeze(pistons_bi)
pistons_ft = np.squeeze(pistons_ft)
pistons_fr = np.squeeze(pistons_fr)
pistons_measured = np.squeeze(pistons_measured)
pistons_measured_noft = np.squeeze(pistons_measured_noft)

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
print('Total duration', stop - start)


antinull = 2/3 * (noisy_data[0] + noisy_data[2]) - 1/3 * noisy_data[1]
null_depth = noisy_data[1] / antinull

antinull_noft = 2/3 * (noisy_data_noft[0] + noisy_data_noft[2]) - 1/3 * noisy_data_noft[1]
null_depth_noft = noisy_data_noft[1] / antinull_noft

antinull_bi = noisy_data_bi[1]
null_depth_bi = noisy_data_bi[0] / antinull_bi

timestamp = datetime.now()
timestamp = Time(timestamp).isot
timestamp = timestamp.replace('-', '')
timestamp = timestamp.replace(':', '')
timestamp = timestamp.replace('.', '')

if save:
    np.savez('results/%s_mag_%s'%(timestamp, magnitude), noisy_data=noisy_data, data=data,
             data_noft=data_noft, noisy_data_noft=noisy_data_noft, data_bi=data_bi,
             noisy_data_bi=noisy_data_bi, pistons_atm=pistons_atm, pistons_fr=pistons_fr,
             pistons_measured=pistons_measured, pistons_measured_noft=pistons_measured_noft,
             pistons_ft=pistons_ft, null_depth=null_depth, null_depth_noft=null_depth_noft,
             null_depth_bi=null_depth_bi, det_charac=np.array([QE, read_noise, ndark, fps]), 
             servo=np.array([delay, servo_gain, servo_int]), atm=np.array([r0, wind_speed, angle]))

print('Std piston no FT (input, measured)', pistons_atm.std()/wavel, pistons_measured_noft.std()/wavel)
print('Std piston with FT', pistons_measured.std()/wavel)
print('Med and std null depth with FT', np.median(null_depth.mean(1)[(null_depth.mean(1)>=0.)&(null_depth.mean(1)<=1)]), np.std(null_depth.mean(1)[(null_depth.mean(1)>=0.)&(null_depth.mean(1)<=1)]))
print('Med and std null depth no FT', np.median(null_depth_noft.mean(1)[(null_depth_noft.mean(1)>=0.)&(null_depth_noft.mean(1)<=1)]), np.std(null_depth_noft.mean(1)[(null_depth_noft.mean(1)>=0.)&(null_depth_noft.mean(1)<=1)]))
print('Med and std null depth with BI', np.median(null_depth_bi.mean(1)[(null_depth_bi.mean(1)>=0.)&(null_depth_bi.mean(1)<=1)]), np.std(null_depth_bi.mean(1)[(null_depth_bi.mean(1)>=0.)&(null_depth_bi.mean(1)<=1)]))
print('---')
print('Med and std null depth with FT', np.median(null_depth.mean(1)), np.std(null_depth.mean(1)))
print('Med and std null depth no FT', np.median(null_depth_noft.mean(1)), np.std(null_depth_noft.mean(1)))
print('Med and std null depth with BI', np.median(null_depth_bi.mean(1)), np.std(null_depth_bi.mean(1)))

# if couplertype == 'tricoupler':
plt.figure()
plt.plot(timeline, pistons_atm/wavel, label='Atm piston')
plt.plot(time_ft, pistons_measured/wavel, '.', label='Corrected piston')
plt.plot(time_ft, pistons_measured_noft/wavel, '.', label='Measured atm piston')
plt.plot(timeline, pistons_bi/wavel, '.', label='Piston in Co-coupler')
plt.plot(timeline, PISTON_REF/wavel*np.ones(timeline.size), '-', c='k', label='Piston ref')
plt.plot(timeline, PISTON_REFBI/wavel*np.ones(timeline.size), '--', c='k', label='Piston ref (co-coupler)')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Delay (count)')
plt.ylabel('Piston (wl0)')

plt.figure()
plt.plot(pistons_atm/wavel, pistons_atm/wavel, label='Atm piston')
plt.plot(pistons_ft/wavel, pistons_measured/wavel, '.', label='Corrected piston')
plt.plot(pistons_ft/wavel, pistons_measured_noft/wavel, '.', label='Measured atm piston')
plt.plot(pistons_atm/wavel, pistons_bi/wavel, '.', label='Piston in Co-coupler')
plt.plot(pistons_ft/wavel, PISTON_REF/wavel*np.ones_like(pistons_ft), '-', c='k', label='Piston ref')
plt.plot(pistons_ft/wavel, PISTON_REFBI/wavel*np.ones(pistons_ft.size), '--', c='k', label='Piston ref (co-coupler)')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Atm piston (wl0)')
plt.ylabel('Piston (wl0)')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
plt.plot(pistons_fr/wavel, noisy_data[1].mean(1), '.', c=colors[0], label='Null')
plt.plot(pistons_fr/wavel, antinull.mean(1), 'x', c=colors[0], label='Antinull')
plt.plot(pistons_fr/wavel, noisy_data_noft[1].mean(1), '.', c=colors[1], label='Null (no FT)')
plt.plot(pistons_fr/wavel, antinull_noft.mean(1), 'x', c=colors[1], label='Antinull (no FT)')
plt.plot(pistons_fr/wavel, noisy_data_bi[0].mean(1), '.', c=colors[2], label='Null (Co-coupler)')
plt.plot(pistons_fr/wavel, antinull_bi.mean(1), 'x', c=colors[2], label='Antinull (Co-coupler)')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Atm piston (wl0)')
plt.ylabel('Flux (count)')

plt.figure()
plt.plot(time_fr, noisy_data[1].mean(1), '--', c=colors[0], label='Null')
plt.plot(time_fr, antinull.mean(1), c=colors[0], label='Antinull')
plt.plot(time_fr, noisy_data_noft[1].mean(1), '.', c=colors[1], label='Null (no FT)')
plt.plot(time_fr, antinull_noft.mean(1), 'x', c=colors[1], label='Antinull (no FT)')
plt.plot(time_fr, noisy_data_bi[0].mean(1), '.', c=colors[2], label='Null (Co-coupler)')
plt.plot(time_fr, antinull_bi.mean(1), 'x', c=colors[2], label='Antinull (Co-coupler)')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Flux (count)')

plt.figure()
plt.plot(pistons_fr/wavel, null_depth.mean(1), '.', label='Null depth')
plt.plot(pistons_fr/wavel, null_depth_noft.mean(1), 'x', label='Null depth (no FT)')
plt.plot(pistons_fr/wavel, null_depth_bi.mean(1), '+', label='Null depth (Co-coupler)')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Atm piston (wl0)')
plt.ylabel('Null depth')

plt.figure()
plt.plot(time_fr, null_depth.mean(1), '.', label='Null depth')
plt.plot(time_fr, null_depth_noft.mean(1), 'x', label='Null depth (no FT)')
plt.plot(time_fr, null_depth_bi.mean(1), '+', label='Null depth (Co-coupler)')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Null depth')