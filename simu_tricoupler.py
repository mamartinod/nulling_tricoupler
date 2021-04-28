# np.array#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:05:38 2021

@author: Marc-Antoine Martinod

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

Interferometric equations for achromatic balanced tricoupler are simple:
    central output: classic interferometric equation with all terms divided by 3
    left: same as central with -2pi/3 shift
    right: same as central with +2pi/3 shift

The script can be improved by adding chromaticity in the tricoupler and
the codirectional coupler.

How to use it:
    * Change the values in **Settings** depending on what you want to simulate
    * Run
    * For finer tuning, check the other parameters of the other sections, especially the servo loop one to deactivate the servo if you don't care.

About the turbulence:
    * a Kolmogorov - Von Karman phase screen is created
    * the screen is shifted after each element of time
    * to avoid wrapping, a new mask is created once the previous mask is entirely rolled

About the detector:
    * 2 detectors are available: C-Red 1 and C-Red 2, uncomment the desired one and comment the other
    * photon noise, dark current and read-out noise is simulated
    * the phase screen is in distance unit which is achromatic, on the contrary of phase unit which is chromatic (\propto 2\pi / \lambda)

About phase of the wavefronts:
    * average value of the phase screen in a sub-pupil gives the atmospheric piston of this pupil
    * the differential piston refers to the difference of pistons between pupils A and B
    * the servo loop applies the differential piston to one beam. It corresponds to the instrumental delay applied to the beam (e.g. by moving a mirror).

About the combiners:
    * Possibility to add photometric outputs
    * Transfer matrices of directional coupler or tricoupler have 5 rows to ease the plotting
    * By convention, the last rows are the photometric outputs, it should be kept this way as much as possible
"""

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
# import cupy as cp
import simu_tricoupler_lib as lib
from astropy.time import Time
from datetime import datetime

class Coupler(object):
    def __init__(self):
        # =============================================================================
        # Settings
        # =============================================================================
        # To save data
        self.save = False
        # To plot data
        self.plot = False
        # To simulate photon noise
        self.activate_photon_noise = False
        # To simulate the detector noise
        self.activate_detector_noise = False
        # To simulate the flux of a star, otherwise the incoming fluxes are equal to 1
        self.activate_flux = False
        # Use an achromatic phase mask instead of air-delaying (chromatic) one beam with respect to the other
        self.activate_achromatic_phase_shift = False
        # To activate turbulence
        self.activate_turbulence = True
        # Use of photometric outputs
        activate_photometric_output = True

        # Set the seed to an integer for repeatable simulation, to ``None'' otherwise
        self.seed = 1

        # Size in pixels of the pupil of the telescope (which will be later cropped into subpupils)
        self.sz = 256
        # Oversampling the array of the telescope for various use (e.g. bigger phase screen to mimic turbulence without wrapping)
        self.oversz = 4

        # Set the values of the phase masks for both tricoupler and directional coupler for each beam
        self.achromatic_phasemask_tricoupler = np.array([np.pi, 0.])
        self.achromatic_phasemask_cocoupler = np.array([np.pi/2, 0.])

        # =============================================================================
        # Telescope and AO parameters
        # =============================================================================
        self.tdiam = 8.2 # Diameter of the telescope (in meter)
        subpup_diam = 1. # Diameter of the sub-pupils (in meter)
        # baseline = 5.55 # Distance between two sub-pupils (in meter) ORIGINAL
        baseline = 6.45 # Distance between two sub-pupils (in meter)
        self.fc_scex = 19.5 # 19.5 l/D is the cutoff frequency of the DM for 1200 modes corrected (source: Vincent)
        self.wavel_r0 = 0.5e-6 # wavelength where r0 is measured (in meters)
        wavel = 1.6e-6 # Wavelength of observation (in meter)
        # bandwidth = 0.2e-6 # Bandwidth around the wavelength of observation (in meter) ORIGINAL
        bandwidth = 0.3e-6 # Bandwidth around the wavelength of observation (in meter)
        self.dwl = 5e-9 # Width of one spectral channel (in meter)

        self.metre_to_pixel = self.sz / self.tdiam # scale factor converting the meter-size in pixel, in pix/m
        self.ao_correc = 8. # How well the AO flattens the wavefront
        self.sub_pupil_centres = np.array([[-1.86333333, -3.2147619 ], # Centres of the subpupils in metres, provided by Barnaby
                                       [-1.86333333,  3.2352381 ],
                                       [ 1.86333333,  3.2352381 ],
                                       [ 3.11238095,  1.0852381 ]])
        self.sub_pupil_centres = (self.sub_pupil_centres * self.metre_to_pixel).astype(int) + self.sz//2

        # =============================================================================
        # Atmo parameters
        # =============================================================================
        self.r0 = 0.16  # Fried parameter at wavelength wavel_r0 (in meters), the bigger, the better the seeing is
        self.ll = self.tdiam * self.oversz # Physical extension of the wavefront (in meter)
        self.L0 = 1e15 # Outer scale for the model of turbulence, keep it close to infinity for Kolmogorov turbulence (the simplest form) (in meter)
        self.wind_speed = 9.8 # speed of the wind (in m/s)
        self.angle = 45 # Direction of the wind

        # =============================================================================
        # Acquisition and detector parameters
        # =============================================================================
        fps = 2000 # frame rate (in Hz)
        delay = 0.001 # delay of the servo loop (in second)
        dit = 1 / fps # Detector Integration Time, time during which the detector collects photon (in second)
        timestep = 1e-2 # time step of the simulation (in second)
        time_obs = 10 # duration of observation (in second)

        # Let's define the axe of time on which any event will happened (turbulence, frame reading, servo loop)
        self.timeline = np.around(np.arange(0, time_obs+delay, timestep, dtype=np.float32), int(-np.log10(timestep)))
        self.time_counter = 0

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
        tricoupler = np.asarray(np.array([[1/3**0.5                         , 1/3**0.5 * np.exp(1j* 2*np.pi/3)  , 0., 0.],
                                          [1/3**0.5 * np.exp(1j* 2*np.pi/3) , 1/3**0.5 * np.exp(1j* 2*np.pi/3)  , 0., 0.],
                                          [1/3**0.5 * np.exp(1j* 2*np.pi/3) , 1/3**0.5                          , 0., 0.],
                                          [0.                               , 0.                                , 1., 0.],
                                          [0.                               , 0.                                , 0., 1.]], dtype=np.complex64))

        # Splitter before combining the beams to get the photometric tap
        tri_splitter = np.array([[1-coeff_tri, 0.],
                                 [0.         , 1-coeff_tri],
                                 [coeff_tri  , 0.],
                                 [0.         , coeff_tri]], dtype=np.float32)
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
        bicoupler = np.asarray(np.array([[1.0/2**0.5                        , 1.0/2**0.5 * np.exp(-1j* np.pi/2) , 0., 0.],
                                         [1.0/2**0.5 * np.exp(-1j* np.pi/2) , 1.0/2**0.5                        , 0., 0.],
                                         [0.0                               , 0.0                               , 0., 0.],
                                         [0.0                               , 0.                                , 1., 0.],
                                         [0.                                , 0.                                , 0., 1.]], dtype=np.complex64))

        # Splitter before combining the beams to get the photometric tap
        bi_splitter = np.array([[1-coeff_bi , 0.],
                                 [0.        , 1-coeff_bi],
                                 [coeff_bi  , 0.],
                                 [0.        , coeff_bi]], dtype=np.float32)

        # The coefficients below are given for intensities although we deal with amplitude of wavefront
        # so a square root must be applied to them
        bi_splitter = bi_splitter**0.5

        # Total combiner
        combiner_bi = bicoupler@bi_splitter


        # =============================================================================
        # Convert physical quantities in pixel
        # =============================================================================
        pupil_rad = self.sz // 2
        self.subpup_diamp = subpup_diam / self.tdiam * self.sz
        baselinep = baseline / self.tdiam * self.sz

        # =============================================================================
        # Servo loop parameters
        # =============================================================================
        servo_gain = 0.
        servo_int = 0.# 1.#0.05 # 0: no fringe tracking, 1.: fringe trcking at high flux, 0.05: fringe tracking at low flux (SNR<=2)
        err_int = np.array([0.], dtype=np.float32)
        diff_piston_command = np.array([0.], dtype=np.float32)

        if not self.activate_achromatic_phase_shift:
            DIFF_PISTON_REF = wavel / 2
            DIFF_PISTON_REFBI = wavel / 4
            self.achromatic_phasemask_tricoupler[:] = 0.
            self.achromatic_phasemask_cocoupler[:] = 0.
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
        self.TIME_OFFSET = 0. # time offset of the shift of the mask after creation of a new one
        count_delay = 1
        count_dit = 1
        debug = []
        # =============================================================================
        # Run
        # =============================================================================
        # Create the spectral dispersion
        self.wl = np.arange(wavel-bandwidth/2, wavel+bandwidth/2, self.dwl, dtype=np.float32)

        # Create the sub-pupil
        self.pupil = lib.createSubPupil(self.sz, int(self.subpup_diamp//2), self.sub_pupil_centres, 5, norm=False)
        self.pupil = np.array(self.pupil, dtype=np.float32)

        # Create the phase screen.
        if self.activate_turbulence:
            self.phs_screen = lib.generatePhaseScreen(self.wavel_r0, self.sz*self.oversz, self.ll, self.r0, self.L0, fc=self.fc_scex, correc=self.ao_correc, pdiam=self.tdiam, seed=self.seed)
            self.phs_screen = np.array(self.phs_screen, dtype=np.float32)
        else:
            self.phs_screen = np.zeros((self.sz*self.oversz, self.sz*self.oversz), dtype=np.float32)

    def step(self):
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


        i_out = np.zeros((5, self.wl.size), dtype=np.float32)
        i_out_noft = np.zeros((5, self.wl.size), dtype=np.float32)
        i_out_bi = np.zeros((5, self.wl.size), dtype=np.float32)

        # =============================================================================
        # Loop over simulated time
        # =============================================================================
        t = self.timeline[self.time_counter]
        self.time_counter += 1

        phs_screen_moved, xyshift = lib.movePhaseScreen(self.phs_screen, self.wind_speed, self.angle, t-self.TIME_OFFSET, self.metre_to_pixel)
        if xyshift[0] > self.phs_screen.shape[0] or xyshift[1] > self.phs_screen.shape[1]:
            if self.seed != None:
                self.seed += 20
            self.phs_screen = lib.generatePhaseScreen(self.wavel_r0, self.sz*self.oversz, self.ll, self.r0, self.L0, fc=self.fc_scex, correc=9, pdiam=self.tdiam, seed=None)
            self.phs_screen = np.array(self.phs_screen, dtype=np.float32)
            self.TIME_OFFSET = t

        shifts.append(xyshift)
        # We stay in phase space hence the simple multiplication below to crop the wavefront.
        phs_pup = self.pupil * phs_screen_moved[phs_screen_moved.shape[0]//2-self.sz//2:phs_screen_moved.shape[0]//2+self.sz//2,\
                                                phs_screen_moved.shape[1]//2-self.sz//2:phs_screen_moved.shape[1]//2+self.sz//2]

        # Measure the piston of the subpupils
        pistons = lib.calculate_pistons(phs_pup, self.sub_pupil_centres, int(self.subpup_diamp//2))

        # Measure the differential atmospheric piston
        # Iterate over each baseline ***
        baseline = 0
        for i, piston_pupA in enumerate(pistons):
            for j in range(i+1, len(pistons)):
                piston_pupB = pistons[j]

                diff_piston_atm = piston_pupA - piston_pupB
                diff_pistons_atm.append(diff_piston_atm)

                # Total differential piston, including the instrumental air-delay between the beams
                if diff_piston_atm_original is None:
                    self.diff_piston_atm_original = diff_piston_atm
                    self.diff_piston_command_original = diff_piston_command
                diff_piston = np.array([DIFF_PISTON_REF + self.diff_piston_atm_original[baseline]], dtype=np.float32) # pupil A - pupil B
                diff_piston_corrected = diff_piston - self.diff_piston_command_original[baseline]

                injection = np.array([lib.calculateInjection(phs_pup[:,:self.sz//2][self.pupil[:,:self.sz//2]!=0], self.wl), \
                                      lib.calculateInjection(phs_pup[:,self.sz//2:][self.pupil[:,self.sz//2:]!=0], self.wl)])

                # Input wavefronts
                a_in = np.array([np.exp(1j*2*np.pi/self.wl*(piston_pupA + diff_piston_corrected) + 1j*self.achromatic_phasemask_tricoupler[0]),\
                                 np.exp(1j*2*np.pi/self.wl*piston_pupB                           + 1j*self.achromatic_phasemask_tricoupler[1])],\
                                dtype=np.complex64)
                if self.activate_flux:
                    a_in *= injection**0.5 * star_photons**0.5
                a_out = combiner_tri@a_in # Deduce the outcoming wavefront after the integrated-optics device
                i_out += np.abs(a_out)**2

                # Same but with no fringe tracking
                a_in_noft = np.array([np.exp(1j*2*np.pi/self.wl*(piston_pupA)    + 1j*self.achromatic_phasemask_tricoupler[0]),\
                                      np.exp(1j*2*np.pi/self.wl*piston_pupB      + 1j*self.achromatic_phasemask_tricoupler[1])],\
                                     dtype=np.complex64)
                if self.activate_flux:
                    a_in_noft *= injection**0.5 * star_photons**0.5
                a_out_noft = combiner_tri@a_in_noft
                i_out_noft += np.abs(a_out_noft)**2

                # Same but with codirectional coupler
                diff_piston_bi = np.array([DIFF_PISTON_REFBI + self.diff_piston_atm_original[baseline]], dtype=np.float32) # pupil 1 - pupil 2
                diff_pistons_bi.append(diff_piston_bi)
                a_in_bi = np.array([np.exp(1j*2*np.pi/self.wl*(piston_pupA + diff_piston_bi) + 1j*self.achromatic_phasemask_cocoupler[0]),\
                                    np.exp(1j*2*np.pi/self.wl*(piston_pupB)                  + 1j*self.achromatic_phasemask_cocoupler[1])],\
                                   dtype=np.complex64)
                if self.activate_flux:
                    a_in_bi *= injection**0.5 * star_photons**0.5
                a_out_bi = combiner_bi@a_in_bi
                i_out_bi += np.abs(a_out_bi)**2

                if count_dit < dit/timestep:
                    count_dit += 1
                else:
                    i_out /= count_dit
                    data.append(i_out)
                    noisy_i_out = lib.addNoise(i_out, QE, read_noise, ndark, fps, self.activate_photon_noise, self.activate_detector_noise, enf)
                    noisy_data.append(noisy_i_out)

                    # No fringe tracking
                    i_out_noft /= count_dit
                    data_noft.append(i_out_noft)
                    noisy_i_out_noft = lib.addNoise(i_out_noft, QE, read_noise, ndark, fps, self.activate_photon_noise, self.activate_detector_noise, enf)
                    noisy_data_noft.append(noisy_i_out_noft)

                    # Codirectional coupler
                    i_out_bi /= count_dit
                    data_bi.append(i_out_bi)
                    noisy_i_out_bi = lib.addNoise(i_out_bi, QE, read_noise, ndark, fps, self.activate_photon_noise, self.activate_detector_noise, enf)
                    noisy_data_bi.append(noisy_i_out_bi)

                    # Store some data
                    diff_pistons_fr.append(diff_piston_atm)
                    injections_fr.append(injection)
                    time_fr.append(t)

                    if count_delay == count_dit: # Capture the first frame of the double cycles on DIT and Delay to send to fringe trakcer
                        noisy_i_out_toft = np.array(noisy_i_out, dtype=np.float32)
                        noisy_i_out_noft_toft = np.array(noisy_i_out_noft, dtype=np.float32)
                        diff_piston_toft = np.array(diff_piston_atm)
                        injection_toft = np.array(injection)

                    # Reinit the cycle of integration of a frame
                    i_out = np.zeros_like(i_out)
                    i_out_noft = np.zeros_like(i_out_noft)
                    i_out_bi = np.zeros_like(i_out_bi)
                    count_dit = 1


                # if couplertype == 'tricoupler':
                if count_delay < delay/timestep:
                    count_delay += 1
                else:
                    # print('Delay', count_dit, count_delay, t)
                    # With fringe tracking and application of the measured piston
                    diff_piston_meas = lib.measurePhase3(noisy_i_out_toft, self.wl)
                    diff_pistons_measured.append(diff_piston_meas)
                    diff_piston_error = diff_piston_meas - DIFF_PISTON_REF
                    err_int += diff_piston_error
                    diff_piston_command = servo_int * err_int + servo_gain * diff_piston_error

                    diff_piston_meas_noft = lib.measurePhase3(noisy_i_out_noft_toft, self.wl)
                    diff_pistons_measured_noft.append(diff_piston_meas_noft)

                    diff_pistons_ft.append(diff_piston_toft)
                    injections_ft.append(injection_toft)
                    time_ft.append(t)
                    count_delay = 1

                baseline += 1

    def format_data(self):
        # =============================================================================
        # Format data
        # =============================================================================
        diff_pistons_atm = np.array(diff_pistons_atm)
        diff_pistons_bi = np.array(diff_pistons_bi)
        diff_pistons_ft = np.array(diff_pistons_ft)
        diff_pistons_fr = np.array(diff_pistons_fr)
        diff_pistons_measured = np.array(diff_pistons_measured)
        diff_pistons_measured_noft = np.array(diff_pistons_measured_noft)
        injections = np.array(injections)
        injections_ft = np.array(injections_ft)
        injections_fr = np.array(injections_fr)

        shifts = np.array(shifts)
        time_fr = np.array(time_fr)
        time_ft = np.array(time_ft)

        diff_pistons_atm = np.squeeze(diff_pistons_atm)
        diff_pistons_bi = np.squeeze(diff_pistons_bi)
        diff_pistons_ft = np.squeeze(diff_pistons_ft)
        diff_pistons_fr = np.squeeze(diff_pistons_fr)
        diff_pistons_measured = np.squeeze(diff_pistons_measured)
        diff_pistons_measured_noft = np.squeeze(diff_pistons_measured_noft)

        data = np.array(data)
        noisy_data = np.array(noisy_data)
        data_bi = np.array(data_bi)
        data_noft = np.array(data_noft)
        noisy_data_noft = np.array(noisy_data_noft)
        noisy_data_bi = np.array(noisy_data_bi)

        data = np.transpose(data, (1,0,2))
        data_noft = np.transpose(data_noft, (1,0,2))
        data_bi = np.transpose(data_bi, (1,0,2))
        noisy_data = np.transpose(noisy_data, (1,0,2))
        noisy_data_noft = np.transpose(noisy_data_noft, (1,0,2))
        noisy_data_bi = np.transpose(noisy_data_bi, (1,0,2))

        print(data_bi.shape)
        input('')

    def calculate_quantities(self):
        # =============================================================================
        # Calculate other quantities
        # =============================================================================
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

    def save_data(self):
        # =============================================================================
        # Save data
        # =============================================================================
        if self.save:
            np.savez('results/%s_mag_%s'%(timestamp, magnitude), noisy_data=noisy_data, data=data,
                     data_noft=data_noft, noisy_data_noft=noisy_data_noft, data_bi=data_bi,
                     noisy_data_bi=noisy_data_bi, diff_pistons_atm=diff_pistons_atm, diff_pistons_fr=diff_pistons_fr,
                     diff_pistons_measured=diff_pistons_measured, diff_pistons_measured_noft=diff_pistons_measured_noft,
                     diff_pistons_ft=diff_pistons_ft, null_depth=null_depth, null_depth_noft=null_depth_noft,
                     null_depth_bi=null_depth_bi, det_charac=np.array([QE, read_noise, ndark, fps]),
                     servo=np.array([delay, servo_gain, servo_int]), atm=np.array([self.r0, self.wind_speed, self.angle]))

    def plot_data(self):
        # =============================================================================
        # Display and plot
        # =============================================================================
        print('Std piston no FT (input, measured)', diff_pistons_atm.std()/wavel, diff_pistons_measured_noft.std()/wavel)
        print('Std piston with FT', diff_pistons_measured.std()/wavel)
        print('Med and std null depth with FT', np.median(null_depth.mean(1)[(null_depth.mean(1)>=0.)&(null_depth.mean(1)<=1)]), np.std(null_depth.mean(1)[(null_depth.mean(1)>=0.)&(null_depth.mean(1)<=1)]))
        print('Med and std null depth no FT', np.median(null_depth_noft.mean(1)[(null_depth_noft.mean(1)>=0.)&(null_depth_noft.mean(1)<=1)]), np.std(null_depth_noft.mean(1)[(null_depth_noft.mean(1)>=0.)&(null_depth_noft.mean(1)<=1)]))
        print('Med and std null depth with BI', np.median(null_depth_bi.mean(1)[(null_depth_bi.mean(1)>=0.)&(null_depth_bi.mean(1)<=1)]), np.std(null_depth_bi.mean(1)[(null_depth_bi.mean(1)>=0.)&(null_depth_bi.mean(1)<=1)]))
        print('---')
        print('Med and std null depth with FT', np.median(null_depth.mean(1)), np.std(null_depth.mean(1)))
        print('Med and std null depth no FT', np.median(null_depth_noft.mean(1)), np.std(null_depth_noft.mean(1)))
        print('Med and std null depth with BI', np.median(null_depth_bi.mean(1)), np.std(null_depth_bi.mean(1)))

        if self.plot:
            # if couplertype == 'tricoupler':
            plt.figure(1)
            plt.plot(timeline, diff_pistons_atm/wavel, label='Atmospheric differential piston')
            plt.plot(time_ft, diff_pistons_measured/wavel, '.', label='Corrected differential piston')
            plt.plot(time_ft, (diff_pistons_measured_noft)/wavel, 'x', label='Measured atmospheric differential piston')
            plt.plot(timeline, diff_pistons_bi/wavel, '.', label='Differential piston in Co-coupler')
            plt.plot(timeline, DIFF_PISTON_REF/wavel*np.ones(timeline.size), '-', c='k', label='Differential piston reference')
            plt.plot(timeline, DIFF_PISTON_REFBI/wavel*np.ones(timeline.size), '--', c='k', label='Differential piston reference (co-coupler)')
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('Delay (count)')
            plt.ylabel('Piston (wl0)')

            plt.figure(2)
            plt.plot(diff_pistons_atm/wavel, diff_pistons_atm/wavel, label='Atmospheric differential piston')
            plt.plot(diff_pistons_ft/wavel, diff_pistons_measured/wavel, '.', label='Corrected differential piston')
            plt.plot(diff_pistons_ft/wavel, diff_pistons_measured_noft/wavel, '.', label='Measured atmospheric differential piston')
            plt.plot(diff_pistons_atm/wavel, diff_pistons_bi/wavel, '.', label='Differential piston in Co-coupler')
            plt.plot(diff_pistons_ft/wavel, DIFF_PISTON_REF/wavel*np.ones_like(diff_pistons_ft), '-', c='k', label='Differential piston reference')
            plt.plot(diff_pistons_ft/wavel, DIFF_PISTON_REFBI/wavel*np.ones(diff_pistons_ft.size), '--', c='k', label='Differential piston reference (co-coupler)')
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('Atm piston (wl0)')
            plt.ylabel('Piston (wl0)')

            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.figure(3)
            plt.plot(diff_pistons_fr/wavel, noisy_data[1].mean(1), '.', c=colors[0], label='Null')
            plt.plot(diff_pistons_fr/wavel, antinull.mean(1), 'x', c=colors[0], label='Antinull')
            plt.plot(diff_pistons_fr/wavel, noisy_data_noft[1].mean(1), '.', c=colors[1], label='Null (no FT)')
            plt.plot(diff_pistons_fr/wavel, antinull_noft.mean(1), 'x', c=colors[1], label='Antinull (no FT)')
            plt.plot(diff_pistons_fr/wavel, noisy_data_bi[0].mean(1), '.', c=colors[2], label='Null (Co-coupler)')
            plt.plot(diff_pistons_fr/wavel, antinull_bi.mean(1), 'x', c=colors[2], label='Antinull (Co-coupler)')
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('Atm piston (wl0)')
            plt.ylabel('Flux (count)')

            plt.figure(4)
            plt.plot(time_fr, noisy_data[1].mean(1), '--', c=colors[0], label='Null output')
            plt.plot(time_fr, antinull.mean(1), c=colors[0], label='Antinull output')
            plt.plot(time_fr, noisy_data_noft[1].mean(1), '.', c=colors[1], label='Null output (no FT)')
            plt.plot(time_fr, antinull_noft.mean(1), 'x', c=colors[1], label='Antinull output (no FT)')
            plt.plot(time_fr, noisy_data_bi[0].mean(1), '.', c=colors[2], label='Null output (Co-coupler)')
            plt.plot(time_fr, antinull_bi.mean(1), 'x', c=colors[2], label='Antinull output (Co-coupler)')
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('Time (s)')
            plt.ylabel('Flux (count)')

            plt.figure(5)
            plt.plot(diff_pistons_fr/wavel, null_depth.mean(1), '.', label='Null depth')
            plt.plot(diff_pistons_fr/wavel, null_depth_noft.mean(1), 'x', label='Null depth (no FT)')
            plt.plot(diff_pistons_fr/wavel, null_depth_bi.mean(1), '+', label='Null depth (Co-coupler)')
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('Atm piston (wl0)')
            plt.ylabel('Null depth')

            plt.figure(6)
            plt.plot(time_fr, null_depth.mean(1), '.', label='Null depth')
            plt.plot(time_fr, null_depth_noft.mean(1), 'x', label='Null depth (no FT)')
            plt.plot(time_fr, null_depth_bi.mean(1), '+', label='Null depth (Co-coupler)')
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('Time (s)')
            plt.ylabel('Null depth')

            plt.show()
