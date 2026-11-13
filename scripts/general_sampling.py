"""
General script for running the Gibbs Sampler

"""

import argparse
import numpy as np
import sys
import os
import yaml
from pathlib import Path
sys.path.insert(0, "..")
from sampler.reader import ObservationReader, ReflectionReader

def main():
    #  Parse arguments
    parser = argparse.ArgumentParser(
        description='General Script for running RHINO Calibration Gibbs Sampler'
    )

    parser.add_argument(
        '-dd',
        '--data_dir', #FIXME actually point to observational file and reflection files
        help='Directory for the data (also reflections)'
    )
    parser.add_argument(
        '-bc',
        '--basis_config',
        help='Path to the basis config yaml'
    )
    parser.add_argument(
        '-niter',
        '--niter',
        help='Number of Gibbs iterations'
    )

    parser.add_argument(
        '-ic',
        '--initial_chains',
        help='Filepath to initial chains if not starting from scratch.',
        default=None
    )

    parser.add_argument(
        '-cs',
        '--chain_savepath',
        help='Save path for the chains .npz file'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help='Makes verbose',
        action='store_true'
    )

    parser.add_argument(
        '-xsi',
        '--chi_squared_interval',
        help='Iterations before print out of the chi squared value',
        default=50
    )

    parser.add_argument(
        '-p',
        '--profile',
        help='Profile code with cProfile. Specify output path.',
        default=None
    )

    args = parser.parse_args()

    # Read in data
    data_path = args.data_dir

    observation = ObservationReader(f'{data_path}/sim_data.hdf5')
    # print(np.unique(observation.states_array))


    reflection_dict = {'antenna':f'{data_path}/antenna_gamma_src.s2p',
                    'heated_load':f'{data_path}/heated_load_gamma_src.s2p',
                    'internal_load':f'{data_path}/internal_load_gamma_src.s2p',
                    'long_open':f'{data_path}/long_open_gamma_src.s2p',
                    'long_short':f'{data_path}/long_short_gamma_src.s2p',
                    'open':f'{data_path}/open_gamma_src.s2p',
                    'short':f'{data_path}/short_gamma_src.s2p',
                    'receiver':f'{data_path}/gamma_rec.s2p',
                    'noise_diode':f'{data_path}/noise_diode_gamma_src.s2p'}

    reflections = ReflectionReader(component_dict=reflection_dict,
                                output_frequencies=observation.freqs,
                                polynomial_interp=True,
                                polynomial_fitting_order=10)

    from sampler.linear_basis import BasisConstructor

    # Read from .yaml
    with open(args.basis_config) as f:
        basis_config = yaml.safe_load(f)

    gains_basis = BasisConstructor(observation.times,
                               observation.freqs,
                               n_time_modes=basis_config['gainsBasis']['nTimeModes'],
                               n_freq_modes=basis_config['gainsBasis']['nFreqModes'],
                               basis_function=basis_config['gainsBasis']['basisFunction'],
                               time_norm=1*(observation.times[-1] - observation.times[0]),
                               frequency_norm=np.median(observation.freqs),
                               zero_time=True)

    unc_basis = BasisConstructor(observation.times,
                                observation.freqs,
                               n_time_modes=basis_config['uncBasis']['nTimeModes'],
                               n_freq_modes=basis_config['uncBasis']['nFreqModes'],
                               basis_function=basis_config['uncBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    cos_basis = BasisConstructor(observation.times,
                                observation.freqs,
                               n_time_modes=basis_config['cosBasis']['nTimeModes'],
                               n_freq_modes=basis_config['cosBasis']['nFreqModes'],
                               basis_function=basis_config['cosBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    sin_basis = BasisConstructor(observation.times,
                                observation.freqs,
                               n_time_modes=basis_config['sinBasis']['nTimeModes'],
                               n_freq_modes=basis_config['sinBasis']['nFreqModes'],
                               basis_function=basis_config['sinBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    rx_basis = BasisConstructor(observation.times,
                                observation.freqs,
                               n_time_modes=basis_config['rxBasis']['nTimeModes'],
                               n_freq_modes=basis_config['rxBasis']['nFreqModes'],
                               basis_function=basis_config['rxBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)


    # Set up thermal bases
    # cal temperature set up: give the same time orders
    # cal_n_time_modes = 6


    ambient_basis = BasisConstructor(observation.times,
                                observation.freqs,
                               n_time_modes=basis_config['ambientBasis']['nTimeModes'],
                                n_freq_modes=1,
                               basis_function=basis_config['ambientBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    ambient_basis_y = BasisConstructor(observation.times,
                                0,
                                n_time_modes=basis_config['ambientBasis']['nTimeModes'],
                                n_freq_modes=1,
                                basis_function=basis_config['ambientBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    heated_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=basis_config['heatedBasis']['nTimeModes'],
                                n_freq_modes=1,
                                basis_function=basis_config['heatedBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    heated_basis_y = BasisConstructor(observation.times,
                                0,
                                n_time_modes=basis_config['heatedBasis']['nTimeModes'],
                                n_freq_modes=1,
                                basis_function=basis_config['heatedBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)
    ####

    antenna_basis = BasisConstructor(observation.times,
                                observation.freqs,
                               n_time_modes=basis_config['antennaBasis']['nTimeModes'],
                               n_freq_modes=basis_config['antennaBasis']['nFreqModes'],
                               basis_function=basis_config['antennaBasis']['basisFunction'],
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    from sampler.operator_setup import generate_smooth_source_matrix, generate_noise_wave_transfer_matrix
    # Noise Wave Operator
    X_noise_wave, nw_slice_dict = generate_noise_wave_transfer_matrix(observation,
                                                                reflections,
                                                                unc_basis,
                                                                cos_basis,
                                                                sin_basis,
                                                                rx_basis)

    del unc_basis, cos_basis, sin_basis, rx_basis
    
    print(nw_slice_dict)
    print(X_noise_wave.shape)

    X_antenna = generate_smooth_source_matrix(observation,
                                          reflections,
                                          source_basis=antenna_basis,
                                          source_label=basis_config['antennaBasis']['sources'])

    del antenna_basis

    from sampler.operator_setup import concat_and_index_operators


    X_ant_noisewave, _ = concat_and_index_operators([X_antenna,
                                                                            #X_noise_diode,
                                                                            X_noise_wave],
                                                                            label_list=['antenna',
                                                                                        #'noise_diode',
                                                                                        'noise_wave'])

    del X_antenna, X_noise_wave


    X_ambient = generate_smooth_source_matrix(observation, reflections,
                                          source_basis=ambient_basis,
                                          source_label=basis_config['ambientBasis']['sources'])

    del ambient_basis
    
    X_heated = generate_smooth_source_matrix(observation, reflections,
                                            source_basis=heated_basis,
                                            source_label=basis_config['heatedBasis']['sources'])

    del heated_basis

    X_calibrator, _ = concat_and_index_operators([X_ambient,
                                                  X_heated],
                                                  ['ambient',
                                                   'heated'])

    del X_ambient, X_heated

    from sampler.priors import generate_smooth_thermistor_operator_matrix

    Y_ambient = generate_smooth_thermistor_operator_matrix(observation,
                                            source_basis=ambient_basis_y,
                                            source_label=basis_config['ambientBasis']['sources'])

    del ambient_basis_y


    Y_heated = generate_smooth_thermistor_operator_matrix(observation,
                                            source_basis=heated_basis_y,
                                            source_label=basis_config['heatedBasis']['sources'])

    del heated_basis_y

    Y_calibrator, _ = concat_and_index_operators([Y_ambient,
                                                  Y_heated],
                                                  ['ambient',
                                                   'heated'])

    del Y_ambient, Y_heated

    from sampler.priors import setup_probe_cal_priors


    cal_source_priors, cal_source_prior_inv_cov = setup_probe_cal_priors(
        observation,
        source_temp_dict={'heated':basis_config['heatedBasis']['sources'],
                          'ambient':basis_config['ambientBasis']['sources']},
        temp_std=0.01)


    ### Attempt a Weiner solution fit

    from sampler.gibbs_sample import construct_gcr_lhs, construct_gcr_rhs, solve_gcr


    cal_source_lhs = construct_gcr_lhs(linear_operator=Y_calibrator,
                                    inv_data_cov=cal_source_prior_inv_cov,
                                    inv_prior_cov=None,)
    cal_source_rhs = construct_gcr_rhs(linear_operator=Y_calibrator,
                                    data=cal_source_priors,
                                    inv_data_cov=cal_source_prior_inv_cov,
                                    omega_data=np.zeros_like(cal_source_priors),
                                    priors=None,
                                    inv_prior_cov=None,
                                    weiner_solution=True,)

    init_cal_src_amps = solve_gcr(cal_source_lhs, cal_source_rhs,linalg_init=False,
                                        atol=1e-10, rtol=1e-12, maxiter=int(1e6))

    U_gain = gains_basis.basis_matrix

    # get an initial guess of the gains params

    def init_gain_guess(observation_obj: ObservationReader,
                        reflections_obj: ReflectionReader,
                        gains_basis: BasisConstructor,
                        noise_source_label: str = "heated_load",
                        load_label: str = 'internal_load',
                        noise_source_temp: int | None = None):

        noise_source_mask = (observation_obj.states_array == noise_source_label)
        load_mask = (observation_obj.states_array == load_label)
        p_ns = observation_obj.data_waterfall[noise_source_mask].mean(axis=0)
        p_l = observation_obj.data_waterfall[load_mask].mean(axis=0)

        if noise_source_temp is not None: t_ns = noise_source_temp
        else:
            t_ns = np.mean(observation_obj.temperatures['heated'])
        t_l = np.mean(observation_obj.temperatures['ambient'])

        gamma_rec = reflections_obj.s11_dict['receiver']

        mean_gain = (p_ns - p_l) / ((t_ns - t_l) * (1 - np.power(np.abs(gamma_rec), 2)))

        fitting_gain = np.zeros_like(observation.data_waterfall)
        fitting_gain[:,:] = mean_gain
        fitting_gain = fitting_gain.flatten()
        init_amp_guess = np.linalg.lstsq(gains_basis.basis_matrix, fitting_gain)[0]

        return init_amp_guess
    

    init_gain_amps = init_gain_guess(observation, reflections, gains_basis)
    init_ant_nw_amps=np.ones_like(X_ant_noisewave[0])
    del gains_basis

    ### Gibbs Sampling Portion
    from sampler.gibbs_sample import GibbsSampleParams

    if args.initial_chains is not None:
        param_dict = np.load(args.initial_chains, allow_pickle=True)
    
        init_gain_amps=param_dict['all_p_gain'][-1]
        init_cal_src_amps=param_dict['all_p_cal_src'][-1]
        init_ant_nw_amps=param_dict['all_p_ant_nw'][-1]
        del param_dict

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Starting Sampling with following amplitudes')
    print(init_gain_amps)
    print(init_cal_src_amps)
    print(init_ant_nw_amps)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Sampling over {len(init_ant_nw_amps)+len(init_cal_src_amps)+len(init_gain_amps)} parameters')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    

    if args.profile is not None:
        import cProfile
        profiler = cProfile.Profile
        profiler.enable()

    param_dict = GibbsSampleParams(
                        data=observation.data_waterfall.flatten(),
                         N=observation.Nw.flatten(),
                         U_gain=U_gain,
                         X_ant_nw=X_ant_noisewave,
                         X_cal_src=X_calibrator,
                         init_gain_amps=init_gain_amps,
                         init_cal_src_amps=init_cal_src_amps,
                         init_ant_nw_amps=init_ant_nw_amps,
                         cal_src_priors=cal_source_priors,
                         inv_cal_src_prior_cov=cal_source_prior_inv_cov,
                         n_iter=int(args.niter),
                         Y_cal_src=Y_calibrator,
                         verbose=args.verbose,
                         chi_square_interval=args.chi_squared_interval,
                         QR=True
                         )
    
    chainssavepath = args.chain_savepath

    # check filepath exists
    chainssavepath = Path(chainssavepath)
    chainssavepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(chainssavepath, **param_dict)

    if args.profile is not None:
        profiler.disable()
        profiler.dump_stats(args.profile)

    print('Done')


if __name__ == '__main__':
    main()
