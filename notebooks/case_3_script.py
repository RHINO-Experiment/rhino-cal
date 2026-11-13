"""
Script for running case 3 and aquiring chains.
"""

def main():
    import numpy as np
    import sys
    sys.path.insert(0, "..")
    from sampler.reader import ObservationReader, ReflectionReader


    # Read in data
    data_path = '../simulated_data/case_3'

    observation = ObservationReader(f'{data_path}/sim_data.hdf5')
    print(np.unique(observation.states_array))



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

    gains_basis = BasisConstructor(observation.times,
                               observation.freqs,
                               n_time_modes=5,
                               n_freq_modes=5,
                               basis_function='legendre',
                               time_norm=1*(observation.times[-1] - observation.times[0]),
                               frequency_norm=np.median(observation.freqs),
                               zero_time=True)

    unc_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=5,
                                n_freq_modes=5,
                                basis_function='legendre',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    cos_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=5,
                                n_freq_modes=5,
                                basis_function='legendre',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    sin_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=5,
                                n_freq_modes=5,
                                basis_function='legendre',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    rx_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=5,
                                n_freq_modes=5,
                                basis_function='legendre',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)


    # Set up thermal bases
    # cal temperature set up: give the same time orders
    cal_n_time_modes = 6


    ambient_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=cal_n_time_modes,
                                n_freq_modes=1,
                                basis_function='polynomial',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    ambient_basis_y = BasisConstructor(observation.times,
                                0,
                                n_time_modes=cal_n_time_modes,
                                n_freq_modes=1,
                                basis_function='polynomial',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    heated_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=cal_n_time_modes,
                                n_freq_modes=1,
                                basis_function='polynomial',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    heated_basis_y = BasisConstructor(observation.times,
                                0,
                                n_time_modes=cal_n_time_modes,
                                n_freq_modes=1,
                                basis_function='polynomial',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)
    ####

    antenna_basis = BasisConstructor(observation.times,
                                observation.freqs,
                                n_time_modes=1,
                                n_freq_modes=1,
                                basis_function='polynomial',
                                time_norm=1*(observation.times[-1] - observation.times[0]),
                                frequency_norm=np.median(observation.freqs),
                                zero_time=True)

    from sampler.operator_setup import generate_noise_wave_transfer_matrix
    from sampler.operator_setup import generate_non_smooth_source_matrix, generate_smooth_source_matrix, generate_noise_wave_transfer_matrix
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
                                          source_label='antenna')

    del antenna_basis

    from sampler.operator_setup import concat_and_index_operators

    # X_ant_noisewave = np.concatenate((X_antenna, X_noise_wave), axis=1)

    X_ant_noisewave, _ = concat_and_index_operators([X_antenna,
                                                                            #X_noise_diode,
                                                                            X_noise_wave],
                                                                            label_list=['antenna',
                                                                                        #'noise_diode',
                                                                                        'noise_wave'])

    del X_antenna, X_noise_wave


    X_ambient = generate_smooth_source_matrix(observation, reflections,
                                          source_basis=ambient_basis,
                                          source_label=['internal_load', 'long_open',
                                                        'long_short', 'open', 'short'])

    del ambient_basis
    
    X_heated = generate_smooth_source_matrix(observation, reflections,
                                            source_basis=heated_basis,
                                            source_label='heated_load')

    del heated_basis

    X_calibrator, _ = concat_and_index_operators([X_ambient,
                                                                    X_heated],
                                                                    ['ambient',
                                                                    'heated'])

    del X_ambient, X_heated

    from sampler.priors import generate_smooth_thermistor_operator_matrix

    Y_ambient = generate_smooth_thermistor_operator_matrix(observation,
                                            source_basis=ambient_basis_y,
                                            source_label=['internal_load', 'long_open',
                                                            'long_short', 'open', 'short'])

    del ambient_basis_y


    Y_heated = generate_smooth_thermistor_operator_matrix(observation,
                                            source_basis=heated_basis_y,
                                            source_label='heated_load')

    del heated_basis_y

    Y_calibrator, _ = concat_and_index_operators([Y_ambient,
                                                                    Y_heated],
                                                                    ['ambient',
                                                                    'heated'])

    del Y_ambient, Y_heated

    from sampler.priors import setup_probe_cal_priors


    cal_source_priors, cal_source_prior_inv_cov = setup_probe_cal_priors(observation,
                                            source_temp_dict={'heated':'heated_load',
                                                                'ambient':['internal_load',
                                                                            'open',
                                                                            'short',
                                                                            'long_open',
                                                                            'long_short']},
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

    init_calibrator_solution = solve_gcr(cal_source_lhs, cal_source_rhs,linalg_init=False,
                                        atol=1e-10, rtol=1e-12, maxiter=int(1e6))


    U_gain = gains_basis.basis_matrix
    
    print(U_gain.shape)

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

    del gains_basis

    ### Gibbs Sampling Portion
    from sampler.gibbs_sample import GibbsSampleParams

    savedir = '../chains/case3'
    param_dict = np.load(f'{savedir}/gibbs_sample_case3_test.npz', allow_pickle=True)

    init_gain_amps=param_dict['all_p_gain'][-1]
    init_cal_src_amps=param_dict['all_p_cal_src'][-1]
    init_ant_nw_amps=param_dict['all_p_ant_nw'][-1]

    print(init_gain_amps)
    print(init_cal_src_amps)
    print(init_ant_nw_amps)

    del param_dict

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
                         n_iter=2000,
                         Y_cal_src=Y_calibrator,
                         verbose=False,
                         chi_square_interval=50,
                         QR=True
                         )

    

    np.savez_compressed(f'{savedir}/gibbs_sample_case3_test.npz', **param_dict)

    # param_dict = np.load(f'{savedir}/gibbs_sample_case3.npz', allow_pickle=True)

    print('Done')


if __name__ == '__main__':
    main()
