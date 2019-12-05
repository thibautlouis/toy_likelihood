"""
Inspired from https://github.com/xgarrido/beyondCV/blob/master/beyondCV/beyondCV.py
"""
import numpy as np
import sacc
from colorize import PrintInColor
try:
    import likelihood_utils
except:
    from pslike import likelihood_utils

def prepare_data(setup, verbose=False):
    data = setup["data"]
    s = sacc.Sacc.load_fits(data['input_spectra'])
    experiments = sorted(data['experiments'])

    try:
        default_cuts = data['defaults']
    except:
        raise KeyError('You must provide a list of default cuts')

    # Translation betwen TEB and sacc C_ell types
    pol_dict = {'T': '0',
                'E': 'e',
                'B': 'b'}

    indices = []
    lens = 0
    for spectrum in data['spectra']:
        exp_1, exp_2 = spectrum['experiments']
        freq_1, freq_2 = spectrum['frequencies']
        # Read off polarization channel combinations
        pols = spectrum.get('polarizations',
                            default_cuts['polarizations']).copy()
        # Read off scale cuts
        scls = spectrum.get('scales',
                            default_cuts['scales']).copy()
        # For the same two channels, do not include ET and TE, only TE
        if (exp_1 == exp_2) and (freq_1 == freq_2):
            if ('ET' in pols):
                pols.remove('ET')
                if ('TE' not in pols):
                    pols.append('TE')
                    scls['TE'] = scls['ET']
        for pol in pols:
            p1, p2 = pol
            tname_1 = exp_1 + '_' + str(freq_1)
            tname_2 = exp_2 + '_' + str(freq_2)
            lmin, lmax = scls[p1 + p2]
            if p1 in ['E', 'B']:
                tname_1 += '_s2'
            else:
                tname_1 += '_s0'
            if p2 in ['E', 'B']:
                tname_2 += '_s2'
            else:
                tname_2 += '_s0'
            dtype = 'cl_' + pol_dict[p1] + pol_dict[p2]
            ind = s.indices(dtype,  # Select power spectrum type
                            (tname_1, tname_2),  # Select channel combinations
                            ell__gt=lmin, ell__lt=lmax)  # Scale cuts
            lens += len(ind)
            indices += list(ind)
            print(tname_1, tname_2, dtype, ind.shape, lmin, lmax)

    # Get rid of all the unselected power spectra
    s.keep_indices(np.array(indices))
    # Pass all of the data downstream
    data.update({'inputs': s})


def get_fg_model(setup, fg_param):

    data = setup["data"]
    lmin, lmax = 2, data["lmax"]
    l = np.arange(lmin, lmax)

    foregrounds = setup["foregrounds"]
    normalisation = foregrounds["normalisation"]
    nu_0 = normalisation["nu_0"]
    ell_0 = normalisation["ell_0"]
    T_CMB = normalisation["T_CMB"]

    from fgspectra import cross as fgc
    from fgspectra import power as fgp
    from fgspectra import frequency as fgf
    cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())

    experiments = data["experiments"]
    all_freqs = np.concatenate([data["freq_%s"%exp] for exp in experiments])
    all_freqs = all_freqs.astype(int)

    model = {}
    model["tt","kSZ"] = fg_param["a_kSZ"] * ksz({"nu": all_freqs},
                                                {"ell": l, "ell_0": ell_0})
    model["tt","cibp"] = fg_param["a_p"] * cibp({"nu": all_freqs, "nu_0": nu_0,
                                                 "temp": fg_param["T_d"], "beta": fg_param["beta_p"]},
                                                {"ell": l, "ell_0": ell_0, "alpha": 2})
    model["tt","radio"] = fg_param["a_s"] * radio({"nu": all_freqs, "nu_0": nu_0, "beta": -0.5 - 2},
                                                  {"ell": l, "ell_0": ell_0, "alpha": 2})
    model["tt","tSZ"] = fg_param["a_tSZ"] * tsz({"nu": all_freqs, "nu_0": nu_0},
                                                {"ell": l, "ell_0": ell_0})
    model["tt","cibc"] = fg_param["a_c"] * cibc({"nu": all_freqs, "nu_0": nu_0,
                                                 "temp": fg_param["T_d"], "beta": fg_param["beta_c"]},
                                                {"ell": l, "ell_0": ell_0, "alpha": 2 - fg_param["n_CIBC"]})

    spectra = ["tt","te","ee"]
    components = foregrounds["components"]
    component_list = {k:components[k] for k in spectra}

    fg_model = {}
    for c1, f1 in enumerate(all_freqs):
        for c2, f2 in enumerate(all_freqs):
            for s in spectra:
                fg_model[s, "all", f1, f2] = np.zeros(len(l))
                for comp in component_list[s]:
                    fg_model[s, comp, f1, f2] = model[s, comp][c1, c2]
                    fg_model[s, "all", f1, f2] += fg_model[s, comp, f1, f2]

    return fg_model

def sampling(setup):

    data = setup["data"]
    lmin, lmax = 2, data["lmax"]
    select = data["select"]
    spec_list= data["spec_list"]

    data_vec, inv_cov, Bbl = data["data_vec"], data["inv_cov"], data["Bbl"]

    def chi2(a_tSZ, a_kSZ, a_p, beta_p, a_c, beta_c, n_CIBC, a_s, T_d,
             _theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):

        # Get nuisance parameters
        fg_params = setup["simulation"]["fg_parameters"]
        from inspect import getfullargspec
        for p in getfullargspec(chi2).args:
            if p in ["_derived", "_theory"]: continue
            fg_params.update({p: locals()[p]})
        fg_model= get_fg_model(setup, fg_params)

        # Get theoritical Cl
        Dls_theo = _theory.get_Cl(ell_factor=True)
        spectra = ["tt", "te", "ee"]
        for s in spectra:
            Dls_theo[s] = Dls_theo[s][lmin:lmax]

        th_vec=[]
        if select == "tt-te-ee":
            for s in spectra:
                for spec in spec_list:
                    m1, m2 = spec.split("x")
                    f1, f2 = int(m1.split("_")[1]), int(m2.split("_")[1])
                    th_vec = np.append(th_vec, np.dot(Bbl[s,spec], Dls_theo[s]+fg_model[s,"all",f1,f2]))
        else:
            for spec in spec_list:
                m1, m2 = spec.split("x")
                f1, f2 = int(m1.split("_")[1]), int(m2.split("_")[1])
                th_vec = np.append(th_vec, np.dot(Bbl[select,spec], Dls_theo[select]+fg_model[select,"all",f1,f2]))

        delta = data_vec-th_vec
        chi2_value = np.dot(delta, inv_cov.dot(delta))
        return -0.5*chi2_value

    info = setup["cobaya"]
    info["likelihood"] = {"chi2": chi2}

    from cobaya.run import run
    return run(info)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SO python likelihood")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding simulation/sampling setup",
                        default=None, required=True)
    parser.add_argument("--debug", help="Check chi2 with respect to input parameters",
                        default=False, action="store_true")
    parser.add_argument("--liketest", help="Check chi2 with respect to expected value",
                        default=False, action="store_true")
    parser.add_argument("--fisher", help="Print parameter standard deviations from Fisher matrix",
                        default=False, action="store_true")
    parser.add_argument("--do-mcmc", help="Use MCMC sampler",
                        default=False, action="store_true")
    parser.add_argument("--get-input-spectra", help="return input spectra corresponding to the sim parameters",
                        default=False, action="store_true")
    parser.add_argument("--use-fisher-covmat", help="Use covariance matrix from Fisher calculation as proposal",
                        default=False, action="store_true")
    parser.add_argument("-i","--sim-id", help="Simulation number",
                        default=None)
    args = parser.parse_args()

    import yaml
    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    if args.get_input_spectra:
        likelihood_utils.write_input_cls(setup, lmax=9000, out_dir='input_spectra')
        return

    # Prepare data
    if args.liketest:
        prepare_data(setup, '0')
        likelihood_utils.debug(setup,test=True)
        return
    
    prepare_data(setup, args.sim_id)

    if args.debug:
        likelihood_utils.debug(setup)
        return

    if args.fisher:
        params = setup.get("cobaya").get("params")
        covmat_params = [k for k, v in params.items() if isinstance(v, dict) and "prior" in v.keys() and "proposal" not in v.keys()]
        covmat = likelihood_utils.fisher(setup, covmat_params)
        return

    # Store configuration & data
    #import pickle
    #pickle.dump(setup, open(args.output_base_dir + "/setup.pkl", "wb"))

    # Do the MCMC
    if args.do_mcmc:
        # Update cobaya setup
        params = setup.get("cobaya").get("params")
        covmat_params = [k for k, v in params.items() if isinstance(v, dict)
                         and "prior" in v.keys() and "proposal" not in v.keys()]
        print("Sampling over", covmat_params, "parameters")
        if args.use_fisher_covmat:
            covmat = likelihood_utils.fisher(setup, covmat_params)
            for i, p in enumerate(covmat_params):
                params[p]["proposal"] = np.sqrt(covmat[i,i])
        else:
            for p in covmat_params:
                v = params.get(p)
                proposal = (v.get("prior").get("max") - v.get("prior").get("min"))/2
                params[p]["proposal"] = proposal
        mcmc_dict = {"mcmc": None}

        setup["cobaya"]["sampler"] = mcmc_dict
        output = setup["cobaya"]["output"] 
        updated_info, results = sampling(setup)

# script:
if __name__ == "__main__":
    main()
