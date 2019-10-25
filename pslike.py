"""
Adapted from https://github.com/xgarrido/beyondCV/blob/master/beyondCV/beyondCV.py
and https://github.com/xgarrido/correlation_coeff_cosmo/blob/master/corrcoeff/corrcoeff.py
"""
import numpy as np
import pylab as plt
import likelihood_utils
import yaml
import argparse

def prepare_data(setup, sim_id=None):

    data = setup["data"]
    loc = data["data_folder"]
    experiments = data["experiments"]

    spectra = ["tt","te","ee"]
    Bbl = {s:[] for s in spectra}
    data_vec = {s:[] for s in spectra}

    for id_exp1, exp1 in enumerate(experiments):
        freqs1 = data["freq_%s" % exp1]
        for id_f1, f1 in enumerate(freqs1):
            for id_exp2, exp2 in enumerate(experiments):
                freqs2 = data["freq_%s" % exp2]
                for id_f2, f2 in enumerate(freqs2):
                    if id_exp1 == id_exp2 and id_f1 > id_f2: continue
                    if id_exp1 > id_exp2: continue

                    spec_name = "%s_%sx%s_%s" % (exp1, f1, exp2, f2)
                    file_name = "%s/Dl_%s" % (loc, spec_name)
                    file_name += ".dat" if not sim_id else "_%05d.dat" % int(sim_id)

                    l, ps = likelihood_utils.read_spectra(file_name)

                    for s in spectra:
                        Bbl[s] += [np.loadtxt("%s/Bbl_%s_%s.dat" % (loc, spec_name, s.upper()))]
                        if s == "te":
                            data_vec[s] = np.append(data_vec[s], (ps["te"]+ps["et"])/2)
                        else:
                            data_vec[s] = np.append(data_vec[s], ps[s])

    cov_mat = np.loadtxt("%s/covariance.dat" % loc)

    simu = setup["simulation"]
    select = data["select"]
    if select == "tt-te-ee":
        vec = np.concatenate([data_vec[spec] for spec in spectra])
        simu.update({"l": l, "data_vec": vec, "inv_cov": np.linalg.inv(cov_mat), "Bbl": Bbl})
    else:
        for count, spec in enumerate(spectra):
            if select == spec:
                n_bins = int(cov_mat.shape[0])
                cov_mat = cov_mat[count*n_bins//3:(count+1)*n_bins//3,
                                  count*n_bins//3:(count+1)*n_bins//3]
                simu.update({"l": l, "data_vec": data_vec[spec],
                             "inv_cov": np.linalg.inv(cov_mat), "Bbl": Bbl[spec]})


def sampling(setup):

    data = setup["data"]
    lmax = data["lmax"]
    select = data["select"]
    nspec = likelihood_utils.get_nspectra(data)

    simu = setup["simulation"]
    data_vec, inv_cov, Bbl = simu["data_vec"], simu["inv_cov"], simu["Bbl"]

    def chi2(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Dls_theo = _theory.get_Cl(ell_factor=True)
        spectra = ["tt", "te", "ee"]
        for s in spectra:
            Dls_theo[s] = Dls_theo[s][:lmax]

        if select == "tt-te-ee":
            th_vec = np.concatenate([np.dot(Bbl[s][n], Dls_theo[s])
                                     for s in spectra for n in range(nspec)])
        else:
            th_vec = np.concatenate([np.dot(Bbl[n], Dls_theo[select])
                                     for n in range(nspec)])

        delta = data_vec-th_vec
        chi2 = np.dot(delta, inv_cov.dot(delta))
        return -0.5*chi2

    info = setup["cobaya"]
    info["likelihood"] = {"chi2": chi2}

    from cobaya.run import run
    return run(info)


def main():
    parser = argparse.ArgumentParser(description="SO python likelihood")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding sim/minization setup",
                        default=None, required=True)
    parser.add_argument("--do-mcmc", help="Use MCMC sampler",
                        default=False, required=False, action="store_true")
    parser.add_argument("--output-base-dir", help="Set the output base dir where to store results",
                        default=".", required=False)
    parser.add_argument("-id","--sim-id", help="Simulation number",
                        default=None, required=False)
    args = parser.parse_args()

    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    likelihood_utils.write_theory_cls(setup, lmax=9000, out_dir=args.output_base_dir + '/sim_spectra')

    prepare_data(setup,args.sim_id)

    # Store configuration & data
    import pickle
    pickle.dump(setup, open(args.output_base_dir + "/setup.pkl", "wb"))

    # Do the MCMC
    if args.do_mcmc:
        # Update cobaya setup
        params = setup.get("cobaya").get("params")
        covmat_params = [k for k, v in params.items() if isinstance(v, dict) and "prior" in v.keys()]
        print("Sampling over", covmat_params, "parameters")
        for p in covmat_params:
            v = params.get(p)
            proposal = (v.get("prior").get("max") - v.get("prior").get("min"))/2
            params[p]["proposal"] = proposal
        mcmc_dict = {"mcmc": None}

        setup["cobaya"]["sampler"] = mcmc_dict
        setup["cobaya"]["output"] = args.output_base_dir + "/mcmc"
        updated_info, results = sampling(setup)

# script:
if __name__ == "__main__":
    main()
