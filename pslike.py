"""
Adapted from https://github.com/xgarrido/beyondCV/blob/master/beyondCV/beyondCV.py
and https://github.com/xgarrido/correlation_coeff_cosmo/blob/master/corrcoeff/corrcoeff.py
"""
import numpy as np
import pylab as plt
import likelihood_utils
import yaml
import argparse
import time
from fgspectra import cross as fgc
from fgspectra import power as fgp
from fgspectra import frequency as fgf


def prepare_data(setup, sim_id=None):

    data = setup["data"]
    loc = data["data_folder"]
    experiments = data["experiments"]

    spectra = ["tt","te","ee"]
    Bbl = {s:[] for s in spectra}
    data_vec = {s:[] for s in spectra}

    spec_list=[]
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
                        Bbl[s,spec_name] = np.loadtxt("%s/Bbl_%s_%s.dat" % (loc, spec_name, s.upper()))
                        if s == "te":
                            data_vec[s] = np.append(data_vec[s], (ps["te"]+ps["et"])/2)
                        else:
                            data_vec[s] = np.append(data_vec[s], ps[s])
                
                    spec_list+=[spec_name]

    cov_mat = np.loadtxt("%s/covariance.dat" % loc)

    select = data["select"]
    if select == "tt-te-ee":
        vec = np.concatenate([data_vec[spec] for spec in spectra])
        data.update({"l": l, "data_vec": vec, "inv_cov": np.linalg.inv(cov_mat), "Bbl": Bbl, "spec_list": spec_list})
    else:
        for count, spec in enumerate(spectra):
            if select == spec:
                n_bins = int(cov_mat.shape[0])
                cov_mat = cov_mat[count*n_bins//3:(count+1)*n_bins//3,
                                  count*n_bins//3:(count+1)*n_bins//3]
                data.update({"l": l, "data_vec": data_vec[spec], "inv_cov": np.linalg.inv(cov_mat), "Bbl": Bbl,  "spec_list": spec_list})


def get_fg_model(setup,fg_param):
    
    data = setup["data"]
    l = np.arange(2,data["lmax"]+2)
    experiments = data["experiments"]

    foregrounds=setup["foregrounds"]
    normalisation = foregrounds["normalisation"]
    nu_0= normalisation["nu_0"]
    ell_0= normalisation["ell_0"]
    T_CMB= normalisation["T_CMB"]
    
    cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())
    
    all_freqs=[]
    for exp in experiments:
        all_freqs  = np.append(all_freqs, data["freq_%s"%exp])
    all_freqs=all_freqs.astype(int)

    spectra = ["tt","te","ee"]

    components= foregrounds["components"]

    component_list={}
    component_list["tt"]=components["tt"]
    component_list["te"]=components["te"]
    component_list["ee"]=components["ee"]

    model={}
    model["tt","kSZ"]=fg_param["a_kSZ"] * ksz({"nu":all_freqs},{"ell":l, "ell_0":ell_0})
    model["tt","cibp"]=fg_param["a_p"] * cibp({"nu": all_freqs, "nu_0":nu_0, "temp":fg_param["T_d"], "beta":fg_param["beta_p"]},{"ell":l, "ell_0":ell_0, "alpha":2})
    model["tt","radio"]=fg_param["a_s"] * radio({"nu": all_freqs, "nu_0":nu_0, "beta":-0.5 - 2},{"ell":l, "ell_0":ell_0, "alpha":2})
    model["tt","tSZ"]=fg_param["a_tSZ"] * tsz( {"nu":all_freqs, "nu_0": nu_0},{"ell":l, "ell_0":ell_0})
    model["tt","cibc"]=fg_param["a_c"] * cibc({"nu": all_freqs, "nu_0":nu_0, "temp":fg_param["T_d"], "beta":fg_param["beta_c"]}, {"ell":l, "ell_0":ell_0, "alpha":2 - fg_param["n_CIBC"]})

    fg_model={}
    for c1,f1 in enumerate(all_freqs):
        for c2,f2 in enumerate(all_freqs):
            for s in spectra:
                fg_model[s,"all",f1,f2]=np.zeros(len(l))
                for comp in component_list[s]:
                    fg_model[s,comp,f1,f2]=model[s,comp][c1,c2]
                    fg_model[s,"all",f1,f2]+=fg_model[s,comp,f1,f2]

    return fg_model


def sampling(setup):
    
    data = setup["data"]
    lmax = data["lmax"]
    select = data["select"]
    spec_list= data["spec_list"]
    
    fg_param= setup["simulation"]["fg_parameters"]
    fg_model= get_fg_model(setup,fg_param)

    simu = setup["simulation"]
    data_vec, inv_cov, Bbl = data["data_vec"], data["inv_cov"], data["Bbl"]

    def chi2(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Dls_theo = _theory.get_Cl(ell_factor=True)
        spectra = ["tt", "te", "ee"]
        for s in spectra:
            Dls_theo[s] = Dls_theo[s][2:lmax+2]
        
        th_vec=[]

        if select == "tt-te-ee":
            for s in spectra:
                for spec in spec_list:
                    m1,m2=spec.split('x')
                    f1,f2=int(m1.split('_')[1]),int(m2.split('_')[1])
                    th_vec=np.append(th_vec,np.dot(Bbl[s,spec], Dls_theo[s]+fg_model[s,"all",f1,f2]))
        else:
            for spec in spec_list:
                m1,m2=spec.split('x')
                f1,f2=int(m1.split('_')[1]),int(m2.split('_')[1])
                th_vec = np.append(th_vec,np.dot(Bbl[select,spec], Dls_theo[select]+fg_model[select,"all",f1,f2]))
                plt.semilogy()
                plt.plot(fg_model[select,"all",f1,f2],label='%s %s'%(f1,f2))
                    
            plt.plot(Dls_theo[select])
            plt.legend()
            plt.show()

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
    parser.add_argument("--debug", help="cCeck chi2 with respect to input parameters",
                            default=False, required=False, action="store_true")
    parser.add_argument("--do-mcmc", help="Use MCMC sampler",
                        default=False, required=False, action="store_true")
    parser.add_argument("--get-input-spectra", help="return input spectra corresponding to the sim parameters",
                            default=False, required=False, action="store_true")
    parser.add_argument("--output-base-dir", help="Set the output base dir where to store results",
                        default=".", required=False)
    parser.add_argument("--use-fisher-covmat", help="Use covariance matrix from Fisher calculation as proposal",
                        default=False, required=False, action="store_true")
    parser.add_argument("-id","--sim-id", help="Simulation number",
                        default=None, required=False)
    args = parser.parse_args()

    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    # Store configuration & data
    import pickle
    pickle.dump(setup, open(args.output_base_dir + "/setup.pkl", "wb"))

    # Do the MCMC
    if args.do_mcmc:
        simu=setup["simulation"]
        prepare_data(setup,args.sim_id)
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
        setup["cobaya"]["output"] = args.output_base_dir + "/mcmc"
        updated_info, results = sampling(setup)

    if args.get_input_spectra:
        likelihood_utils.write_simu_cls(setup, lmax=9000, out_dir=args.output_base_dir + '/sim_spectra')
    if args.debug:
        prepare_data(setup,args.sim_id)
        likelihood_utils.debug(setup)


# script:
if __name__ == "__main__":
    main()
