"""
Adapted from https://github.com/xgarrido/beyondCV/blob/master/beyondCV/beyondCV.py
and https://github.com/xgarrido/correlation_coeff_cosmo/blob/master/corrcoeff/corrcoeff.py
"""
import numpy as np
import pylab as plt
import likelihood_utils
import yaml
import argparse

def prepare_data(setup,sim_id=None):

    data = setup["data"]
    loc = data["data_folder"]
    experiments=data["experiments"]
    select=data["select"]

    Bbl={}
    data_vec={}

    for spec in ["tt","te","ee"]:
        data_vec[spec]=[]
        Bbl[spec]=[]

    for id_exp1,exp1 in enumerate(experiments):
        freqs1=data["freq_%s"%exp1]
        for id_f1,f1 in enumerate(freqs1):
            for id_exp2,exp2 in enumerate(experiments):
                freqs2=data["freq_%s"%exp2]
                for id_f2,f2 in enumerate(freqs2):
                    if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                    if  (id_exp1>id_exp2) : continue

                    spec_name="%s_%sx%s_%s"%(exp1,f1,exp2,f2)

                    if sim_id is not None:
                        file_name="%s/Dl_%s_%05d.dat"%(loc,spec_name,int(sim_id))
                    else:
                        file_name="%s/Dl_%s.dat"%(loc,spec_name)

                    l,ps=likelihood_utils.read_spectra(file_name)

                    data_vec["tt"]=np.append(data_vec["tt"],ps["tt"])
                    data_vec["te"]=np.append(data_vec["te"],(ps["te"]+ps["et"])/2)
                    data_vec["ee"]=np.append(data_vec["ee"],ps["ee"])

                    Bbl["tt"]+=[np.loadtxt("%s/Bbl_%s_TT.dat"%(loc,spec_name))]
                    Bbl["te"]+=[np.loadtxt("%s/Bbl_%s_TE.dat"%(loc,spec_name))]
                    Bbl["ee"]+=[np.loadtxt("%s/Bbl_%s_EE.dat"%(loc,spec_name))]

    cov_mat=np.loadtxt("%s/covariance.dat"%loc)

    simu = setup["simulation"]

    for count,spec in enumerate(["tt","te","ee"]):
        if select==spec:
            n_bins=int(cov_mat.shape[0])
            cov_mat= cov_mat[count*n_bins//3:(count+1)*n_bins//3,count*n_bins//3:(count+1)*n_bins//3]
            simu.update({"l": l, "data_vec": data_vec[spec], "inv_cov": np.linalg.inv(cov_mat), "Bbl":Bbl[spec]})

    if select=="tt-te-ee":
        vec=[]
        for spec in (["tt","te","ee"]):
            vec=np.append(vec,data_vec[spec])
        simu.update({"l": l, "data_vec": vec, "inv_cov": np.linalg.inv(cov_mat), "Bbl":Bbl})


def sampling(setup):

    lmax =setup["data"]["lmax"]
    select =setup["data"]["select"]
    nspec= likelihood_utils.get_nspectra(setup["data"])

    simu = setup["simulation"]
    data_vec,inv_cov,Bbl = simu["data_vec"], simu["inv_cov"], simu["Bbl"]

    def chi2(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Dls_theo = _theory.get_Cl(ell_factor=True)
        for s in ["tt", "te", "ee"]:
            Dls_theo[s] = Dls_theo[s][:lmax]
        th_vec=[]
        for n in range(nspec):
            th_vec=np.append(th_vec,np.dot(Bbl[n],Dls_theo[select]))

        delta=data_vec-th_vec
        chi2 = np.dot(delta, inv_cov.dot(delta))
        return -0.5*chi2

    def chi2_joint(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Dls_theo = _theory.get_Cl(ell_factor=True)
        th_vec=[]
        for s in ["tt", "te", "ee"]:
            Dls_theo[s] = Dls_theo[s][:lmax]
            for n in range(nspec):
                th_vec=np.append(th_vec,np.dot(Bbl[s][n],Dls_theo[s]))
        delta=data_vec-th_vec
        chi2 = np.dot(delta, inv_cov.dot(delta))
        return -0.5*chi2

    info = setup["cobaya"]

    if select=="tt-te-ee":
        info["likelihood"] = {"chi2": chi2_joint}
    else:
        info["likelihood"] = {"chi2": chi2}


    from cobaya.run import run
    return run(info)


def main():
    parser = argparse.ArgumentParser(description="SO python likelihood")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding sim/minization setup",default=None, required=True)
    parser.add_argument("--do-mcmc", help="Use MCMC sampler", default=False, required=False, action="store_true")
    parser.add_argument("--output-base-dir", help="Set the output base dir where to store results",default=".", required=False)
    parser.add_argument("-id","--sim-id", help="Simulation number",default=None, required=False)

    args = parser.parse_args()

    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    likelihood_utils.write_theory_cls(setup,lmax=9000,out_dir='sim_spectra')

    prepare_data(setup,args.sim_id)

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
