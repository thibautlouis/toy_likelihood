import numpy as np
import pylab as plt
import os

def read_spectra(fname):
    data=np.loadtxt(fname)
    l=data[:,0]
    spectra=['tt','te','tb','et','bt','ee','eb','be','bb']
    ps={}
    for c,f in enumerate(spectra):
        ps[f]=data[:,c+1]
    return(l,ps)

def get_cosmo_Dls(setup, lmax, ell_factor=True):
    # Get simulation parameters
    simu = setup["simulation"]
    cosmo = simu["cosmo. parameters"]
    # CAMB use As
    if "logA" in cosmo:
        cosmo["As"] = 1e-10*np.exp(cosmo["logA"])
        del cosmo["logA"]

    # Get cobaya setup
    from copy import deepcopy
    info = deepcopy(setup["cobaya"])
    info["params"] = cosmo
    # Fake likelihood so far
    info["likelihood"] = {"one": None}
    from cobaya.model import get_model
    model = get_model(info)
    model.likelihood.theory.needs(Cl={"tt": lmax, "ee": lmax, "te": lmax, "bb":lmax})
    model.logposterior({}) # parameters are fixed
    Dls = model.likelihood.theory.get_Cl(ell_factor=ell_factor)
    return Dls

def get_fg_Dls(setup,lmax):
    from pslike import get_fg_model
    foregrounds=setup["foregrounds"]
    setup["data"]["lmax"]= lmax
    fg_param=setup["simulation"]["fg_parameters"]
    fg_model=get_fg_model(setup,fg_param)
    return fg_model


def write_simu_cls(setup,lmax,out_dir,lmin=2):
    
    os.makedirs(out_dir, exist_ok=True)
    # get cosmological Dls
    Dls = get_cosmo_Dls(setup, lmax, ell_factor=True)
    l = np.arange(len(Dls["tt"]))
    # get foreground Dls
    fg_model= get_fg_Dls(setup,lmax)
    np.savetxt("%s/cosmo_spectra.dat"%out_dir, np.transpose([l[lmin:lmax],Dls["tt"][lmin:lmax],Dls["ee"][lmin:lmax],Dls["bb"][lmin:lmax],Dls["te"][lmin:lmax]]))

    data = setup["data"]
    experiments = data["experiments"]

    all_freqs=[]
    for exp in experiments:
        all_freqs  = np.append(all_freqs, data["freq_%s"%exp])
    all_freqs=all_freqs.astype(int)

    foregrounds= setup["foregrounds"]
    spectra = ["tt","te","ee"]
    components= foregrounds["components"]

    component_list={}
    component_list["tt"]=components["tt"]
    component_list["te"]=components["te"]
    component_list["ee"]=components["ee"]

    for c1,f1 in enumerate(all_freqs):
        for c2,f2 in enumerate(all_freqs):
            for s in spectra:
                for comp in component_list[s]:
                    np.savetxt("%s/%s_%s_%sx%s.dat"%(out_dir,s,comp,f1,f2), np.transpose([l[lmin:lmax],fg_model[s,"all",f1,f2][lmin:lmax] ]))


