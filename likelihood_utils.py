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

def get_nspectra(data):
    nfreq=0
    for exp in data["experiments"]:
        nfreq+=len(data["freq_%s"%exp])
    nspec=int(nfreq*(nfreq+1)/2)
    return nspec

def get_theory_cls(setup, lmax, ell_factor=True):
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
    Cls = model.likelihood.theory.get_Cl(ell_factor=ell_factor)
    return Cls

def write_theory_cls(setup,lmax,out_dir):
    try:
        os.makedirs(out_dir)
    except:
        pass
    Cls=get_theory_cls(setup, lmax, ell_factor=True)
    l=np.arange(len(Cls['tt']))
    np.savetxt('%s/input_spectra.dat'%out_dir, np.transpose([l[2:],Cls['tt'][2:],Cls['ee'][2:],Cls['bb'][2:],Cls['te'][2:]]))
                                                             


