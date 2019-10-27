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
    os.makedirs(out_dir, exist_ok=True)
    Cls = get_theory_cls(setup, lmax, ell_factor=True)
    l = np.arange(len(Cls['tt']))
    np.savetxt('%s/input_spectra.dat'%out_dir, np.transpose([l[2:],Cls['tt'][2:],Cls['ee'][2:],Cls['bb'][2:],Cls['te'][2:]]))

def fisher(setup, covmat_params):
    data = setup["data"]
    lmin, lmax = data["lmin"], data["lmax"]
    select = data["select"]
    ls = np.arange(lmin, lmax)
    nell = np.alen(ls)

    from copy import deepcopy

    params = covmat_params
    epsilon = 0.01
    if select == "tt-te-ee":
        deriv = np.empty((len(params), 2, 2, nell))
    else:
        deriv = np.empty((len(params), nell))

    for i, p in enumerate(params):
        setup_mod = deepcopy(setup)
        parname = p if p != "logA" else "As"
        value = setup["simulation"]["cosmo. parameters"][parname]
        setup_mod["simulation"]["cosmo. parameters"][parname] = (1-epsilon)*value
        Cl_minus = get_theory_cls(setup_mod, lmax)
        setup_mod["simulation"]["cosmo. parameters"][parname] = (1+epsilon)*value
        Cl_plus = get_theory_cls(setup_mod, lmax)

        d = {}
        for s in ["tt", "te", "ee", "r"]:
            if s == "r":
                plus = Cl_plus["te"]/np.sqrt(Cl_plus["tt"]*Cl_plus["ee"])
                minus = Cl_minus["te"]/np.sqrt(Cl_minus["tt"]*Cl_minus["ee"])
            else:
                plus, minus = Cl_plus[s], Cl_minus[s]
            delta = (plus[lmin:lmax] - minus[lmin:lmax])/(2*epsilon*value)
            d[s] = delta if p != "logA" else delta*value

        if select == "tt-te-ee":
            deriv[i] = np.array([[d["tt"], d["te"]],
                                 [d["te"], d["ee"]]])
        else:
            deriv[i] = d[study.lower()]

    # Compute covariance matrix
    Cls = get_theory_cls(setup, lmax)
    Cl_TT = Cls["tt"][lmin:lmax]
    Cl_TE = Cls["te"][lmin:lmax]
    Cl_EE = Cls["ee"][lmin:lmax]
    N_TT, N_EE = 0, 0
    if select == "tt-te-ee":
        C = np.array([[Cl_TT + N_TT, Cl_TE],
                      [Cl_TE, Cl_EE + N_EE]])
    elif select.lower() == "tt":
        C = 2*(Cl_TT + N_TT)**2
    elif select.lower() == "te":
        C = (Cl_TT + N_TT)*(Cl_EE + N_EE) + Cl_TE**2
    elif select.lower() == "ee":
        C = 2*(Cl_EE + N_EE)**2
    elif select.lower() == "r":
        R = Cl_TE/np.sqrt(Cl_TT*Cl_EE)
        C = R**4 - 2*R**2 + 1 + N_TT/Cl_TT + N_EE/Cl_EE + (N_TT*N_EE)/(Cl_TT*Cl_EE) \
            + R**2*(0.5*(N_TT/Cl_TT - 1)**2 + 0.5*(N_EE/Cl_EE - 1)**2 - 1)

    inv_C = C**-1
    if select == "tt-te-ee":
        for l in range(nell):
            inv_C[:,:,l] = np.linalg.inv(C[:,:,l])

    # Fisher matrix
    nparam = len(params)
    fisher = np.empty((nparam,nparam))
    for p1 in range(nparam):
        for p2 in range(nparam):
            somme = 0.0
            if select == "tt-te-ee":
                for l in range(nell):
                    m1 = np.dot(inv_C[:,:,l], deriv[p1,:,:,l])
                    m2 = np.dot(inv_C[:,:,l], deriv[p2,:,:,l])
                    somme += (2*ls[l]+1)/2*np.trace(np.dot(m1, m2))
            else:
                somme = np.sum((2*ls+1)*inv_C*deriv[p1]*deriv[p2])
            fisher[p1, p2] = somme

    cov = np.linalg.inv(fisher)
    print("eigenvalues = ", np.linalg.eigvals(cov))
    for count, p in enumerate(params):
        if p == "logA":
            value = np.log(1e10*setup_mod["simulation"]["cosmo. parameters"]["As"])
        else:
            value = setup_mod["simulation"]["cosmo. parameters"][p]
        print(p, value, np.sqrt(cov[count,count]))
    return cov
