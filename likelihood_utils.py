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

def get_cosmo_ps(setup, lmax, ell_factor=True):
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

def get_fg_ps(setup,lmax):
    from pslike import get_fg_model
    foregrounds=setup["foregrounds"]
    setup["data"]["lmax"]= lmax
    fg_param=setup["simulation"]["fg_parameters"]
    fg_model=get_fg_model(setup,fg_param)
    return fg_model


def write_simu_cls(setup,lmax,out_dir,lmin=2):

    os.makedirs(out_dir, exist_ok=True)
    # get cosmological Dls
    Dls = get_cosmo_ps(setup, lmax, ell_factor=True)
    l = np.arange(len(Dls["tt"]))
    # get foreground Dls
    fg_model= get_fg_ps(setup,lmax)
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


def debug(setup):

    data = setup["data"]
    lmin, lmax = data["lmin"], data["lmax"]
    select = data["select"]
    spec_list = data["spec_list"]

    data_vec, inv_cov, Bbl = data["data_vec"], data["inv_cov"], data["Bbl"]

    fg_param = setup["simulation"]["fg_parameters"]
    fg_model = get_fg_ps(setup,lmax)

    Dls_theo = get_cosmo_ps(setup, lmax, ell_factor=True)
    spectra = ["tt", "te", "ee"]
    for s in spectra:
        Dls_theo[s] = Dls_theo[s][lmin:lmax]

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


    delta = data_vec-th_vec
    chi2 = np.dot(delta, inv_cov.dot(delta))
    print ("%s chi2/dof: %.02f/%d "%(select, chi2,len(data_vec)))


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
        cosmo = setup_mod["simulation"]["cosmo. parameters"]
        value = cosmo[p]
        if p == "logA":
            value = 1e-10*np.exp(value)
            p = "As"
            cosmo["As"] = value
            del cosmo["logA"]
        cosmo[p] = (1-epsilon)*value
        Cl_minus = get_cosmo_ps(setup_mod, lmax)
        cosmo[p] = (1+epsilon)*value
        Cl_plus = get_cosmo_ps(setup_mod, lmax)

        d = {}
        for s in ["tt", "te", "ee", "r"]:
            if s == "r":
                plus = Cl_plus["te"]/np.sqrt(Cl_plus["tt"]*Cl_plus["ee"])
                minus = Cl_minus["te"]/np.sqrt(Cl_minus["tt"]*Cl_minus["ee"])
            else:
                plus, minus = Cl_plus[s], Cl_minus[s]
            delta = (plus[lmin:lmax] - minus[lmin:lmax])/(2*epsilon*value)
            d[s] = delta if p != "As" else delta*value

        if select == "tt-te-ee":
            deriv[i] = np.array([[d["tt"], d["te"]],
                                 [d["te"], d["ee"]]])
        else:
            deriv[i] = d[study.lower()]

    # Compute covariance matrix
    Cls = get_cosmo_ps(setup, lmax)
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
