import numpy as np
import pylab as plt
import os

def read_spectra(fname):
    data=np.loadtxt(fname)
    l=data[:,0]
    spectra=["tt","te","tb","et","bt","ee","eb","be","bb"]
    ps={}
    for c,f in enumerate(spectra):
        ps[f]=data[:,c+1]
    return(l,ps)

def get_cosmo_ps(setup, lmax, ell_factor=True):
    from copy import deepcopy

    # Get simulation parameters
    simu = setup["simulation"]
    camb_cosmo = deepcopy(simu["cosmo. parameters"])
    # CAMB use As
    if "logA" in camb_cosmo:
        camb_cosmo["As"] = 1e-10*np.exp(camb_cosmo["logA"])
        del camb_cosmo["logA"]

    # Get cobaya setup
    info = deepcopy(setup["cobaya"])
    info["params"] = camb_cosmo
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


def write_input_cls(setup,lmax,out_dir,plot=False):

    os.makedirs(out_dir, exist_ok=True)
    # get cosmological Dls
    Dls = get_cosmo_ps(setup, lmax, ell_factor=True)
    l = np.arange(len(Dls["tt"]))
    # get foreground Dls
    fg_model= get_fg_ps(setup,lmax)
    
    np.savetxt("%s/cosmo_spectra.dat"%out_dir, np.transpose([l[2:lmax],Dls["tt"][2:lmax],Dls["ee"][2:lmax],Dls["bb"][2:lmax],Dls["te"][2:lmax]]))

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
                    np.savetxt("%s/%s_%s_%sx%s.dat"%(out_dir,s,comp,f1,f2), np.transpose([l[2:lmax],fg_model[s,comp,f1,f2] ]))
                    if plot==True:
                        if s=="tt":
                            plt.semilogy()
                        plt.plot(l[2:lmax],fg_model[s,comp,f1,f2],label="%s %sx%s"%(comp,f1,f2))
                if plot==True:
                    plt.plot(l[2:lmax],Dls[s][2:lmax],color='gray')
                    plt.plot(l[2:lmax],fg_model[s,'all',f1,f2],color='black',label='all')
                    if s=="tt":
                        plt.ylim(10**-1,10**4)
                    plt.legend()
                    plt.show()

def debug(setup):

    data = setup["data"]
    lmin, lmax = 2, data["lmax"]
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
    
    from copy import deepcopy

    data = setup["data"]
    select = data["select"]
    spec_list = data["spec_list"]
    data_vec, inv_cov, Bbl = data["data_vec"], data["inv_cov"], data["Bbl"]
    lmin, lmax = 2, data["lmax"]
    params = covmat_params
    epsilon = 0.01
    spectra = ["tt", "te", "ee"]

    th_vec={}
    for p in params:
        setup_mod = deepcopy(setup)
        cosmo = setup_mod["simulation"]["cosmo. parameters"]
        foreground= setup_mod["simulation"]["fg_parameters"]
        if p in cosmo:
            value = cosmo[p]
    
            cosmo[p] = (1-epsilon)*value
            Cl_minus = get_cosmo_ps(setup_mod, lmax)
            cosmo[p] = (1+epsilon)*value
            Cl_plus = get_cosmo_ps(setup_mod, lmax)
            
            delta={}
            for s in spectra:
                delta[s]=(Cl_plus[s][lmin:lmax] - Cl_minus[s][lmin:lmax])/(2*epsilon*value)
     
            th_vec[p]=[]
            if select == "tt-te-ee":
                for s in spectra:
                    for spec in spec_list:
                        th_vec[p]=np.append(th_vec[p],np.dot(Bbl[s,spec], delta[s]))
            else:
                for spec in spec_list:
                    th_vec[p] = np.append(th_vec[p],np.dot(Bbl[select,spec], delta[select]))

        elif p in foreground:
            
            value = foreground[p]
            
            foreground[p] = (1+epsilon)*value
            fg_model_plus = get_fg_ps(setup_mod,lmax)
            foreground[p] = (1-epsilon)*value
            fg_model_minus = get_fg_ps(setup_mod,lmax)
            
            th_vec[p]=[]
            if select == "tt-te-ee":
                for s in spectra:
                    for spec in spec_list:
                        m1,m2=spec.split('x')
                        f1,f2=int(m1.split('_')[1]),int(m2.split('_')[1])
                        th_vec[p]=np.append(th_vec[p],np.dot(Bbl[s,spec], (fg_model_plus[s,"all",f1,f2]-fg_model_minus[s,"all",f1,f2])/(2*epsilon*value)))
            else:
                for spec in spec_list:
                    m1,m2=spec.split('x')
                    f1,f2=int(m1.split('_')[1]),int(m2.split('_')[1])
                    th_vec[p] = np.append(th_vec[p],np.dot(Bbl[select,spec], (fg_model_plus[select,"all",f1,f2]-fg_model_minus[select,"all",f1,f2])/(2*epsilon*value)))

    nparam = len(params)
    fisher = np.empty((nparam,nparam))

    for i1,p1 in enumerate(params):
        for i2,p2 in enumerate(params):
            fisher[i1,i2]=  np.dot(th_vec[p1], inv_cov.dot(th_vec[p2]))

    cosmo = setup["simulation"]["cosmo. parameters"]
    foreground= setup["simulation"]["fg_parameters"]

    cov = np.linalg.inv(fisher)
    for count, p in enumerate(params):
        if p in cosmo:
            value = cosmo[p]
        elif p in foreground:
            value=foreground[p]
        print("param:",p,", sigma:",np.sqrt(cov[count,count]),", Fisher S/N", value/np.sqrt(cov[count,count]) )

    return cov

