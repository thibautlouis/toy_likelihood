import numpy as np
import matplotlib.pyplot as plt
import sacc

isim = 0
sim_suffix = "%05d"%isim
freqs=['93', '145', '225']
pols=['T', 'E', 'B']
map_types={'T':'0','E':'e','B':'b'}

###############################################
#
# All of this stuff is me reading in the data.
# No SACC at all yet.
#
# Read input covariance.
# Input covariance order is:
cov_order=['93_93_TT', '93_145_TT', '93_225_TT',
           '145_145_TT', '145_225_TT', '225_225_TT',
           '93_93_TE', '93_145_TE', '93_225_TE',
           '145_145_TE', '145_225_TE', '225_225_TE',
           '93_93_EE', '93_145_EE', '93_225_EE',
           '145_145_EE', '145_225_EE', '225_225_EE']
id_cov_order = {n:i for i,n in enumerate(cov_order)}
len_x_cov = len(cov_order)
covar_in = np.loadtxt("covariance.dat")
n_ells = len(covar_in) // len_x_cov
covar_in = covar_in.reshape([len_x_cov, n_ells, len_x_cov, n_ells])

# This function returns the covariance block corresponding to
# (f1a_p1a,f1b_p1b) x (f2a_p2a,f2b_p2b)
# where f and p are frequency and polarization channels.
def get_ell_covar(f1a, p1a, f1b, p1b, f2a, p2a, f2b, p2b):
    fx_1 = f1a+"_"+f1b
    fx_2 = f2a+"_"+f2b
    px_1 = p1a+p1b

    # We fake the covariance for ET
    if px_1 == 'ET':
        px_1 = 'TE'
    px_2 = p2a+p2b
    if px_2 == 'ET':
        px_2 = 'TE'

    # We fake the covariance for these combinations
    if px_1 in ['TB','EB', 'BT', 'BE', 'BB']:
        if px_2 == px_1:
            return np.identity(n_ells)
        else:
            return np.zeros([n_ells, n_ells])
    elif px_2 in ['TB','EB', 'BT', 'BE', 'BB']:
        return np.zeros([n_ells, n_ells])

    print(fx_1,px_1,fx_2,px_2)
    i1 = id_cov_order[fx_1+'_'+px_1]
    i2 = id_cov_order[fx_2+'_'+px_2]
    return covar_in[i1][ :, i2,:]

# This function returns the bandpower window
# for (fa_pa,fb_pb)
def get_bbl(fa, pa, fb, pb):
    fname_prefix = 'Bbl_LAT_'+fa+'xLAT_'+fb+'_'
    px = pa+pb
    if px in ['EE', 'EB', 'BE', 'BB']:
        px = 'EE'
    elif px in ['TE','TB', 'ET', 'BT']:
        px = 'TE'
    fname = fname_prefix+px+'.dat'

    return np.loadtxt(fname)

# Read the power spectra into dictionary
data = {}
for ifa, fa in enumerate(freqs):
    data[fa]={}
    for fb in freqs[ifa:]:
        fname = "Dl_LAT_%sxLAT_%s_%s.dat"%(fa,fb,sim_suffix)
        d = np.loadtxt(fname, unpack=True)
        dc = {'ls': d[0],
              'TT': d[1],
              'TE': d[2],
              'TB': d[3],
              'ET': d[4],
              'BT': d[5],
              'EE': d[6],
              'EB': d[7],
              'BE': d[8],
              'BB': d[9]}
        data[fa][fb]=dc

# Read bandpasses
data_bandpasses={}
data_beams={}
for f in freqs:
    n,bn = np.loadtxt("BP_LAT_"+f+".txt", unpack=True)
    data_bandpasses[f]={'nu':n, 'b_nu':bn}
    data_beams[f]={'l':np.arange(10000), 'bl':np.ones(10000)}


###############################################
#
# Alright, the fun starts here.

# Create SACC file
s = sacc.Sacc()

# Create SACC tracers
# A "tracer" is effectively one map of the sky.
# The tracers relevant for this are of the `NuMap`
# type, which are defined by a spin, a frequency
# bandpass and a beam.
# Note that we create 2 tracers for each frequency,
# one for intensity (spin-0) and one for
# polarization (spin-2).
# I think this makes sense, since bandpasses and
# beams could be different in I and P.
for f in freqs:
    # Spin-0 tracers
    s.add_tracer('NuMap', 'LAT_'+f+'_s0', 0,
                 nu=data_bandpasses[f]['nu'],
                 bpss_nu=data_bandpasses[f]['nu'],
                 ell=data_beams[f]['l'],
                 beam_ell=data_beams[f]['bl'])
    s.add_tracer('NuMap', 'LAT_'+f+'_s2', 2,
                 nu=data_bandpasses[f]['nu'],
                 bpss_nu=data_bandpasses[f]['nu'],
                 ell=data_beams[f]['l'],
                 beam_ell=data_beams[f]['bl'])

# This function allows you to iterate over
# all possible cross-power spectra.
# The order in which we store things will be
# T93,E93,B93,T145,E145,B145,T225,E225,B225
# This is just a choice, there is no requirement
# in SACC on which order should be used.
# **However**, you must enter the data in the same
# order in which you will then pass the covariance
# matrix.
def get_x_iterator():
    for ifa, fa in enumerate(freqs):
        for fb in freqs[ifa:]:
            for ipa, pa in enumerate(pols):
                if fa == fb:
                    polsb = pols[ipa:]
                else:
                    polsb = pols
                for pb in polsb:
                    yield (fa,fb,pa,pb)
    
# Create data vector
# We do this by iterating over all possible
# cross-correlations and adding each of them
# using the `Sacc.add_ell_cl` method.
for i_x,(fa,fb,pa,pb) in enumerate(get_x_iterator()):
    if pa=='T':
        ta_name = 'LAT_' + fa + '_s0'
    else:
        ta_name = 'LAT_' + fa + '_s2'

    if pb=='T':
        tb_name = 'LAT_' + fb + '_s0'
    else:
        tb_name = 'LAT_' + fb + '_s2'

    # Power spectrum types are
    # - cl_00 -> TT
    # - cl_0e -> TE
    # - cl_0b -> TB
    # etc.
    # We use 0 instead of T because we don't
    # want to have to define an increasing
    # number of types for e.g. tSZ, kSZ, delta_g
    # etc. The philosophy is that the
    # power spectrum/correlation function types
    # only care about the spins of the quantities
    # being correlated, and that the physical
    # meaning of those quantities should be specified
    # at the level of the tracers.
    cl_type = 'cl_'+map_types[pa]+map_types[pb]
    ls = data[fa][fb]['ls']
    cls=data[fa][fb][pa+pb]

    # Window functions. For each power spectrum we
    # encode them all into a `sacc.Window` object.
    bbl=get_bbl(fa,pa,fb,pb)
    ls_w = np.arange(bbl.shape[-1])
    wins = sacc.Window(ls_w, bbl.T)

    # Add the power spectrum with its corresponding
    # window function.
    s.add_ell_cl(cl_type, ta_name, tb_name,
                 ls, cls, window=wins,
                 window_id=range(len(ls)))

    print(i_x,fa,fb,pa,pb,cl_type,ta_name,tb_name)


# Count data
nmaps = len(freqs) * len(pols)
n_x = (nmaps * (nmaps + 1)) // 2
n_cls = n_ells * n_x

# Create covariance matrix and add it to the Sacc object.
# Note that we add the full covariance in one go. Because
# of this, the ordering of the covariance should be the same
# as the order in which we have entered the power spectra.
cov_full = np.zeros([n_x, n_x, n_ells, n_ells])
for ix1,(f1a,f1b,p1a,p1b) in enumerate(get_x_iterator()):
    for ix2,(f2a,f2b,p2a,p2b) in enumerate(get_x_iterator()):
        cv = get_ell_covar(f1a, p1a, f1b, p1b, f2a, p2a, f2b, p2b)
        cov_full[ix1,ix2,:,:]=cv
        print(f1a,p1a,f1b,p1b,f2a,p2a,f2b,p2b,cv.shape)
cov_full=np.transpose(cov_full,axes=[0,2,1,3]).reshape([n_cls, n_cls])
s.add_covariance(cov_full)

# Done! Save file.
s.save_fits("data_sacc_"+sim_suffix+".fits", overwrite=True)
