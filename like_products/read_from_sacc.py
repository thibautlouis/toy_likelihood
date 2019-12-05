import sacc
import numpy as np
import matplotlib.pyplot as plt

s = sacc.Sacc.load_fits("data_sacc_00000_ww.fits")

print(s.tracers.keys()); exit(1)

s.remove_selection(data_type='cl_0e',tracers=('93_s0','145_s2'),ell__gt=400)
l, cl, cov, bl = s.get_ell_cl('cl_0e','93_s0', '145_s2', return_cov=True, return_windows=True)
print(l)
print(cl.shape)
print(cov.shape)
print(bl[1].shape)
for b in bl[1]:
    plt.plot(bl[0],b)
plt.show()
exit(1)
xcorrs = s.get_tracer_combinations()
print(xcorrs)
print(dir(s.data[0]))
print(s.data[0].tags['window'].weight[:,s.data[0].tags['window_id']].shape)
print(s.data[1].tags)

