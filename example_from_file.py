import numpy as np
import scipy.special
from dyson import BlockLanczosSymmSE
from pyscf import gto, scf, dft, gw, lib
from moment_gw import AGW

import sys
print(f"Attempting load of molecule from {sys.argv[1]}.")
mol = gto.M(
        atom=gto.mole.fromfile(sys.argv[1]),
        basis="cc-pvdz",
        verbose=5,
        max_memory=195000, # units MB
)

# MeanField / HartreeFock calculation
mf = dft.RKS(mol)
mf.xc = "hf"
mf.kernel()

#  exact GW; too expensive for this
#exact = gw.GW(mf, freq_int="exact")
#exact.kernel()
#print(np.max(exact.mo_energy[mf.mo_occ > 0]))

mf = mf.density_fit()

#for n in range(3):
gw = AGW(mf)
gw.diag_sigma = True 
conv, gf, se = gw.kernel(nmom=5, vhf_df=True)
gf.remove_uncoupled(tol=1e-8)
print(gf.get_occupied().energy)

