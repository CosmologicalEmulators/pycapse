from juliacall import Main as jl
import numpy as np

jl.seval("using Capse")
jl.seval("using SimpleChains")
jl.seval("using BSON")
jl.seval("using Static")

__capse_compute_Cl = jl.seval('Capse.get_Cℓ')
__load_emu_jl = jl.seval('BSON.load')

def compute_Cl(cosmo, emu):
    Xil = __capse_compute_Cl(jl.collect(cosmo), emu)
    return np.array(Xil)

def load_emu(path):
    loaded = __load_emu_jl(path)
    emu = loaded["Cℓ"]
    return emu

def get_lgrid(emu):
    return np.array(emu.ℓgrid)
