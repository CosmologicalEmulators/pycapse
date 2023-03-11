from juliacall import Main as jl
import numpy as np

jl.seval("using Capse")
jl.seval("using AbstractEmulator")
jl.seval("using SimpleChains")
jl.seval("using BSON")
jl.seval("using Static")

__capse_compute_Cl = jl.seval('Capse.get_Cℓ')
__load_emu_jl = jl.seval('BSON.load')
__get_lgrid = jl.seval('Capse.get_ℓgrid')

def compute_Cl(cosmo, emu):
    Cl = __capse_compute_Cl(jl.collect(cosmo), emu)
    return np.array(Cl)

def compute_Cl_vec(cosmo_vec, emu):
    Cl = __capse_compute_Cl(jl.collect(np.transpose(cosmo_vec)), emu)
    return np.array(Cl)

def load_emu(path):
    loaded = __load_emu_jl(path)
    emu = loaded["Cℓ"]
    return emu

def get_lgrid(emu):
    return np.array(__get_lgrid(emu))
