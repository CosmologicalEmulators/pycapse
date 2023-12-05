from juliacall import Main as jl
import juliacall as jc
import numpy as np

jl.seval("using Capse")

__capse_compute_Cl = jl.seval('Capse.get_Cℓ')
__get_lgrid = jl.seval('Capse.get_ℓgrid')
__get_emulator_description = jl.seval('Capse.get_emulator_description')
simplechainsemulator = jl.seval('Capse.SimpleChainsEmulator')
luxemulator = jl.seval('Capse.LuxEmulator')
__init_emulator = jl.seval('Capse.init_emulator')
__cl_emulator = jl.seval('Capse.CℓEmulator')
__load_emulator = jl.seval('Capse.load_emulator')

def compute_Cl(cosmo, emu):
    Cl = __capse_compute_Cl(jl.collect(cosmo), emu)
    return np.array(Cl)

#TODO #3 is this method needed?
def compute_Cl_vec(cosmo_vec, emu):
    Cl = __capse_compute_Cl(jl.collect(np.transpose(cosmo_vec)), emu)
    return np.array(Cl)

"""def load_emu(path):
    loaded = __load_emu_jl(path)
    emu = loaded["Cℓ"]
    return emu"""

def get_lgrid(emu):
    return np.array(__get_lgrid(emu))

#def get_emulator_description(emu):
#    __get_emulator_description(emu)

def get_emulator_description(emu):
    input_dict = emu.TrainedEmulator.Description
    print("The parameters the model has been trained are, in the following order: "+input_dict["parameters"]+".")
    print("The emulator has been trained by "+input_dict["author"]+".")
    print(input_dict["author"]+" email is "+input_dict["author_email"]+".")
    print(input_dict["miscellanea"])

    return None

def nested_dict_convert(mydict):
    for key, value in mydict.items():
        if isinstance(value, dict):
            nested_dict_convert(value)
            mydict[key] = jc.convert(jl.Dict, value)

    return mydict

def init_emulator(NN_dict, weights, emu_kind):
    new_nn = jc.convert(jl.Dict, nested_dict_convert(NN_dict))
    return __init_emulator(new_nn, jl.collect(weights), emu_kind)

def cl_emulator(trained_emu, lgrid, InMinMax, OutMinMax):
    return __cl_emulator(trained_emu, lgrid, InMinMax, OutMinMax)

def get_parameters_list(emu):
    return emu.TrainedEmulator.Description["parameters"].split(", ")

def load_emulator(path, emu = simplechainsemulator, weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl"):
    return __load_emulator(path, emu = emu, weights_file = weights_file, inminmax_file = inminmax_file,
    outminmax_file = outminmax_file, nn_setup_file = nn_setup_file,
    postprocessing_file = postprocessing_file)
