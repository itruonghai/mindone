# from OmniGen.transformer import Phi3Config
from OmniGen import OmniGen
from mindnlp.transformers import Phi3Config
from OmniGen.transformer import Phi3Transformer
import mindspore as ms
import mindspore.nn as nn
from typing import Dict, Tuple, Union

def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name)
        ms_params[name] = value
    return ms_params

model_name = '/home/nthai/.cache/huggingface/hub/models--Shitao--OmniGen-v1/snapshots/58e249c7c7634423c0ba41c34a774af79aa87889'
config = Phi3Config.from_pretrained(model_name)


print(config)
model = OmniGen(config)
print(model)
ms_param = mindspore_params(model)
def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> nn.Cell:
    if isinstance(ckpt, str):
        print(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
    else:
        param_dict = ckpt

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    if not (len(param_not_load) == len(ckpt_not_load) == 0):
        print(
            "Exist ckpt params not loaded: {} (total: {}), or net params not loaded: {} (total: {})".format(
                ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
            )
        )
    return model

# ckpt = ms.load_checkpoint('/mnt/disk2/nthai/mindone/examples/OmniGen/models/omnigen.ckpt')
load_ckpt_params(model, '/mnt/disk2/nthai/mindone/examples/OmniGen/models/omnigen.ckpt')
# print(ms_param)
vae =  vae = AutoencoderKL.from_pretrained()
