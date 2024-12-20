from safetensors import safe_open
import mindspore as ms 
import numpy as np 

state_dict = {}
with safe_open('/home/nthai/.cache/huggingface/hub/models--Shitao--OmniGen-v1/snapshots/58e249c7c7634423c0ba41c34a774af79aa87889/model.safetensors', framework = 'pt', device = 'cpu') as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

target_data = []
for k in state_dict:
    print(k)
    if "." not in k:
        # only for GroupNorm
        ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        # print(". ", ms_name, k)

    else:
        if "norm" in k:
            ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            # print("NORM.  ", ms_name, k)
        else:
            ms_name = k

    val = state_dict[k].detach().numpy().astype(np.float32)
    target_data.append({"name": ms_name, "data": ms.Tensor(val, dtype=ms.float32)})