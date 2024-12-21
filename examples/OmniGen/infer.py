from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1", '/home/nthai/.cache/huggingface/hub/models--Shitao--OmniGen-v1/snapshots/58e249c7c7634423c0ba41c34a774af79aa87889/vae')  
prompt = "A woman holds a bouquet of flowers and faces the camera."
image = pipe(
    prompt, 
    height = 256,
    width = 256,
    guidance_scale=2.5, 
    num_inference_steps=50,
    use_kv_cache = False)
image[0].save('a.png')
# import pdb
# pdb.set_trace()