from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
prompt = "A woman holds a bouquet of flowers and faces the camera"
image = pipe(prompt, guidance_scale=2.5, num_inference_steps=50)