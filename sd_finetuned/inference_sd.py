from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

model_base = "compvis/stable-diffusion-v1-4"
lora_model_path = "./sd_finetuned/sd-btxrd-model-lora"

pipe = DiffusionPipeline.from_pretrained(model_base, use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lora_model_path)
pipe.to("cuda")

prompt = "xray multiple osteochondromas femur"
image = pipe(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 1},
).images[0]
image.save(f"./sd_finetuned/{prompt}.png")
