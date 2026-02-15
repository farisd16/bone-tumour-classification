from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# TODO: Add argument parsing for model_base, lora_model_path and prompt

model_base = "sd-legacy/stable-diffusion-v1-5"
lora_model_path = "./latent_diffusion_finetuned/lora_weights/sd-1-5-btxrd-model-lora-rank-32-batch-4/checkpoint-5000"

pipe = DiffusionPipeline.from_pretrained(
    model_base, use_safetensors=True, safety_checker=None
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lora_model_path)
pipe.to("cuda")

prompt = "X-ray image of osteochondroma in the femur, lateral view"
image = pipe(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 1},
).images[0]
image.save(f"./latent_diffusion_finetuned/{prompt}.png")
