# export model
# python export_model_1024vqvae.py --output_path stable_diffusion_1024vae

# run
FLAGS_use_cuda_managed_memory=true \
python infer_ldm-1024vae.py  \
--model_dir ./stable_diffusion_1024vae/ \
--scheduler "euler_ancestral" \
--backend paddle-tensorrt \
--device_id=3 \
--use_fp16=true \
--inference_steps=50 \
--batch_size=2 \
--benchmark_steps=10 \
--height=1024 --width=1536