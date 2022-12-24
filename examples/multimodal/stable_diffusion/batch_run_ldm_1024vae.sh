# export model
# python export_model_1024vqvae.py --output_path stable_diffusion_1024vae

# run

# use_backend=paddle
use_backend=paddle-tensorrt

use_device=3
echo "====== 1024x1024"
for batch in $(seq 1 10); do
echo "== batch ${batch} =="
rm -rf ./stable_diffusion_1024vae/*/shape*
rm -rf ./stable_diffusion_1024vae/*/_opt*
FLAGS_use_cuda_managed_memory=true \
python infer_ldm-1024vae.py  \
--model_dir ./stable_diffusion_1024vae/ \
--scheduler "euler_ancestral" \
--backend ${use_backend} \
--device_id=${use_device} \
--use_fp16=true \
--inference_steps=50 \
--batch_size=${batch} \
--benchmark_steps=10 \
--height=1024 --width=1024
done
echo "====== 1024x1536"
for batch in $(seq 1 10); do
echo "== batch ${batch} =="
rm -rf ./stable_diffusion_1024vae/*/shape*
rm -rf ./stable_diffusion_1024vae/*/_opt*
FLAGS_use_cuda_managed_memory=true \
python infer_ldm-1024vae.py  \
--model_dir ./stable_diffusion_1024vae/ \
--scheduler "euler_ancestral" \
--backend ${use_backend} \
--device_id=${use_device} \
--use_fp16=true \
--inference_steps=50 \
--batch_size=${batch} \
--benchmark_steps=10 \
--height=1024 --width=1536
done