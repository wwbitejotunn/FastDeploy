## stable diffusion 暂用部署方案
1. 代码目录: FastDeploy/examples/multimodal/stable_diffusion

2. 安装fast deploy: FastDeploy/python
注意, 其中TRT_DIRECTORY需要替换为本机tensorrt地址
环境:cuda:11.2, cudnn:8.2, tensorrt:8.4.0.6

```
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_TEXT=ON
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export TRT_DIRECTORY=/Paddle/TensorRT-8.4.0.6
export CUDA_DIRECTORY=/usr/local/cuda
python setup.py build
python setup.py bdist_wheel
```

3. 导出模型: FastDeploy/examples/multimodal/stable_diffusion
为了兼容现版本的interpret 算子进入trt, 导出固定shape的模型

```
# 安装依赖
cd FastDeploy/examples/multimodal/stable_diffusion
pip install -r requirements_paddle.txt

python export_model_512.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5_512
```

3.1 导出1024 vae模型
```
python export_model_ldm.py --pretrained_model_name_or_path ./pretrained_paddle_model --output_path ldm_1024
```

4. 运行推理: FastDeploy/examples/multimodal/stable_diffusion

```
cd FastDeploy/examples/multimodal/stable_diffusion
# 使用paddle-tensorrt运行推理
python infer.py  --model_dir stable-diffusion-v1-5_512 \
                 --scheduler "euler_ancestral" \
                 --backend paddle-tensorrt \
                 --device_id=2 --use_fp16=true \
                 --inference_steps=50
```

4.1 运行推理, 使用1024vae模型, 支持多种bsz设置

```
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
```
可参考batch_run_ldm_1024vae.sh