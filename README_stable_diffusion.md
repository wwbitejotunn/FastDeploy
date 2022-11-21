## stable diffusion 暂用部署方案
1. 代码目录: FastDeploy/examples/multimodal/stable_diffusion

2. 安装fast deploy: FastDeploy/python
```
export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_TEXT=ON
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export TRT_DIRECTORY=/Paddle/TensorRT-8.4.1.5
export CUDA_DIRECTORY=/usr/local/cuda
python setup.py build
python setup.py bdist_wheel
```

3. 导出模型: FastDeploy/examples/multimodal/stable_diffusion
为了兼容现版本的interpret 算子进入trt, 导出固定shape的模型
```
python export_model_512.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5_512
```

3. 运行推理: FastDeploy/examples/multimodal/stable_diffusion
```
python infer.py  --model_dir stable-diffusion-v1-5_512 \
                 --scheduler "euler_ancestral" \
                 --backend paddle-tensorrt \
                 --device_id=2 --use_fp16=true \
                 --inference_steps=50
```