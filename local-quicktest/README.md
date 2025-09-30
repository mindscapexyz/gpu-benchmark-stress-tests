# Local workspace specification
- NVIDIA 3060, 12GB VRAM
- 64 GB System RAM

# LLM selected
- Qwen 2.5-7B with INT4 GPTQ quantization can fit
- Or Phi-3-Mini 3.8B


# How to execute
1. Spin the vllm server u want
phi3-mini
```
source local-quicktest/vllm-phi3-mini.sh
```

or 

```
source local-quicktest/vllm-qwen2.5-7b.sh
```

2. Update the locustfile.py in this folder to point to the correct model based on the vllm that you spin
3. Run local-quicktest/locust-exec.sh
4. Configure the number of users and ramp up in the Web UI (usually 5 users, 1 ramp up)