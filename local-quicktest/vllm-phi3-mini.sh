CUDA_VISIBLE_DEVICES=0 vllm serve microsoft/Phi-3-mini-4k-instruct \
  --dtype bfloat16 \
  --api-key token \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --kv-cache-dtype auto # 3060 doesnt natively support fp16/fp8
