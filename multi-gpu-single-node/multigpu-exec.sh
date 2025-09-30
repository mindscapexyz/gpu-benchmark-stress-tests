for i in {0..7}; do # assuming 8 gpus
  screen -dmS vllm$i bash -c "
  CUDA_VISIBLE_DEVICES=$i vllm serve microsoft/Phi-3-mini-4k-instruct \ # change this to the model you want to test
    --dtype bfloat16 \
    --api-key token \
    --host 0.0.0.0 --port $((8000 + i)) \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --kv-cache-dtype auto"
done