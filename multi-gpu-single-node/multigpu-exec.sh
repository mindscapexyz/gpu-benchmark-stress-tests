for i in {0..2}; do
    screen -S "run_$i" -X quit 2>/dev/null
    screen -dmS "run_$i" bash -c "cd \"$(pwd)\" && \
    LD_LIBRARY_PATH=\"$(pwd)/emilia/lib/python3.10/site-packages/nvidia/cudnn/lib\" \
    CUDA_VISIBLE_DEVICES=$i \
    ./emilia/bin/python3 main.py \
    --batch_size 4 \
    --compute_type bfloat16 \
    --whisper_arch large-v3 \
    --global-size 3 --local-index $i"
done