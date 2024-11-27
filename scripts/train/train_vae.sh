export WANDB_API_KEY='5f397c82b6010a3fc65ce65adcb37fe9cc3a57a0'
export HF_ENDPOINT="https://hf-mirror.com"

config="configs/train/sdxl_vae.yaml"

source /opt/conda/bin/activate hipa

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=55558 \
    scripts/train/train_vae.py -n naive_vae -b ${config} -t --distributed-backend nccl
 