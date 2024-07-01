config="/mnt/data/longtaozheng/talking-head/configs/train/naive_vae.yaml"

source /opt/conda/bin/activate hipa

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    --node_rank=0 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=55558 \
    scripts/train/train_vae.py -n naive_vae -b ${config} -t
