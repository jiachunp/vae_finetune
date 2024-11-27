      
#!/bin/bash

__conda_setup="$('/opt/conda/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate hipa 
pip install pytorch-lightning==2.0.8
# pip install -e facenet_pytorch
pip install mediapipe
pip install wandb
pip install natsort

RUN_DIR="/mnt/data/jiachunpan/talking-pose-vae"
export NCCL_IB_TIMEOUT=3600 
export NCCL_IB_RETRY_CNT=3600 
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=LOC
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=3600
export NCCL_IB_RETRY_CNT=3600 


export PYTHONPATH="./":$PYTHONPATH
export MLP_WORKER_GPU=8
export MLP_WORKER_NUM=$WORLD_SIZE
export MLP_ROLE_INDEX=$RANK
export MLP_WORKER_0_HOST=$MASTER_ADDR
export MLP_WORKER_0_PORT=$MASTER_PORT 


# Check and set MLP_WORKER_GPU
# if [ -z "$MLP_WORKER_GPU" ]; then
#     export MLP_WORKER_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
# fi


echo "RUN_DIR=$RUN_DIR"
echo "MLP_WORKER_GPU=$MLP_WORKER_GPU"
echo "MLP_WORKER_NUM=$MLP_WORKER_NUM"
echo "MLP_ROLE_INDEX=$MLP_ROLE_INDEX"
echo "MLP_WORKER_0_HOST=$MLP_WORKER_0_HOST"
echo "MLP_WORKER_0_PORT=$MLP_WORKER_0_PORT"  


save_name="_train_sd_vae_pose_scale_up"

export WANDB_API_KEY='56542920d6680a5c5b2f3912207c6e377561f1d0'
export HF_ENDPOINT="https://hf-mirror.com"
config="/mnt/data/jiachunpan/talking-pose-vae/configs/train/sd_vae_pose.yaml" 
   
#resumedir="/mnt/data/yifanzhang/talking-head/logs/2024-07-19T00-19-21_train-sd_vae_video_dynamic_multi/"
torchrun --nproc_per_node=${MLP_WORKER_GPU} \
    --node_rank=${MLP_ROLE_INDEX} \
    --nnodes=${MLP_WORKER_NUM} \
    --master_addr=${MLP_WORKER_0_HOST} \
    --master_port=${MLP_WORKER_0_PORT} \
    train_vae.py  -b ${config} -t  --distributed-backend nccl > $RUN_DIR$save_name.log 2>&1  

#-r ${resumedir}  

 