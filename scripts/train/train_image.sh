source /opt/conda/bin/activate talking_head

export WANDB_API_KEY='ed5069227da5d2bfc22ddd654a7f3a2b87475c1f'

torchrun --nnodes=1 --nproc_per_node=8 scripts/finetune.py --wandb --config configs/finetune/naive_finetune.yaml

python scripts/inference.py --source_image examples/reference_images/5.jpg --driving_audio examples/driving_audios/4.wav

python scripts/inference_no_ref_load_weight.py --source_image examples/reference_images/5.jpg --driving_audio examples/driving_audios/4.wav


python scripts/ref_inference.py --source_image examples/reference_images/5.jpg --driving_audio examples/driving_audios/4.wav
