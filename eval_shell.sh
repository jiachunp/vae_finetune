
# save_eval_file=vae_eval_out/sd_sora_val_fvd_fid_crop_is_inflate_2d_weight_ori_real.log 
# num_frames=8
# sample_frame=32
# is_vis=True
# process_type=ori
# output_folder=vae_eval_out/sd_sora_val_fvd_fid_crop_is_inflate_2d_weight_ori
# config_file=/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml
# ckpt_path=/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors 
# input_path=/mnt/workspace/ai-story/zhuoqun.luo/dataset/sora/Videos
# CUDA_VISIBLE_DEVICES=0,1 python eval_video_vae_pipline.py --process_type ${process_type} --output_folder ${output_folder} --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path}  --input_path ${input_path}  > svd_sora_new_ori.log  2>&1
#--is_vis ${is_vis} 

# save_eval_file=vae_eval_out/svd_k600.log 
# num_frames=8
# sample_frame=32 
# process_type=ori 
# config_file=/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml
# ckpt_path=/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors 
# input_path=/mnt/workspace/ai-story/zhuoqun.luo/dataset/mini_vae_test/k600_sample_500 
# CUDA_VISIBLE_DEVICES=0,1 python eval_video_vae_pipline.py --process_type ${process_type} --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path}  --input_path ${input_path}  > svd_k600.log  2>&1 

 
# save_eval_file=vae_eval_out/svd_k600_ori.log 
# num_frames=8
# sample_frame=32 
# process_type=ori 
# config_file=/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml
# ckpt_path=/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors 
# input_path=/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/scripts/util/k600_sample_500_bucket/k600_sample_500_bucket_less.txt
# CUDA_VISIBLE_DEVICES=0 python eval_video_vae_pipline.py --process_type ${process_type}   --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path}  --input_path ${input_path}  > svd_k600_ori.log  2>&1  
  
  


# save_eval_file=vae_eval_out/svd_k600_r224.log 
# num_frames=8
# sample_frame=32 
# process_type=resize
# #output_folder=vae_eval_out/svd_sora_512
# config_file=/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml
# ckpt_path=/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors 
# input_path=/mnt/workspace/ai-story/zhuoqun.luo/dataset/mini_vae_test/k600_sample_500 
# CUDA_VISIBLE_DEVICES=0,1 python eval_video_vae_pipline.py --process_type ${process_type} --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path}  --input_path ${input_path}  > svd_k600_r224.log  2>&1 

# save_eval_file=vae_eval_out/svd_k600_resize_512.log 
# num_frames=8
# sample_frame=32 
# process_type=short_resize_512 
# config_file=/mnt/workspace/ai-story/zhuoqun.luo/workspace/debug/ti2v/configs/inference/svd_vae_infer.yaml
# ckpt_path=/mnt/data/ai-story/zhuoqun.luo/model_zoo/svd.safetensors 
# input_path=/mnt/workspace/ai-story/zhuoqun.luo/dataset/mini_vae_test/k600_sample_500 
# CUDA_VISIBLE_DEVICES=0,1 python eval_video_vae_pipline.py --process_type ${process_type} --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path}  --input_path ${input_path}  > svd_k600_resize_512.log  2>&1

 

# is_vis=True
# save_eval_file=vae_eval_out/sdxl_tuning_s50W.log 
# output_folder=vae_eval_out/sdxl_tuning_s50W
# num_frames=20
# sample_frame=20 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae2_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-04T10-24-16_train-sdxl_vae2_multi/checkpoints/trainstep_checkpoints/epoch\=000010-step\=000500000.ckpt 
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=0 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}  > sdxl_tuning_s50W.log   2>&1 



# is_vis=True
# save_eval_file=vae_eval_out/sdxl_vae_full_component_35W.log 
# output_folder=vae_eval_out/sdxl_vae_full_component_35W
# num_frames=20
# sample_frame=20 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_full_component_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-09T00-41-28_train-sdxl_vae_full_component_multi/checkpoints/trainstep_checkpoints/epoch=000317-step=000350000.ckpt  
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=1 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}  > sdxl_vae_full_component_35W.log   2>&1 




# is_vis=True
# save_eval_file=vae_eval_out/evaluation_sdxl_vae_video_causal_componential_35W.log 
# output_folder=vae_eval_out/evaluation_sdxl_vae_video_causal_componential_35W
# num_frames=20
# sample_frame=20 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_video_causal_componential_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-12T18-00-27_train-sdxl_vae_video_causal_componential_multi/checkpoints/trainstep_checkpoints/epoch\=000011-step\=000050000.ckpt
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=7 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}  > evaluation_sdxl_vae_video_causal_componential_35W.log   2>&1 


# is_vis=True
# save_eval_file=vae_eval_out/evaluation_sdxl_vae_video_componential_continual_22epoch.log 
# output_folder=vae_eval_out/evaluation_sdxl_vae_video_componential_continual_22epoch
# num_frames=20
# sample_frame=20
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_video_componential_continual_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-13T20-20-43_train-sdxl_vae_video_componential_continual_multi/checkpoints/trainstep_checkpoints/epoch=000022.ckpt
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=7 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}   > evaluation_sdxl_vae_video_componential_continual_22epoch.log   2>&1 

# is_vis=True
# save_eval_file=vae_eval_out/evaluation_sdxl_vae_video_componential_continual_new_6epoch.log 
# output_folder=vae_eval_out/evaluation_sdxl_vae_video_componential_continual_new_6epoch
# num_frames=20
# sample_frame=20 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_video_componential_continual_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-15T19-01-15_train-sdxl_vae_video_componential_continual_multi/checkpoints/trainstep_checkpoints/epoch\=000006.ckpt
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=7 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}   > evaluation_sdxl_vae_video_componential_continual_new_6epoch.log   2>&1 


# is_vis=True
# save_eval_file=vae_eval_out/evaluation_sdxl_vae_video_componential_dynamic_13epoch.log 
# output_folder=vae_eval_out/evaluation_sdxl_vae_video_componential_dynamic_13epoch
# num_frames=14
# sample_frame=14 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_video_componential_dynamic_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-16T23-13-07_train-sdxl_vae_video_componential_dynamic_multi/checkpoints/trainstep_checkpoints/epoch\=000013.ckpt
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=1 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}   > evaluation_sdxl_vae_video_componential_dynamic_13epoch.log   2>&1 


# is_vis=True
# save_eval_file=vae_eval_out/evaluation_sdxl_vae_video_dynamic_from_scratch_4epoch.log 
# output_folder=vae_eval_out/evaluation_sdxl_vae_video_dynamic_from_scratch_4epoch
# num_frames=14
# sample_frame=14 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_video_componential_dynamic_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-18T00-48-45_train-sdxl_vae_video_componential_dynamic_multi/checkpoints/trainstep_checkpoints/epoch\=000004.ckpt
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=1 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}   > evaluation_sdxl_vae_video_dynamic_from_scratch_4epoch.log   2>&1 


# is_vis=True
# save_eval_file=vae_eval_out/evaluation_sdxl_vae_video_dynamic_from_scratch_image_video_joint_4epoch.log 
# output_folder=vae_eval_out/evaluation_sdxl_vae_video_dynamic_image_video_joint_4epoch
# num_frames=14
# sample_frame=14 
# config_file=/mnt/data/yifanzhang/talking-head/configs/train/sdxl_vae_video_componential_dynamic_multi.yaml
# ckpt_path=/mnt/data/yifanzhang/talking-head/logs/2024-07-18T10-42-49_train-sdxl_vae_video_componential_dynamic_multi/checkpoints/trainstep_checkpoints/epoch\=000004.ckpt
# input_path=/mnt/data/public/dataset/talking_head/video_datasets
# metadata_paths=/mnt/data/public/dataset/talking_head/vae_csv/vfhq_val_metadata.csv
# CUDA_VISIBLE_DEVICES=1 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}   > evaluation_sdxl_vae_video_dynamic_image_video_joint_4epoch.log   2>&1 

is_vis=True
save_eval_file=evaluation_sd_vae.log 
output_folder=vae_eval_out/evaluation_sd_vae
num_frames=14
sample_frame=14 
config_file=/mnt/data/jiachunpan/talking-pose-vae/configs/train/sd_vae_pose.yaml
ckpt_path=/mnt/data/jiachunpan/talking-pose-vae/logs/2024-10-23T18-39-35_train-sd_vae_pose/checkpoints/trainstep_checkpoints/epoch=000018-step=000010000.ckpt
input_path=/mnt/data/public/dataset/talking_body/training_videos/luoxiang_512_dwpose
metadata_paths=/mnt/data/public/dataset/talking_body/embeddings/luoxiang_512/metadata.jsonl
CUDA_VISIBLE_DEVICES=3 python eval_video_vae_pipline.py --sample_frame ${sample_frame} --save_eval_file ${save_eval_file} --num_frames ${num_frames} --config_file ${config_file} --ckpt_path ${ckpt_path} --input_path ${input_path} --metadata_paths ${metadata_paths}  --is_vis ${is_vis} --output_folder ${output_folder}   > evaluation_sd_vae.log   2>&1 




