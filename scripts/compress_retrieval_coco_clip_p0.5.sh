#!/bin/bash
save_path=output/retrieval_coco_clip_compression_p0.5
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=4 --master_port 20603 compress_retrieval_clip_dtp.py --p 0.5 --epoch 5 \
--pretrained pretrained/clip_large_retrieval_coco.pth \
--config ./configs/retrieval_coco_clip.yaml \
--enable-sparse-api --weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir $save_path 
>$save_path/train.log
