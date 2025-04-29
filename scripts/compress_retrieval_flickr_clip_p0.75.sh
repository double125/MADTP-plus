#!/bin/bash
# save_path=output/retrieval_flickr_clip_compression_p0.55_onlysparse_lr1e5_w2KD_m8n4_faster
save_path=output/retrieval_flickr_clip_compression_p0.75
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=4 --master_port 20603 compress_retrieval_clip_dtp.py --p 0.75 --epoch 10 \
--pretrained pretrained/clip_large_retrieval_flickr.pth \
--config ./configs/retrieval_flickr_clip.yaml \
--enable-sparse-api --weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir $save_path >$save_path/train.log

