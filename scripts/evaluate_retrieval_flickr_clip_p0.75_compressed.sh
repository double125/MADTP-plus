#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_retrieval_clip_dtp.py --evaluate \
--pretrained output/retrieval_flickr_clip_compression_p0.75/checkpoint_best.pth \
--config ./configs/retrieval_flickr_clip.yaml \
--enable-sparse-api \
--from-dense-checkpoint 0 \
--weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir output/retrieval_flickr_clip_compression_p0.75