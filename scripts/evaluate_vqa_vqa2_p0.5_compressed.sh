#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_vqa_dtp.py --evaluate \
--pretrained output/vqa_vqa2_compression_p0.5/checkpoint_best.pth \
--config ./configs/vqa.yaml \
--enable-sparse-api \
--from-dense-checkpoint 0 \
--weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir output/vqa_vqa2_compression_p0.5