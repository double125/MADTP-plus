#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_nlvr_dtp.py --evaluate \
--pretrained output/nlvr_nlvr2_compression_p0.7/checkpoint_best.pth \
--config ./configs/nlvr.yaml \
--enable-sparse-api \
--from-dense-checkpoint 0 \
--weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir output/nlvr_nlvr2_compression_p0.7