#!/bin/bash
save_path=output/nlvr_nlvr2_compression_p0.7
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=4 --master_port 20603 compress_nlvr_dtp.py --epoch 25 --p 0.7 \
--pretrained pretrained/model_base_nlvr.pth \
--config ./configs/nlvr.yaml \
--enable-sparse-api --weight-sparse-pattern "m8n4_1d_faster" \
--output_dir $save_path >$save_path/train.log

