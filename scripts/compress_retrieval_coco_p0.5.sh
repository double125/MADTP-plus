#!/bin/bash
save_path=output/retrieval_coco_compression_p0.45
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=8 --master_port 30603 compress_retrieval_dtp.py --p 0.45 --epoch 5 \
--pretrained pretrained/model_base_retrieval_coco.pth \
--config ./configs/retrieval_coco.yaml \
--enable-sparse-api --weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir $save_path >$save_path/train.log