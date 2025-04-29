#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_caption_dtp.py --evaluate \
--pretrained output/caption_coco_compression_p0.5/model_base_caption_capfilt_large_coco_p0.5_compressed.pth \
--config ./configs/caption_coco.yaml \
--enable-sparse-api \
--from-dense-checkpoint 0 \
--weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir output/caption_coco_compression_p0.5