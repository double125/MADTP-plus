#!/bin/bash
save_path=output/caption_coco_compression_p0.5
mkdir $save_path

#!/bin/bash
python -m torch.distributed.run --nproc_per_node=4 --master_port 30703 compress_caption_dtp.py --p 0.5 --epoch 10 \
--pretrained pretrained/model_base_caption_capfilt_large.pth \
--config ./configs/caption_coco.yaml \
--enable-sparse-api --weight-sparse-pattern "m8n4_1d_faster" \
--amp \
--token_reduce \
--output_dir $save_path >$save_path/train.log 2>&1