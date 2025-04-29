#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1,2,3
save_path=output/retrieval_flickr_compression_p0.5
mkdir $save_path

#!/bin/bash
python -m torch.distributed.run --nproc_per_node=4 --master_port 20603 compress_retrieval_flickr_dtp.py --p 0.5 --epoch 10 \
--pretrained pretrained/model_base_retrieval_flickr.pth \
--config ./configs/retrieval_flickr.yaml \
--enable-sparse-api --weight-sparse-pattern "m8n2_1d_faster" \
--amp \
--token_reduce \
--output_dir $save_path >$save_path/train.log
