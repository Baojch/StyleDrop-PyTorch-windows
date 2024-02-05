set EVAL_CKPT=assets/ckpts/cc3m-285000.ckpt
set ADAPTER=./style_adapter/0102.pth
set OUTPUT_DIR=./output
accelerate launch --num_processes 1 --mixed_precision fp16 train_t2i_custom_v2.py --config=configs/custom.py