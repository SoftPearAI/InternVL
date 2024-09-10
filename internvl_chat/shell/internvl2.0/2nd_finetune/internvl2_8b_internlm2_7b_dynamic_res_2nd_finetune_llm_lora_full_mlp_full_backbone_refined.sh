set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export WANDB_PROJECT="internvl2"

OUTPUT_DIR='/mnt/vdb/dberezin/internvl/work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_llm_lora_full_mlp_full_backbone_refined'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --run_name internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_llm_lora_full_mlp_full_backbone_8epochs_refined \
  --model_name_or_path "OpenGVLab/InternVL2-8B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/llava_6200_labeled_img_v6.0_refined_train.json" \
  --eval_meta_path "./shell/data/llava_6200_labeled_img_v6.0_refined_val.json" \
  --test_meta_path "./shell/data/llava_6200_labeled_img_v6.0_refined_test.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone False \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 8 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 8 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --do_eval True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "wandb" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
