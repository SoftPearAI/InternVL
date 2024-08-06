set -x

PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACC=2

export MASTER_PORT=34229
export LAUNCHER="pytorch"

OUTPUT_DIR='work_dirs/00_internvl_chat_v1_5_internlm2_8b_dynamic_res_finetune_freezellm_unfreeze_vm_loravm16_basesysprompt'
RUN_NAME='00_internvl_chat_v1_5_internlm2_8b_dynamic_res_finetune_freezellm_unfreeze_vm_loravm16_basesysprompt'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# --use_backbone_lora 32 \
  # --drop_path_rate 0.4 \

deepspeed  --include localhost:4,5,6,7 --master_port ${MASTER_PORT} internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/InternVL2-8B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/home/danielko/src/softpear-rnd/generation/video/data/auto_captioning/InternVL/internvl_chat/shell/internlm2_20b_dynamic/metadata.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --use_llm_lora 32 \
  --use_backbone_lora 16 \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 5 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "epoch" \
  --do_eval True \
  --eval_meta_path "/home/danielko/src/softpear-rnd/generation/video/data/auto_captioning/InternVL/internvl_chat/shell/internlm2_20b_dynamic/eval_metadata.json" \
  --save_strategy "epoch" \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --run_name ${RUN_NAME} \
  --report_to "wandb"