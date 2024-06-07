source scripts/setup.sh

lr="3e-5"
batch_size=16 # maximum batch_size is 32 on V100
noise_schedule="linear"

DATA_DIR="download/train/train_demo_TCGA/images"
CLASSIFIER_PATH="checkpoints/19classes_TCGA_classifier-128x128--batch_size_64--lr_1e-3/model000000.pt"
OPENAI_LOGDIR="./checkpoints/19class_TCGA_diffusion-128x128--batch_size_${batch_size}--lr_${lr}--ns_sched_${noise_schedule}"

MODEL_FLAGS="--model_type unet \
		--image_size 128 \
		--attention_resolutions 32,16,8 \
		--class_cond True \
		--dropout 0.1 \
		--learn_sigma True \
		--num_channels 256 \
		--num_heads 4 \
		--num_res_blocks 2 \
		--resblock_updown True \
		--use_new_attention_order True \
		--use_fp16 True \
		--use_scale_shift_norm True"

DIFFUSION_FLAGS="--diffusion_steps 1000 \
--noise_schedule $noise_schedule"

CLASSIFIER_FLAGS="
--classifier_scale 1.0 \
--classifier_depth 2 \
--classifier_path $CLASSIFIER_PATH"

TRAIN_FLAGS="--lr $lr \
--batch_size $batch_size \
"

DATA_OPTIONS="\
--data_dir "$DATA_DIR" \
--class_subset default \
"

DIFFUSION_SAMPLING_FLAGS=" \
--produce_diffusion_samples True \
--diffusion_sample_interval 1000 \
--num_diffusion_samples 1
"

# If srun needs to be used for multi-node or multi-GPU execution, add something like this to the front of the command line
# srun -u --jobid <jobid> --nodes 1 --ntasks 2
CMD=( 
    python -u scripts/image_train.py \
    $DATA_OPTIONS \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $CLASSIFIER_FLAGS \
    $TRAIN_FLAGS \
    $DIFFUSION_SAMPLING_FLAGS
)

OPENAI_LOGDIR="$OPENAI_LOGDIR" \
AUTO_RESUME="True" \
NCCL_DEBUG=INFO \
"${CMD[@]}"