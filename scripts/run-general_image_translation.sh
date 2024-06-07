source scripts/setup.sh

noise_schedule="linear"
batch_size=8
mode="$1"
DATA_DIR="$2"
OPENAI_LOGDIR="$3"

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

source "${SCRIPT_DIR}/dataset_specific_configurations/${mode}.sh"

MODEL_FLAGS="--model_type $model_type \
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
--use_scale_shift_norm True \
--model_path "$model_path"
"

DIFFUSION_FLAGS="--diffusion_steps 1000 \
--noise_schedule $noise_schedule"


CLASSIFIER_FLAGS="
--classifier_scale 1.0 \
--classifier_depth 2 \
--classifier_path $classifier_path"

TRANSLATION_FLAGS="
--translation_spec $OPENAI_LOGDIR/modification.pkl \
--batch_size $batch_size \
"

DATA_DIR="$DATA_DIR"

# If srun needs to be used for multi-node or multi-GPU execution, add something like this to the front of the command line
# srun -u --jobid <jobid> --nodes 1 --ntasks 2 \
CMD=(
    python -u scripts/general_image_translation.py \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $CLASSIFIER_FLAGS \
    $TRANSLATION_FLAGS \
    $DATA_OPTIONS
)
OPENAI_LOGDIR="$OPENAI_LOGDIR" \
"${CMD[@]}"