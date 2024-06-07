source scripts/setup.sh

lr="1e-3"
batch_size=64

DATA_DIR="download/train/train_demo_TCGA/images"
classes_subset="default"
OPENAI_LOGDIR="./checkpoints/19classes_TCGA_classifier-128x128--batch_size_${batch_size}--lr_${lr}"

CLASSIFIER_FLAGS="\
--image_size 128 \
--classifier_depth 2
"

TRAIN_FLAGS="--lr ${lr} --batch_size ${batch_size}"
DATA_DIR="$DATA_DIR"

# If srun needs to be used for multi-node or multi-GPU execution, add something like this to the front of the command line
# srun --jobid 34360485 -u --nodes 1 --ntasks 2 --pty \
CMD=(
    python -u scripts/classifier_train.py \
    --data_dir "$DATA_DIR" \
    $CLASSIFIER_FLAGS \
    $TRAIN_FLAGS
)

OPENAI_LOGDIR="$OPENAI_LOGDIR" \
NCCL_DEBUG=INFO \
AUTO_RESUME=True \
"${CMD[@]}"