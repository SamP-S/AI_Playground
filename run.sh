
# parser.add_argument('--LOAD_FROM', type=str, help='The model to load from')
# parser.add_argument('--NUM_EPOCHS', type=int, help='The number of epochs', default=50)
# parser.add_argument('--BATCH_SIZE', type=int, help='The batch size', default=32)
# parser.add_argument('--DATA_DIR', type=str, help='The data directory', default="data/data/bricks/50k_cycles")
# parser.add_argument('--MODEL_DIR', type=str, help='The model directory', default="data/data/models/50k_cycles")
# parser.add_argument('--MODEL_NAME', type=str, help='The model name', default="resnet50")
# parser.add_argument('--USE_GPU', type=int, help='Use GPU if available', default=1)

# resnet50
# mobilenet_v3_large

DATA="data/the_dataset/complete_datasets"
MODELS="data/the_dataset/models"

function train_model() {
    local model=$1
    local ds=$2

    echo "start training $model @ $ds"
    mkdir -p $MODELS/$ds
    python pytorch_train.py --DATA_DIR "$DATA/$ds" --MODEL_DIR $MODELS/$ds/$model --MODEL_NAME "$model" > $MODELS/$ds/$model.log
    echo "finish training $model @ $ds"
}

#train_model "resnet50" "mix_add_bg_nLEGO"
#train_model "resnet50" "mix_add_bg_wLEGO"
#train_model "resnet50" "mix_light_post_nLEGO"
#train_model "resnet50" "mix_light_post_wLEGO"
#train_model "resnet50" "mix_medium_post_nLEGO"
#train_model "resnet50" "mix_medium_post_wLEGO"
#train_model "resnet50" "mix_raw_nLEGO"
#train_model "resnet50" "mix_raw_wLEGO"
#train_model "resnet50" "mix_strong_post_nLEGO"
#train_model "resnet50" "mix_strong_post_wLEGO"
#train_model "resnet50" "real"

train_model "mobilenet_v3_large" "mix_medium_post_nLEGO"

