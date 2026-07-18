#!/system/bin/sh
D=/data/local/tmp/nntrainer/causallm
cd $D
export LD_LIBRARY_PATH=$D
export NNTR_NUM_THREADS=4
export NNTR_MHA_GPU=1
export NNTR_GPU_SVM_POOL=1
PROMPT="$(cat $D/prompt_1k.txt)"
if [ "$1" = "img" ]; then export NNTR_KV_IMG_ATTN=1; fi
./nntrainer_causallm models/gemma2_lg "$PROMPT"
