#!/system/bin/sh
D=/data/local/tmp/nntrainer/causallm
cd $D
export LD_LIBRARY_PATH=$D NNTR_NUM_THREADS=4 NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1
PROMPT="$(cat $D/prompt_1p2k.txt)"
./nntrainer_causallm models/gemma4_lmint4 "$PROMPT"
