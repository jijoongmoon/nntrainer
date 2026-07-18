#!/system/bin/sh
D=/data/local/tmp/nntrainer/causallm
cd $D
export LD_LIBRARY_PATH=$D
export NNTR_NUM_THREADS=4
export NNTR_FC_INT8_GPU=1
export NNTR_MHA_GPU=1
export NNTR_GPU_SVM_POOL=1
export NNTR_KV_IMG_ATTN=1
export NNTR_GPU_CLMEM_POOL=1
# no prompt arg -> uses sample_input (templated) from nntr_config.json
./nntrainer_causallm models/gemma4_qint4fp16
