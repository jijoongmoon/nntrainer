#!/usr/bin/env bash
# Build + run the standalone validation of the REAL GatedDeltaNetLayer (LayerImpl)
# against the P1 goldens. Compiles the layer + driver against the prebuilt
# build/libnntrainer.so (no meson reconfigure).
set -euo pipefail
NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
DIR="$NNT/Applications/CausalLM/models/qwen3_5_moe"
OUT=/home/aisjetson/jijoongmoon/nntrainer-s/build/refbin/gdn_layer_driver
mkdir -p "$(dirname "$OUT")"
cd "$BUILD"
echo "[gdn-layer] compiling layer + driver..."
python3 - "$DIR" "$OUT" "$BUILD" <<'PY'
import json, shlex, subprocess, sys
d, out, build = sys.argv[1], sys.argv[2], sys.argv[3]
cc = json.load(open('compile_commands.json'))
c = [x for x in cc if x['file'].endswith('nntrainer/tensor/tensor.cpp')][0]
toks = shlex.split(c['command'])
keep = [t for t in toks if t.startswith(('-I','-D','-std=','-march=')) or t in ('-fPIC','-pthread','-O3','-ftree-vectorize')]
obl = '/home/aisjetson/jijoongmoon/nntrainer-s/subprojects/OpenBLAS/build/lib'
argv = ['c++','-O2',*keep, f'-I{d}',
        f'{d}/gated_delta_net_layer.cpp', f'{d}/gdn_layer_driver.cpp',
        '-o', out, f'-L{build}/nntrainer','-lnntrainer',
        f'-L{obl}',f'-Wl,-rpath-link,{obl}','-L/usr/local/cuda/lib64',
        f'-Wl,-rpath,{build}/nntrainer',f'-Wl,-rpath,{obl}','-Wl,-rpath,/usr/local/cuda/lib64']
print('  c++ ... (%d flags) + 2 srcs' % len(keep)); sys.exit(subprocess.call(argv))
PY
echo "[gdn-layer] running..."
export LD_LIBRARY_PATH="$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
"$OUT"
