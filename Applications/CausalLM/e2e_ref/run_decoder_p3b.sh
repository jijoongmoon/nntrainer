#!/usr/bin/env bash
# Build + run the P3b Qwen3_5Moe decoder-layer assembly CPU reference vs goldens.
set -euo pipefail
NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
SRC="$NNT/Applications/CausalLM/e2e_ref/decoder_p3b_test.cpp"
OUT=/home/aisjetson/jijoongmoon/nntrainer-s/build/refbin/decoder_p3b
mkdir -p "$(dirname "$OUT")"
cd "$BUILD"
echo "[p3b] compiling..."
python3 - "$SRC" "$OUT" "$BUILD" <<'PY'
import json, shlex, subprocess, sys
src, out, build = sys.argv[1], sys.argv[2], sys.argv[3]
cc = json.load(open('compile_commands.json'))
c = [x for x in cc if x['file'].endswith('nntrainer/tensor/tensor.cpp')][0]
toks = shlex.split(c['command'])
keep = [t for t in toks if t.startswith(('-I','-D','-std=','-march=')) or t in ('-fPIC','-pthread','-O3','-ftree-vectorize')]
obl = '/home/aisjetson/jijoongmoon/nntrainer-s/subprojects/OpenBLAS/build/lib'
argv = ['c++','-O2',*keep,src,'-o',out,f'-L{build}/nntrainer','-lnntrainer',
        f'-L{obl}',f'-Wl,-rpath-link,{obl}','-L/usr/local/cuda/lib64',
        f'-Wl,-rpath,{build}/nntrainer',f'-Wl,-rpath,{obl}','-Wl,-rpath,/usr/local/cuda/lib64']
print('  c++ ... (%d flags)' % len(keep)); sys.exit(subprocess.call(argv))
PY
echo "[p3b] running..."
export LD_LIBRARY_PATH="$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
"$OUT"
