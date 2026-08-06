#!/usr/bin/env bash
# Build + run the P3b Qwen3_5Moe decoder-layer assembly CPU reference vs goldens.
set -euo pipefail
NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
SRC="$NNT/Applications/CausalLM/e2e_ref/e2e_p3c_test.cpp"
OUT=/home/aisjetson/jijoongmoon/nntrainer-s/build/refbin/e2e_p3c
mkdir -p "$(dirname "$OUT")"
cd "$BUILD"
echo "[p3c] compiling..."
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
echo "[p3c] running..."
export LD_LIBRARY_PATH="$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
rc=0
echo "### case 1: 2 layers [G,A], S=7"
E2E_DIR=/home/aisjetson/jijoongmoon/attn_p3/e2e_bin "$OUT" || rc=$?
echo "### case 2: 4 layers [G,G,G,A] (3:1), S=9"
E2E_DIR=/home/aisjetson/jijoongmoon/attn_p3/e2e_bin4 "$OUT" || rc=$?
exit $rc
