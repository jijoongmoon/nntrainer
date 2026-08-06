#!/usr/bin/env bash
# Build + run the P3a Qwen3_5Moe full-attention CPU reference vs goldens.
# Standalone against the existing build/ libnntrainer.so (no meson reconfigure).
set -euo pipefail

NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
SRC="$NNT/Applications/CausalLM/attn_ref/attn_p3a_test.cpp"
OUT=/home/aisjetson/jijoongmoon/nntrainer-s/build/refbin/attn_p3a
mkdir -p "$(dirname "$OUT")"

cd "$BUILD"
echo "[attn-p3a] compiling..."
python3 - "$SRC" "$OUT" "$BUILD" <<'PY'
import json, shlex, subprocess, sys
src, out, build = sys.argv[1], sys.argv[2], sys.argv[3]
cc = json.load(open('compile_commands.json'))
c = [x for x in cc if x['file'].endswith('nntrainer/tensor/tensor.cpp')][0]
toks = shlex.split(c['command'])
keep = [t for t in toks
        if t.startswith(('-I', '-D', '-std=', '-march='))
        or t in ('-fPIC', '-pthread', '-O3', '-ftree-vectorize')]
obl = '/home/aisjetson/jijoongmoon/nntrainer-s/subprojects/OpenBLAS/build/lib'
argv = ['c++', '-O2', *keep, src, '-o', out,
        f'-L{build}/nntrainer', '-lnntrainer',
        f'-L{obl}', f'-Wl,-rpath-link,{obl}', '-L/usr/local/cuda/lib64',
        f'-Wl,-rpath,{build}/nntrainer', f'-Wl,-rpath,{obl}',
        '-Wl,-rpath,/usr/local/cuda/lib64']
print('  ' + ' '.join(shlex.quote(a) for a in argv[:6]) + ' ... (%d flags)' % len(keep))
sys.exit(subprocess.call(argv))
PY

echo "[attn-p3a] running..."
export LD_LIBRARY_PATH="$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
rc=0
echo "### case 1: B=1 S=6 nH=4 nKV=2 hd=16"
ATT_B=1 ATT_S=6 ATT_NH=4 ATT_NKV=2 ATT_HD=16 ATT_HID=32 ATT_ROT=4 \
  ATT_DIR=/home/aisjetson/jijoongmoon/attn_p3/bin "$OUT" || rc=$?
if [ -d /home/aisjetson/jijoongmoon/attn_p3/bin_case2 ]; then
  echo "### case 2: B=2 S=12 nH=6 nKV=3 hd=24"
  ATT_B=2 ATT_S=12 ATT_NH=6 ATT_NKV=3 ATT_HD=24 ATT_HID=48 ATT_ROT=6 \
    ATT_DIR=/home/aisjetson/jijoongmoon/attn_p3/bin_case2 "$OUT" || rc=$?
fi
exit $rc
