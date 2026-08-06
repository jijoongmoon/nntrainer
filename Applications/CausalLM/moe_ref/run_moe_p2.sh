#!/usr/bin/env bash
# Build + run the P2 Qwen3_5Moe SparseMoeBlock CPU reference vs goldens.
# Standalone against the existing build/ libnntrainer.so (no meson reconfigure).
set -euo pipefail

NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
SRC="$NNT/Applications/CausalLM/moe_ref/moe_p2_test.cpp"
OUT=/home/aisjetson/jijoongmoon/nntrainer-s/build/refbin/moe_p2
mkdir -p "$(dirname "$OUT")"

cd "$BUILD"

echo "[moe-p2] compiling..."
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

echo "[moe-p2] running..."
export LD_LIBRARY_PATH="$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
rc=0
echo "### case 1: T=6 E=8 K=2 HID=32 INTER=16 SINTER=16"
MOE_T=6 MOE_E=8 MOE_K=2 MOE_HID=32 MOE_INTER=16 MOE_SINTER=16 \
  MOE_DIR=/home/aisjetson/jijoongmoon/moe_p2/bin "$OUT" || rc=$?
echo "### case 2: T=20 E=16 K=4 HID=40 INTER=24 SINTER=20"
MOE_T=20 MOE_E=16 MOE_K=4 MOE_HID=40 MOE_INTER=24 MOE_SINTER=20 \
  MOE_DIR=/home/aisjetson/jijoongmoon/moe_p2/bin_case2 "$OUT" || rc=$?
exit $rc
