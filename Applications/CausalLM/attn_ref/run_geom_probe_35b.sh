#!/usr/bin/env bash
# Build + run the P1 GatedDeltaNet CPU reference test against the P0 goldens.
# Compiles standalone against the existing build/ libnntrainer.so (no meson reconfigure).
set -euo pipefail

NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
SRC="$NNT/Applications/CausalLM/attn_ref/geom_probe_35b.cpp"
OUT=/home/aisjetson/jijoongmoon/nntrainer-s/build/refbin/geom_probe_35b
mkdir -p "$(dirname "$OUT")"

cd "$BUILD"

echo "[geom] compiling..."
# Drive the compiler from python with an argv list (no shell word-splitting), so
# string-valued -D macros from compile_commands survive intact. Reuse the exact
# include/define/arch flags of a representative tensor TU.
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
argv = ['c++', '-O2', *keep,
        '-I/home/aisjetson/jijoongmoon/nntrainer-s/Applications/CausalLM/layers',
        '-I/home/aisjetson/jijoongmoon/nntrainer-s/Applications/CausalLM',
        src, '-o', out,
        f'-L{build}/Applications/CausalLM/layers', '-lmha_core_layer',
        f'-Wl,-rpath,{build}/Applications/CausalLM/layers',
        f'-L{build}/nntrainer', '-lnntrainer',
        f'-L{obl}', f'-Wl,-rpath-link,{obl}', '-L/usr/local/cuda/lib64', '-lcudart',
        f'-Wl,-rpath,{build}/nntrainer', f'-Wl,-rpath,{obl}',
        '-Wl,-rpath,/usr/local/cuda/lib64']
print('  ' + ' '.join(shlex.quote(a) for a in argv[:6]) + ' ... (%d flags)' % len(keep))
sys.exit(subprocess.call(argv))
PY

echo "[geom] running..."
export LD_LIBRARY_PATH="$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
rc=0
rc=0
NNTR_ENGINE=cuda NNTR_FC_CUDA_DENSE_DBG=1 NNTR_CUDA_DBG=1 "$OUT" || rc=$?
exit $rc
