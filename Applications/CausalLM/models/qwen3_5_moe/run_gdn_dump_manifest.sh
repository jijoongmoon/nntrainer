#!/usr/bin/env bash
# Build + run the weight-manifest dumper against a real model dir.
# Usage: run_gdn_dump_manifest.sh <model_dir> [manifest_out] [weights_bin]
# With [weights_bin]: also positional-loads the bin (P4 load validation).
set -euo pipefail
NNT=/home/aisjetson/jijoongmoon/nntrainer-s
BUILD="$NNT/build"
DIR="$NNT/Applications/CausalLM/models/qwen3_5_moe"
MODEL_DIR="${1:?usage: run_gdn_dump_manifest.sh <model_dir> [manifest_out] [weights_bin]}"
MANIFEST_OUT="${2:-$MODEL_DIR/weights_manifest.txt}"
WEIGHTS_BIN="${3:-}"
OUT=/tmp/gdn_dump_manifest_bin
cd "$BUILD"
echo "[init] compiling driver..."
python3 - "$DIR" "$OUT" "$BUILD" "$NNT" <<'PY'
import json, shlex, subprocess, sys
d, out, build, nnt = sys.argv[1:5]
cc = json.load(open('compile_commands.json'))
c = [x for x in cc if x['file'].endswith('CausalLM/main.cpp')][0]
toks = shlex.split(c['command'])
keep = [t for t in toks if t.startswith(('-I','-D','-std=','-march=')) or t in ('-fPIC','-pthread','-O3','-ftree-vectorize')]
obl = nnt + '/subprojects/OpenBLAS/build/lib'
shim = '/home/aisjetson/.local/lib/nntr-opencl-shim'
argv = ['c++','-O2',*keep, f'{d}/gdn_dump_manifest.cpp','-o',out,
        f'-L{build}/Applications/CausalLM','-lcausallm',
        f'-L{build}/nntrainer','-lnntrainer',
        f'-L{obl}', f'-L{shim}', f'-Wl,-rpath-link,{obl}','-L/usr/local/cuda/lib64',
        f'-Wl,-rpath,{build}/Applications/CausalLM',f'-Wl,-rpath,{build}/nntrainer',
        f'-Wl,-rpath,{obl}','-Wl,-rpath,/usr/local/cuda/lib64']
print('  c++ ... (%d flags) -lcausallm -lnntrainer' % len(keep)); sys.exit(subprocess.call(argv))
PY
echo "[init] running (NNTR_ENGINE=cpu) -> $MANIFEST_OUT"
export LD_LIBRARY_PATH="$BUILD/Applications/CausalLM:$BUILD/Applications/CausalLM/models/qwen3_5_moe:$BUILD/nntrainer:$NNT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NNTR_ENGINE=cpu
if [ -n "$WEIGHTS_BIN" ]; then
  "$OUT" "$MODEL_DIR" "$WEIGHTS_BIN" > "$MANIFEST_OUT"
else
  "$OUT" "$MODEL_DIR" > "$MANIFEST_OUT"
fi
tail -3 "$MANIFEST_OUT"
wc -l "$MANIFEST_OUT"
