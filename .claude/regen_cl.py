#!/usr/bin/env python3
# Regenerate builddir/.../<name>.cpp from the edited .cl source, matching the
# meson configure_file transform (each line -> R"(line)" "\n").
import sys
name = sys.argv[1]  # e.g. two_conv_attention
var = sys.argv[2] if len(sys.argv) > 2 else name + "_kernel"
src = f"nntrainer/tensor/cl_operations/cl_kernels/{name}.cl"
out = f"builddir/nntrainer/tensor/cl_operations/cl_kernels/{name}.cpp"
lines = open(src).read().split('\n')
body = '\n'.join('R"(' + ln + ')" "\\n"' for ln in lines)
content = (
    f'#include "{name}.h"\n\n'
    f'namespace nntrainer {{\n'
    f'const std::string {var} = {{ \n'
    f'{body}\n'
    f'}};\n'
    f'}} // namespace nntrainer\n'
)
open(out, 'w').write(content)
print(f"regenerated {out} ({len(lines)} lines)")
