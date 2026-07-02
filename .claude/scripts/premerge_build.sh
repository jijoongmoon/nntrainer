set -e
cd /tmp/nntr_premerge
# match the working x86 config; disable sibling apps needing opencv/jsoncpp
sed -i 's#^  subdir(.YOLOv2/jni.)#  # subdir("YOLOv2/jni")#' Applications/meson.build 2>/dev/null || true
meson setup build -Denable-app=true -Denable-test=false -Denable-tflite-backbone=false -Denable-tflite-interpreter=false -Denable-transformer=true >/tmp/premerge_setup.log 2>&1 || { echo SETUP_FAIL; tail -5 /tmp/premerge_setup.log; exit 1; }
ninja -C build nntr_causallm >/tmp/premerge_ninja.log 2>&1 && echo PREMERGE_BUILD_OK || { echo PREMERGE_BUILD_FAIL; tail -8 /tmp/premerge_ninja.log; }
