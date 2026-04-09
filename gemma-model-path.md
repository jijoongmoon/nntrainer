/sdcard/Android/data/com.example.QuickAI/files/models/gemma-4-E2B-it/gemma-4-E2B-it.litertlm

# Why not /data/local/tmp ?
#
# /data/local/tmp is owned by the `shell` user (what adb shell runs as),
# not by arbitrary apps. On stock (user) builds it carries the
# shell_data_file SELinux context and the untrusted_app SELinux domain is
# denied read access — chmod 777 does not help. Use the LauncherApp's
# own external files dir instead: it needs no runtime permissions, is
# writable by adb, and is readable by the owning app on every Android
# version.
#
# First install LauncherApp (so the framework creates the dir), then:
#   adb push gemma-4-E2B-it.litertlm \
#     /sdcard/Android/data/com.example.QuickAI/files/models/gemma-4-E2B-it/
#
# The service pre-creates the nested models/ directory on first boot,
# so `adb push` works without `mkdir -p`.
