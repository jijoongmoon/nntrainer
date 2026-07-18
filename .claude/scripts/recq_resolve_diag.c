// recq_resolve_diag — localize WHY clGetExtensionFunctionAddressForPlatform
// returns NULL for the QCOM recording fns through nntrainer's loader path.
// Mirrors nntrainer's loader EXACTLY: dlopen("libOpenCL.so") + dlsym, then
// enumerates every platform and tries to resolve clNewRecordingQCOM against
// each, via BOTH clGetExtensionFunctionAddressForPlatform and the deprecated
// clGetExtensionFunctionAddress. Also tries a NULL platform.
// Build: aarch64 NDK clang, -ldl (no -lOpenCL — we dlopen like nntrainer does).
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int (*pfnGetPlatformIDs)(unsigned, void **, unsigned *);
typedef int (*pfnGetPlatformInfo)(void *, unsigned, size_t, void *, size_t *);
typedef void *(*pfnGetExtForPlatform)(void *, const char *);
typedef void *(*pfnGetExt)(const char *);

#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_EXTENSIONS 0x0904

static const char *kFns[] = {"clNewRecordingQCOM", "clEndRecordingQCOM",
                             "clEnqueueRecordingQCOM", "clReleaseRecordingQCOM"};

int main(int argc, char **argv) {
  const char *lib = argc > 1 ? argv[1] : "libOpenCL.so";
  void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    printf("dlopen(%s) FAILED: %s\n", lib, dlerror());
    return 1;
  }
  printf("dlopen(%s) OK\n", lib);

  pfnGetPlatformIDs getIds = (pfnGetPlatformIDs)dlsym(h, "clGetPlatformIDs");
  pfnGetPlatformInfo getInfo =
    (pfnGetPlatformInfo)dlsym(h, "clGetPlatformInfo");
  pfnGetExtForPlatform getExtP =
    (pfnGetExtForPlatform)dlsym(h, "clGetExtensionFunctionAddressForPlatform");
  pfnGetExt getExt =
    (pfnGetExt)dlsym(h, "clGetExtensionFunctionAddress");
  printf("dlsym: getIds=%p getInfo=%p getExtForPlatform=%p getExt=%p\n",
         (void *)getIds, (void *)getInfo, (void *)getExtP, (void *)getExt);
  if (!getIds || !getExtP) {
    printf("FAIL: core resolvers missing\n");
    return 1;
  }

  void *plats[8];
  unsigned n = 0;
  if (getIds(8, plats, &n) != 0 || n == 0) {
    printf("FAIL: clGetPlatformIDs n=%u\n", n);
    return 1;
  }
  printf("platforms: %u\n", n);

  // NULL-platform resolve (some ICD loaders only answer this way)
  for (int f = 0; f < 4; ++f) {
    void *p = getExtP(NULL, kFns[f]);
    printf("  [NULL plat] getExtForPlatform(%s) = %p\n", kFns[f], p);
  }
  if (getExt) {
    for (int f = 0; f < 4; ++f) {
      void *p = getExt(kFns[f]);
      printf("  [no-plat ] getExt(%s) = %p\n", kFns[f], p);
    }
  }

  for (unsigned i = 0; i < n; ++i) {
    char name[256] = {0}, ext[4096] = {0};
    if (getInfo) {
      getInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
      getInfo(plats[i], CL_PLATFORM_EXTENSIONS, sizeof(ext), ext, NULL);
    }
    int advertises = strstr(ext, "cl_qcom_recordable_queues") ? 1 : 0;
    printf("platform[%u] %p name='%s' advertises_recordable=%d\n", i, plats[i],
           name, advertises);
    for (int f = 0; f < 4; ++f) {
      void *p = getExtP(plats[i], kFns[f]);
      printf("    getExtForPlatform(%s) = %p\n", kFns[f], p);
    }
  }
  return 0;
}
