// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/filament/render_context_filament.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "experimental/filament/filament/filament_context.h"


#if defined(TLS_FILAMENT_CONTEXT)
static thread_local mujoco::FilamentContext* g_filament_context = nullptr;
#else
static mujoco::FilamentContext* g_filament_context = nullptr;
#endif

static void CheckFilamentContext() {
  if (g_filament_context == nullptr) {
    mju_error("Missing context; did you call mjrf_makeFilamentContext?");
  }
}

extern "C" {

void mjrf_defaultFilamentConfig(mjrFilamentConfig* config) {
  memset(config, 0, sizeof(mjrFilamentConfig));
}

void mjrf_makeFilamentContext(const mjModel* m, mjrContext* con,
                             const mjrFilamentConfig* config) {
  // TODO: Support multiple contexts and multiple threads. For now, we'll just
  // assume a single, global context.
  if (g_filament_context != nullptr) {
    mju_error("Context already exists!");
  }
  g_filament_context = new mujoco::FilamentContext(config);
  g_filament_context->Init(m);
}

void mjrf_defaultContext(mjrContext* con) {
  memset(con, 0, sizeof(mjrContext));
}

// File-based resource for serving filament .filamat assets from disk.
struct FilamentFileResource {
  std::vector<char> data;
  int size = 0;
};

static bool g_filament_provider_registered = false;

// Discover the directory containing the mujoco library itself, so we can
// resolve "assets/" relative to it (e.g. inside a Python wheel's
// site-packages/mujoco/).  Falls back to empty string (CWD) if discovery fails.
static std::string GetLibraryAssetsDir() {
  // Use the address of an exported mujoco function to locate the library.
  // dladdr / GetModuleHandleEx resolve the shared object containing this symbol.
  void* anchor = reinterpret_cast<void*>(&mj_forward);

#if defined(_WIN32) || defined(__CYGWIN__)
  HMODULE hModule = NULL;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                        (LPCTSTR)anchor, &hModule)) {
    char path[MAX_PATH];
    if (GetModuleFileName(hModule, path, MAX_PATH)) {
      std::string lib_path(path);
      size_t last_slash = lib_path.find_last_of("\\/");
      if (last_slash != std::string::npos) {
        return lib_path.substr(0, last_slash + 1) + "assets/";
      }
    }
  }
#else
  Dl_info info;
  if (dladdr(anchor, &info) && info.dli_fname) {
    std::string lib_path(info.dli_fname);
    size_t last_slash = lib_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return lib_path.substr(0, last_slash + 1) + "assets/";
    }
  }
#endif
  return "";
}

static void EnsureFilamentResourceProvider() {
  if (g_filament_provider_registered) return;
  g_filament_provider_registered = true;

  // Compute the library-relative assets path once.
  static std::string lib_assets_dir = GetLibraryAssetsDir();

  static mjpResourceProvider provider;
  mjp_defaultResourceProvider(&provider);

  provider.open = [](mjResource* resource) -> int {
    // Strip "filament:" prefix to get the asset filename.
    std::string_view name(resource->name);
    auto pos = name.find(':');
    std::string subpath = (pos != std::string_view::npos)
        ? std::string(name.substr(pos + 1))
        : std::string(name);

    // Try library-relative path first (e.g. site-packages/mujoco/assets/),
    // then fall back to CWD-relative "assets/" for standalone binaries.
    std::string candidates[2] = {
      lib_assets_dir + subpath,
      "assets/" + subpath,
    };

    for (const auto& path : candidates) {
      if (path.empty()) continue;
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file.is_open()) continue;

      auto* fr = new FilamentFileResource();
      fr->size = static_cast<int>(file.tellg());
      fr->data.resize(fr->size);
      file.seekg(0, std::ios::beg);
      file.read(fr->data.data(), fr->size);
      resource->data = fr;
      return fr->size;
    }
    return 0;
  };
  provider.read = [](mjResource* resource, const void** buffer) -> int {
    auto* fr = static_cast<FilamentFileResource*>(resource->data);
    if (!fr) return 0;
    *buffer = fr->data.data();
    return fr->size;
  };
  provider.close = [](mjResource* resource) {
    delete static_cast<FilamentFileResource*>(resource->data);
    resource->data = nullptr;
  };

  provider.prefix = "filament";
  int slot = mjp_registerResourceProvider(&provider);
  if (slot < 0) {
    mju_error("Failed to register filament resource provider (slot=%d)", slot);
  }
}

void mjrf_makeContext(const mjModel* m, mjrContext* con, int fontscale) {
  mjrf_freeContext(con);
  EnsureFilamentResourceProvider();
  mjrFilamentConfig cfg;
  mjrf_defaultFilamentConfig(&cfg);
  cfg.width = m->vis.global.offwidth;
  cfg.height = m->vis.global.offheight;
  mjrf_makeFilamentContext(m, con, &cfg);
}

void mjrf_freeContext(mjrContext* con) {
  // mjr_freeContext may be called multiple times.
  if (g_filament_context) {
    delete g_filament_context;
    g_filament_context = nullptr;
  }
  mjrf_defaultContext(con);
}

void mjrf_render(mjrRect viewport, mjvScene* scn, const mjrContext* con) {
  CheckFilamentContext();
  g_filament_context->Render(viewport, scn);
}

void mjrf_uploadMesh(const mjModel* m, const mjrContext* con, int meshid) {
  CheckFilamentContext();
  g_filament_context->UploadMesh(m, meshid);
}

void mjrf_uploadTexture(const mjModel* m, const mjrContext* con, int texid) {
  CheckFilamentContext();
  g_filament_context->UploadTexture(m, texid);
}

void mjrf_uploadHField(const mjModel* m, const mjrContext* con, int hfieldid) {
  CheckFilamentContext();
  g_filament_context->UploadHeightField(m, hfieldid);
}

void mjrf_setBuffer(int framebuffer, mjrContext* con) {
  CheckFilamentContext();
  g_filament_context->SetFrameBuffer(framebuffer);
}

void mjrf_readPixels(unsigned char* rgb, float* depth, mjrRect viewport,
                          const mjrContext* con) {
  CheckFilamentContext();
  g_filament_context->ReadPixels(viewport, rgb, depth);
}

uintptr_t mjrf_uploadGuiImage(uintptr_t tex_id, const unsigned char* pixels,
                             int width, int height, int bpp,
                             const mjrContext* con) {
  CheckFilamentContext();
  return g_filament_context->UploadGuiImage(tex_id, pixels, width, height, bpp);
}

double mjrf_getFrameRate(const mjrContext* con) {
  CheckFilamentContext();
  return g_filament_context->GetFrameRate();
}

void mjrf_updateGui(const mjrContext* con) {
  if (g_filament_context != nullptr) {
    g_filament_context->UpdateGui();
  }
}

}  // extern "C"
