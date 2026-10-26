# Third-party code

Patches for third-party projects used here. The projects themselves aren't in this repo: clone each one at
the commit below and apply its patch.

| Folder | Upstream | Commit | Licence | What's here |
|---|---|---|---|---|
| `depth-anything-3` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | `41736238f5bced4debf3f2a12375d2466874866d` | Apache-2.0 | `depth-anything-3.patch` |

Depth Anything 3 goes in `third_party/depth_anything_3`, which `.gitignore` leaves out:

    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/depth_anything_3
    git -C third_party/depth_anything_3 checkout 41736238f5bced4debf3f2a12375d2466874866d
    git -C third_party/depth_anything_3 apply ../../external/depth-anything-3/depth-anything-3.patch

The patch stubs out the export functions, since only inference is used here.
