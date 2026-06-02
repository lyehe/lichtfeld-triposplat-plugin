#!/usr/bin/env bash
# One-shot installer for the TripoSplat LichtFeld Studio plugin on Linux / macOS.
#
# Creates a single symlink:
#   <LFS plugins>/triposplat_plugin   -> this plugin directory
#
# The plugin is fully self-contained: the `triposplat` Python package is
# vendored at ./triposplat/, and the ~3.8 GB model weights (VAST-AI/TripoSplat,
# 5 safetensors files) are downloaded on first load into ./models/.
#
# Usage:
#   ./install.sh
#   LFS_PLUGINS_DIR=~/.lichtfeld/plugins ./install.sh
#   FORCE=1 ./install.sh                   # replace non-link targets

set -euo pipefail

plugin_root="$(cd "$(dirname "$0")" && pwd)"
lfs_plugins_dir="${LFS_PLUGINS_DIR:-$HOME/.lichtfeld/plugins}"

if [[ ! -f "$plugin_root/triposplat/triposplat.py" ]]; then
    echo "Vendored triposplat package not found at $plugin_root/triposplat/." >&2
    echo "This install is corrupt - re-clone the plugin repo." >&2
    exit 1
fi

plugin_link="$lfs_plugins_dir/triposplat_plugin"

replace_symlink() {
    local link="$1" target="$2"
    mkdir -p "$(dirname "$link")"
    if [[ -L "$link" ]]; then
        rm -f "$link"
    elif [[ -e "$link" ]]; then
        if [[ "${FORCE:-}" == "1" ]]; then
            rm -rf "$link"
        else
            echo "Refusing to replace non-link: $link  (set FORCE=1 to override)" >&2
            exit 1
        fi
    fi
    ln -s "$target" "$link"
}

echo ""
echo "== TripoSplat plugin installer =="
echo "Plugin root:     $plugin_root"
echo "LFS plugins dir: $lfs_plugins_dir"

echo ""
echo "Creating symlink..."
replace_symlink "$plugin_link" "$plugin_root"
echo "  [OK] $plugin_link"
echo "       -> $plugin_root"

echo ""
echo "Install complete."

echo ""
echo "Next steps:"
echo "  1. Launch LichtFeld Studio."
echo "  2. Wait for first-run 'uv sync' (heavy CUDA wheels)."
echo "  3. Open the 'TripoSplat' panel and generate from an image."
echo ""
echo "To uninstall: ./uninstall.sh"
