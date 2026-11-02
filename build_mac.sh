#!/usr/bin/env bash
set -euo pipefail

echo "===================================================="
echo " AutoAnswer macOS Build Script"
echo "===================================================="

# Detect Architecture
ARCH="$(uname -m)"
echo "[INFO] Detected Host Architecture: ${ARCH}"

# Virtual Environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    echo "[WARNING] .venv/bin/activate not found. Using system python."
fi

# PyInstaller check
if ! command -v pyinstaller &> /dev/null; then
    echo "[ERROR] PyInstaller not found. Please install dependencies first."
    exit 1
fi

# Clean previous builds
echo "[1/3] Cleaning previous builds..."
rm -rf build dist

# Build .app bundle
APP_NAME="AutoAnswer_Mac_${ARCH}"
echo "[2/3] Building ${APP_NAME}.app..."
pyinstaller --noconfirm --windowed \
    --name "${APP_NAME}" \
    --add-data "action.schema.json:." \
    --collect-all customtkinter \
    --collect-all rapidocr_onnxruntime \
    --collect-all onnxruntime \
    main.py


# Package into DMG if hdiutil is available (macOS native tool)
echo "[3/3] Packaging into DMG..."
if command -v hdiutil &> /dev/null; then
    DMG_PATH="dist/${APP_NAME}.dmg"
    hdiutil create -volname "${APP_NAME}" -srcfolder "dist/${APP_NAME}.app" -ov -format UDZO "${DMG_PATH}"
    echo "===================================================="
    echo " macOS Build Succeeded: ${DMG_PATH}"
    echo "===================================================="
else
    echo "[INFO] hdiutil not found (non-macOS build host). App bundle located at: dist/${APP_NAME}.app"
fi
