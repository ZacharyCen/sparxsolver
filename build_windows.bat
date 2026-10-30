@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo  AutoAnswer Windows x64 Build Script
echo ====================================================

:: 1. Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] .venv\Scripts\activate.bat not found. Using system python.
)

:: 2. Verify pyinstaller
where pyinstaller >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] PyInstaller not found. Please run: pip install pyinstaller
    exit /b 1
)

:: 3. Clean prior build artifacts
echo [1/3] Cleaning prior builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

:: 4. Build onedir package first for quick diagnosis
echo [2/3] Building onedir package for validation...
pyinstaller --noconfirm --onedir --windowed ^
    --name "AutoAnswer_Win_x64_dir" ^
    --add-data "action.schema.json;." ^
    --collect-all customtkinter ^
    --collect-all rapidocr_onnxruntime ^
    --collect-all onnxruntime ^
    main.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Onedir build failed!
    exit /b %ERRORLEVEL%
)

:: 5. Build final single-file EXE
echo [3/3] Building final single-file AutoAnswer_Win_x64.exe...
pyinstaller --noconfirm --onefile --windowed ^
    --name "AutoAnswer_Win_x64" ^
    --add-data "action.schema.json;." ^
    --collect-all customtkinter ^
    --collect-all rapidocr_onnxruntime ^
    --collect-all onnxruntime ^
    main.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Single-file build failed!
    exit /b %ERRORLEVEL%
)

echo ====================================================
if exist "OpenAI-KEY.txt" copy /Y "OpenAI-KEY.txt" "dist\OpenAI-KEY.txt" >nul
if exist "OpenAI-KEY.txt" copy /Y "OpenAI-KEY.txt" "dist\AutoAnswer_Win_x64_dir\OpenAI-KEY.txt" >nul
echo  Build Completed Successfully!
echo  Executable: dist\AutoAnswer_Win_x64.exe
echo ====================================================
pause
