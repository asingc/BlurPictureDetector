@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: InitPhotoProcessing.bat
:: ----------------------------------------------------------------------------
:: One-shot / re-runnable bootstrapper for BlurPictureDetector.
::
:: What it does, assuming Anaconda or Miniconda is already installed and
:: "conda" is available on PATH (e.g. run from an "Anaconda Prompt"):
::   1. Verifies conda is installed and locates its base installation directory.
::   2. Creates an isolated conda environment (Python 3.12) for the project.
::   3. Detects an NVIDIA GPU/driver (nvidia-smi) and installs CUDA-enabled
::      PyTorch if found, otherwise CPU-only PyTorch.
::   4. Downloads the latest BlurPictureDetector source as a GitHub ZIP archive
::      (no git required) and updates the local app folder in place (existing
::      output\ and other local-only data are never deleted).
::   5. Installs dlib via conda-forge (prebuilt Windows binary - no C++
::      compiler / CMake needed), then the rest of requirements.txt via pip.
::   6. Installs face_recognition_models straight from source (PyPI releases
::      of it are unreliable and can leave face-recognition non-functional).
::   7. Writes a small RunPhotoProcessing.bat launcher.
::
:: Requires: Anaconda or Miniconda already installed, with "conda" on PATH
:: (run this from an "Anaconda Prompt", or a shell where conda has been
:: added to PATH). Get it from:
::   https://www.anaconda.com/download
::   https://docs.conda.io/en/latest/miniconda.html
::
:: Safe to re-run at any time: it re-checks/updates the environment,
:: dependencies, and pulls the latest app code.
::
:: Usage:
::   InitPhotoProcessing.bat [install_root]
::     install_root  optional override for the directory the app source,
::                   conda environment, and launcher are placed in (default:
::                   the current directory, i.e. the folder this script is
::                   run from). The conda environment is created here rather
::                   than under the Anaconda/Miniconda base install so it
::                   works even when conda was installed system-wide (e.g.
::                   under C:\ProgramData) where a non-admin user cannot
::                   create new environments.
:: ============================================================================

set "REPO_OWNER=asingc"
set "REPO_NAME=BlurPictureDetector"
set "REPO_BRANCH=main"
set "ENV_NAME=blurpicturedetector"
set "PYTHON_VERSION=3.12"

set "INSTALL_ROOT=%CD%"
if not "%~1"=="" set "INSTALL_ROOT=%~1"

echo ============================================================
echo  BlurPictureDetector - environment setup
echo  Install root: %INSTALL_ROOT%
echo ============================================================
echo.

if not exist "%INSTALL_ROOT%" mkdir "%INSTALL_ROOT%"

set "APP_DIR=%INSTALL_ROOT%\app"
set "WORK_DIR=%INSTALL_ROOT%\_setup_tmp"

if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"

:: ----------------------------------------------------------------------------
:: [1/9] Verify Anaconda / Miniconda is already installed
:: ----------------------------------------------------------------------------
echo [1/9] Checking for an existing Anaconda/Miniconda installation...
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: no "conda" command found on PATH.
    echo.
    echo This script requires Anaconda or Miniconda to already be installed,
    echo with "conda" available on PATH ^(e.g. run this from an "Anaconda
    echo Prompt", or open a regular prompt after adding conda to PATH^).
    echo.
    echo Get Anaconda/Miniconda from:
    echo   https://www.anaconda.com/download
    echo   https://docs.conda.io/en/latest/miniconda.html
    goto :fail
)

for /f "usebackq delims=" %%B in (`conda info --base`) do set "CONDA_DIR=%%B"
if not exist "%CONDA_DIR%\Scripts\conda.exe" (
    echo ERROR: found "conda" on PATH but could not determine its base
    echo installation directory ^(via "conda info --base"^).
    goto :fail
)
echo       Found conda at "%CONDA_DIR%".

:: Create the environment under install_root (not under CONDA_DIR\envs) so
:: this works without admin rights even when conda itself was installed
:: system-wide (e.g. C:\ProgramData\anaconda3), which a non-admin user
:: cannot write into.
set "ENV_DIR=%INSTALL_ROOT%\envs\%ENV_NAME%"
set "ENV_PY=%ENV_DIR%\python.exe"

:: ----------------------------------------------------------------------------
:: [2/9] Conda environment
:: ----------------------------------------------------------------------------
echo       Accepting Anaconda Terms of Service for default channels (best effort)...
"%CONDA_DIR%\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
"%CONDA_DIR%\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul

if exist "%ENV_PY%" (
    echo [2/9] Conda environment "%ENV_NAME%" already present - skipping creation.
) else (
    echo [2/9] Creating conda environment "%ENV_NAME%" ^(python %PYTHON_VERSION%^)...
    "%CONDA_DIR%\Scripts\conda.exe" create -y -p "%ENV_DIR%" python=%PYTHON_VERSION%
    if errorlevel 1 goto :fail
)

:: ----------------------------------------------------------------------------
:: [3/9] Detect CUDA / NVIDIA GPU
:: ----------------------------------------------------------------------------
echo [3/9] Detecting NVIDIA GPU / CUDA driver...
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo       NVIDIA driver found - will install CUDA-enabled PyTorch ^(cu126^).
    set "TORCH_INDEX=--index-url https://download.pytorch.org/whl/cu126"
) else (
    echo       No NVIDIA driver detected - will install CPU-only PyTorch.
    set "TORCH_INDEX="
)

:: ----------------------------------------------------------------------------
:: [4/9] Download latest app code (no git required)
:: ----------------------------------------------------------------------------
echo [4/9] Downloading latest source from GitHub ^(%REPO_OWNER%/%REPO_NAME%@%REPO_BRANCH%^)...
set "APP_ZIP=%WORK_DIR%\source.zip"
if exist "%APP_ZIP%" del /f /q "%APP_ZIP%"
call :download "https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%REPO_BRANCH%.zip" "%APP_ZIP%"
if not exist "%APP_ZIP%" (
    echo ERROR: failed to download the application source.
    goto :fail
)

set "EXTRACT_DIR=%WORK_DIR%\extracted"
if exist "%EXTRACT_DIR%" rmdir /s /q "%EXTRACT_DIR%"
mkdir "%EXTRACT_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%APP_ZIP%' -DestinationPath '%EXTRACT_DIR%' -Force"
if errorlevel 1 (
    echo ERROR: failed to extract the application source.
    goto :fail
)

set "EXTRACTED_SRC=%EXTRACT_DIR%\%REPO_NAME%-%REPO_BRANCH%"
if not exist "%EXTRACTED_SRC%" (
    echo ERROR: unexpected archive layout - "%EXTRACTED_SRC%" not found.
    goto :fail
)

if not exist "%APP_DIR%" mkdir "%APP_DIR%"
echo       Updating "%APP_DIR%" ^(existing output\ and local data are preserved^)...
robocopy "%EXTRACTED_SRC%" "%APP_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 (
    echo ERROR: robocopy failed to update the application folder.
    goto :fail
)

:: ----------------------------------------------------------------------------
:: [5/9] Hard-to-build native dependency: dlib (via conda-forge, prebuilt)
:: ----------------------------------------------------------------------------
echo [5/9] Installing dlib ^(conda-forge prebuilt binary - no compiler required^)...
"%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge dlib
if errorlevel 1 goto :fail

:: ----------------------------------------------------------------------------
:: [6/9] PyTorch
:: ----------------------------------------------------------------------------
echo [6/9] Installing PyTorch...
"%ENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%ENV_PY%" -m pip install torch torchvision !TORCH_INDEX!
if errorlevel 1 goto :fail

:: ----------------------------------------------------------------------------
:: [7/9] Remaining Python dependencies (dlib already handled by conda above)
:: ----------------------------------------------------------------------------
echo [7/9] Installing remaining dependencies from requirements.txt...
set "REQ_FILTERED=%WORK_DIR%\requirements.filtered.txt"
findstr /V /R "^dlib" "%APP_DIR%\requirements.txt" > "%REQ_FILTERED%"
"%ENV_PY%" -m pip install -r "%REQ_FILTERED%"
if errorlevel 1 goto :fail

:: ----------------------------------------------------------------------------
:: [8/9] face_recognition_models (PyPI releases of this are unreliable / can
:: silently fail to register; installing straight from the source repo is
:: the fix the upstream face_recognition project itself recommends). Without
:: this, the dlib face-recognition provider fails at runtime with
:: "Please install `face_recognition_models`..." even though face-recognition
:: itself installed fine above.
::
:: face_recognition_models' __init__.py still imports the old `pkg_resources`
:: API (from setuptools). Recent setuptools versions (81+) dropped
:: pkg_resources, which makes that import raise ModuleNotFoundError - caught
:: by face_recognition's own broad except clause and re-reported as the same
:: misleading "please install face_recognition_models" message even though
:: the package IS installed. Pin setuptools<81 to keep pkg_resources available.
:: ----------------------------------------------------------------------------
echo [8/9] Pinning setuptools ^(pkg_resources compatibility for face_recognition_models^)...
"%ENV_PY%" -m pip install "setuptools<81"
if errorlevel 1 goto :fail

echo       Installing face_recognition_models ^(from source - PyPI releases are unreliable^)...
"%ENV_PY%" -m pip install --upgrade --force-reinstall --no-deps git+https://github.com/ageitgey/face_recognition_models
if errorlevel 1 goto :fail

echo       Verifying face_recognition_models imports correctly...
"%ENV_PY%" -c "import face_recognition_models" 2>nul
if errorlevel 1 (
    echo ERROR: face_recognition_models installed but failed to import.
    goto :fail
)

:: ----------------------------------------------------------------------------
:: [9/9] Convenience launcher
:: ----------------------------------------------------------------------------
echo [9/9] Writing launcher...
(
    echo @echo off
    echo "%ENV_PY%" "%APP_DIR%\1_prep_review.py" %%*
) > "%INSTALL_ROOT%\RunPhotoProcessing.bat"

echo.
echo ============================================================
echo  Setup complete.
echo.
echo  App folder:   %APP_DIR%
echo  Python env:   %ENV_DIR%
echo.
echo  To process a folder of photos, run:
echo    "%INSTALL_ROOT%\RunPhotoProcessing.bat" "C:\path\to\photos"
echo.
echo  Re-run this script any time to update everything to the latest.
echo ============================================================
pause
goto :eof

:: ----------------------------------------------------------------------------
:: :download <url> <output_path>  -- curl if available, else PowerShell.
:: ----------------------------------------------------------------------------
:download
where curl >nul 2>nul
if not errorlevel 1 (
    curl -L --fail --silent --show-error -o %2 %1
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri %1 -OutFile %2"
)
goto :eof

:fail
echo.
echo Setup FAILED. See the messages above for details.
pause
exit /b 1
