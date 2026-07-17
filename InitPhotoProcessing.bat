@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: InitPhotoProcessing.bat
:: ----------------------------------------------------------------------------
:: One-shot / re-runnable bootstrapper for BlurPictureDetector.
::
:: What it does. Designed to be downloaded and run on its own (no git clone,
:: no pre-opened "Anaconda Prompt" required) - it only needs Anaconda or
:: Miniconda to already be installed SOMEWHERE on the machine:
::   1. Looks for an existing Anaconda/Miniconda installation - first on PATH,
::      then in common per-user/per-machine install folders - so it works from
::      a plain Command Prompt, not just an "Anaconda Prompt". If none is
::      found anywhere, it prints an install URL and stops.
::   2. Creates an isolated conda environment (Python 3.12) for the project.
::   3. Detects an NVIDIA GPU/driver (nvidia-smi) and installs CUDA-enabled
::      PyTorch if found, otherwise CPU-only PyTorch.
::   4. Downloads the latest BlurPictureDetector source as a GitHub ZIP archive
::      (no git required) and updates the local app folder in place (existing
::      output\ and other local-only data are never deleted).
::   5. Installs dlib via conda-forge (prebuilt Windows binary - no C++
::      compiler / CMake needed) - GPU-accelerated build + cudnn when an
::      NVIDIA GPU was detected, otherwise the CPU build - then the rest of
::      requirements.txt via pip.
::   6. Installs face_recognition_models straight from source (PyPI releases
::      of it are unreliable and can leave face-recognition non-functional).
::   7. Writes a small RunPhotoProcessing.bat launcher.
::
:: Requires: Anaconda or Miniconda already installed somewhere on this
:: machine (does not need to be on PATH - common install folders are
:: searched automatically). If it isn't installed yet, get it from:
::   https://www.anaconda.com/download
::   https://docs.conda.io/en/latest/miniconda.html
:: then re-run this script (a plain Command Prompt/double-click is fine,
:: an "Anaconda Prompt" is not required).
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
:: [1/9] Find an existing Anaconda / Miniconda installation
:: ----------------------------------------------------------------------------
:: Checks PATH first (fast path when run from an "Anaconda Prompt"), then
:: falls back to scanning common per-user / per-machine install folders so
:: this also works from a plain Command Prompt where conda was never added
:: to PATH. Does not install anything itself - if nothing is found, it
:: prints the download URL and stops so the user can install once and
:: re-run this script.
echo [1/9] Checking for an existing Anaconda/Miniconda installation...
set "CONDA_DIR="

where conda >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%B in (`conda info --base 2^>nul`) do set "CONDA_DIR=%%B"
)

if not defined CONDA_DIR (
    echo       "conda" not found on PATH - checking common install locations...
    for %%D in (
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\anaconda3"
        "%LOCALAPPDATA%\miniconda3"
        "%LOCALAPPDATA%\anaconda3"
        "%LOCALAPPDATA%\Continuum\anaconda3"
        "%ProgramData%\miniconda3"
        "%ProgramData%\anaconda3"
        "C:\miniconda3"
        "C:\anaconda3"
    ) do (
        if not defined CONDA_DIR (
            if exist "%%~D\Scripts\conda.exe" set "CONDA_DIR=%%~D"
        )
    )
)

if not defined CONDA_DIR (
    echo ERROR: no Anaconda/Miniconda installation found ^(checked PATH and
    echo common install folders^).
    echo.
    echo This script requires Anaconda or Miniconda to be installed first.
    echo Install Miniconda ^(smaller, recommended^) or Anaconda, then re-run
    echo this script:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo   https://www.anaconda.com/download
    goto :fail
)

if not exist "%CONDA_DIR%\Scripts\conda.exe" (
    echo ERROR: found a possible conda installation at "%CONDA_DIR%" but
    echo "%CONDA_DIR%\Scripts\conda.exe" is missing.
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
    echo       NVIDIA driver found - will install CUDA-enabled PyTorch ^(cu126^) and GPU-accelerated dlib.
    set "TORCH_INDEX=--index-url https://download.pytorch.org/whl/cu126"
    set "HAS_NVIDIA_GPU=1"
) else (
    echo       No NVIDIA driver detected - will install CPU-only PyTorch and dlib.
    set "TORCH_INDEX="
    set "HAS_NVIDIA_GPU=0"
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
:: Prefer the GPU build when an NVIDIA GPU was detected in step 3. conda-forge
:: ships dlib-cpu/dlib-gpu as separate packages; unlike the plain "dlib"
:: metapackage (which only requires cudnn to BUILD the CUDA variant, not to
:: run it), we explicitly install "cudnn" alongside dlib-gpu here so
:: cudnn_ops64_9.dll etc. actually exist at runtime - otherwise dlib fails
:: with "Could not locate cudnn_ops64_9.dll" / "Invalid handle. Cannot load
:: symbol cudnnCreateTensorDescriptor" (PyTorch's own bundled cuDNN from its
:: pip wheel is invisible to dlib, a separate native library). Falls back to
:: the CPU build if the GPU install fails for any reason (e.g. no compatible
:: cudnn/CUDA version available for this driver).
::
:: Notes from hands-on testing:
:: - Remove any previously installed dlib/dlib-cpu/dlib-gpu first: solving
::   with an existing (different-variant) dlib already in the env produces
::   unsatisfiable conflicts across dozens of historical dlib-gpu/cuda
::   builds and can take several minutes before failing.
:: - "--override-channels -c conda-forge" avoids also pulling in the slower
::   "defaults" channel, which otherwise makes the solve far slower.
:: - Installing dlib-gpu can pull in a newer numpy (e.g. 2.x) as a conda
::   dependency, silently corrupting the pip-installed numpy<2 (torch 2.2.2
::   needs numpy<2 - see requirements.txt comment). Step 7 below re-asserts
::   the numpy<2 pin via pip --force-reinstall to fix this deterministically
::   regardless of which branch ran here.
"%CONDA_DIR%\Scripts\conda.exe" remove -y -p "%ENV_DIR%" dlib dlib-cpu dlib-gpu --force >nul 2>nul
if "%HAS_NVIDIA_GPU%"=="1" (
    echo [5/9] Installing dlib-gpu + cudnn ^(conda-forge, CUDA-accelerated^)...
    "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels dlib-gpu cudnn
    if errorlevel 1 (
        echo       dlib-gpu install failed - falling back to CPU-only dlib.
        "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels dlib-cpu
        if errorlevel 1 goto :fail
    )
) else (
    echo [5/9] Installing dlib-cpu ^(conda-forge prebuilt binary - no compiler required^)...
    "%CONDA_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels dlib-cpu
    if errorlevel 1 goto :fail
)

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

echo       Re-asserting numpy^<2 pin ^(dlib-gpu's conda-forge numpy dependency can bump it to 2.x, which breaks torch's Tensor.numpy^(^) bridge^)...
"%ENV_PY%" -m pip install --force-reinstall "numpy>=1.26.0,<2"
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
