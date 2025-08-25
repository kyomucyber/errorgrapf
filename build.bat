@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem === Nettoyage de l'env global qui pollue Python ===
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
set "MPLBACKEND=Agg"

set "APP_NAME=ErrorGraph"
set "ENTRY=Graph_ui.py"
set "ICON=ressources\app.ico"

rem === Venv ===
if not exist ".venv\" py -3.11 -m venv .venv || exit /b 1
call ".venv\Scripts\activate.bat" || exit /b 1

echo [*] MAJ pip/setuptools/wheel...
python -m pip install -U pip wheel setuptools || goto :fail

echo [*] Installation deps...
pip install -r requirements.txt || goto :fail
pip install "pyinstaller>=6.3" || goto :fail

echo [*] (Re)installation propre de NumPy...
pip uninstall -y numpy >nul 2>&1
pip cache purge >nul 2>&1
pip install --no-cache-dir --only-binary=:all: "numpy==2.2.1" || goto :fail

rem === Preflight: test sys.path et numpy ===
set "_TMPPY=%TEMP%\preflight_graph_%RANDOM%.py"
> "%_TMPPY%" (
  echo import sys, platform
  echo bad = [p for p in sys.path if 'opal-rt' in p.lower()]
  echo import numpy; print("Python:", platform.python_version()); print("NumPy:", numpy.__version__)
  echo import sys; sys.exit(0 if not bad else 1)
)
python "%_TMPPY%"
set "_RC=%ERRORLEVEL%"
del "%_TMPPY%" >nul 2>&1
if not "%_RC%"=="0" (
  echo [!] Traces d'OPAL-RT detectees, on isole PATH pendant la build...
)
rem On isole de toute facon le PATH pour la compilation
set "PATH=%CD%\.venv\Scripts;%SystemRoot%\System32;%SystemRoot%"

for %%D in ("build" "dist") do if exist "%%~D" rmdir /s /q "%%~D"

echo [*] Compilation PyInstaller...
python -S -m PyInstaller --noconfirm --clean ^
  --name "%APP_NAME%" ^
  --noconsole ^
  --icon "%ICON%" ^
  --add-data "ressources;ressources" ^
  --add-data "MySettings;MySettings" ^
  --add-data "MyData;MyData" ^
  --collect-all numpy ^
  --collect-all matplotlib ^
  --collect-all skimage ^
  --collect-all cv2 ^
  "%ENTRY%" || goto :fail

echo.
echo [OK] Exe cree : "%CD%\dist\%APP_NAME%\%APP_NAME%.exe"
exit /b 0

:fail
echo.
echo [X] Erreur : la compilation a echoue.
exit /b 1
