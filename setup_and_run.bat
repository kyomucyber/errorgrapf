@echo off
setlocal

REM ====== Détection Python (priorité 3.12) ======
where py >nul 2>&1
if %errorlevel%==0 (
  py -3.12 -c "import sys;print(sys.version)" >nul 2>&1 && set PYCMD=py -3.12
  if not defined PYCMD ( py -3 -c "import sys;print(sys.version)" >nul 2>&1 && set PYCMD=py -3 )
) else (
  where python >nul 2>&1 || (echo Python introuvable. Installe Python 3.12+ puis relance.& exit /b 1)
  python -c "import sys;exit(0 if sys.version_info[:2]>=(3,12) else 1)" && set PYCMD=python
)
if not defined PYCMD (
  echo Python 3.12+ non trouve. Installe Python 3.12+ puis relance.
  exit /b 1
)

echo Using: %PYCMD%

REM ====== Venv ======
set VENV=.venv
if not exist %VENV% (
  %PYCMD% -m venv %VENV% || (echo Echec creation venv & exit /b 1)
)

call %VENV%\Scripts\activate.bat || (echo Echec activation venv & exit /b 1)

REM ====== Pip & outils ======
python -m pip install -U pip setuptools wheel

REM ====== Requirements par defaut si absent ======
if not exist requirements.txt (
  echo numpy>=2.1^,<3>requirements.txt
  echo pandas>=2.2^,<3>>requirements.txt
  echo scipy>=1.12^,<2>>requirements.txt
  echo scikit-image>=0.23^,<1>>requirements.txt
  echo opencv-python>=4.9^,<5>>requirements.txt
  echo matplotlib>=3.8^,<4>>requirements.txt
  echo Pillow>=10^,<11>>requirements.txt
  echo pytesseract>=0.3.10^,<0.4>>requirements.txt
)

REM ====== Installation deps ======
python -m pip install -r requirements.txt || (echo Echec installation deps & exit /b 1)

REM ====== Lancement ======
if exist Graph_ui.py (
  python Graph_ui.py
) else (
  echo Graph_ui.py introuvable dans %cd%
)

endlocal
