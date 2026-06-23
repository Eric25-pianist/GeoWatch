@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .[dev]
geowatch validate configs\default.yaml --strict-deps
pytest
endlocal
