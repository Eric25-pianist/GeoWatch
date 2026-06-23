@echo off
setlocal
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
geowatch %*
endlocal
