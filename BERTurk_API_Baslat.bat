@echo off
cd /d "%~dp0"
echo BERTurk API baslatiliyor...
py -m uvicorn api_berturk:app --reload --port 8000
pause