@echo off
REM BridgeBot v2 demo: builds what is missing, then starts the API (:8000) and the UI (:7860).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating venv...
  py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\python -m pip install -r requirements.txt
)
set PY=.venv\Scripts\python.exe
if not exist .env copy .env.example .env >nul
if not exist data\corpus\etiquette-in-japan.md %PY% scripts\build_corpus.py
if not exist data\calendar.db %PY% scripts\build_calendar.py
%PY% -c "import os; os.environ['HF_HUB_OFFLINE']='1'; from sentence_transformers import SentenceTransformer as S, CrossEncoder as C; S('BAAI/bge-small-en-v1.5'); C('cross-encoder/ms-marco-MiniLM-L-6-v2')" 2>nul || %PY% scripts\download_models.py
echo Building / loading the index (first run takes about a minute)...
%PY% -c "from bridgebot.corpus import get_indexes; ix=get_indexes(); print(len(ix.chunks), 'chunks; rebuilt =', ix.rebuilt)"
start "BridgeBot API :8000" %PY% -m uvicorn bridgebot.api:app --port 8000
start "BridgeBot UI :7860" %PY% app.py
echo API docs: http://localhost:8000/docs
echo UI:       http://localhost:7860
