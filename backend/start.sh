#/bin/bash
source ./.venv/bin/activate
uv pip install -r requirements.txt
uv run uvicorn server:app --reload