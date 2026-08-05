$python = Join-Path $PSScriptRoot '..\.venv-ai\Scripts\python.exe'
& $python -m uvicorn app:app --host 127.0.0.1 --port 8000
