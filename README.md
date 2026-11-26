# RAG Assessment

## Windows setup

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run

```powershell
python main.py task1
python main.py retrieval
python main.py reranker
python main.py prompts
python main.py evaluate
python main.py serve
```

Swagger: http://127.0.0.1:8000/docs

## API

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post -Headers @{"X-API-Key"="week7-demo-key"} -ContentType "application/json" -Body '{"question":"How many casual leaves per year?","k":4}'
```

## Tests

```powershell
pytest -q
```

The supplied policies are synthetic assessment documents. Replace them with approved company policies before production use. Evaluation metrics are calculated at runtime and are not pre-filled.


RAGAS evaluation uses one worker and a 600-second timeout to avoid local Ollama request timeouts. Set `RAGAS_LLM_MODEL` to a smaller installed Ollama model if needed.
