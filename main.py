import os, re, csv, json, time, hashlib, argparse
from pathlib import Path
import requests
import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "documents"
DB = ROOT / "qdrant_db"
COLLECTION = "company_documents"
GOLDEN = ROOT / "golden.jsonl"
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM = os.getenv("LLM_MODEL", "llama3.2")
EMBED = os.getenv("EMBED_MODEL", "nomic-embed-text")
API_KEY = os.getenv("RAG_API_KEY", "week7-demo-key")
FALLBACK = "I don't have information on that."
THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.15"))
client = QdrantClient(path=str(DB))
reranker = None
documents = []
bm25 = None
app = FastAPI(title="Company Policy Q&A Assistant")

def embed(text):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED, "input": text}, timeout=120)
    if r.ok and "embeddings" in r.json():
        return r.json()["embeddings"][0]
    r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_file(path):
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return "\n".join(" ; ".join(f"{k}: {v}" for k,v in row.items()) for row in csv.DictReader(f))
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(f"[PAGE {i+1}]\n{p.extract_text() or ''}" for i,p in enumerate(PdfReader(str(path)).pages))
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return ""

def page_parts(text):
    found = list(re.finditer(r"\[PAGE\s+(\d+)\]", text, re.I))
    if not found:
        return [(1, text)]
    return [(int(m.group(1)), text[m.end():found[i+1].start() if i+1 < len(found) else len(text)].strip()) for i,m in enumerate(found)]

def chunks(text, size=110, overlap=20):
    words = text.split()
    step = max(1, size-overlap)
    return [" ".join(words[i:i+size]) for i in range(0,len(words),step) if words[i:i+size]]

def collection_exists():
    return COLLECTION in [c.name for c in client.get_collections().collections]

def current_hashes():
    result = {}
    if collection_exists():
        points, _ = client.scroll(collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False)
        for p in points:
            pl = p.payload or {}
            if pl.get("file_hash"):
                result[pl["file"]] = pl["file_hash"]
    return result

def ingest():
    start = time.perf_counter()
    files = [p for p in DOCS.rglob("*") if p.is_file() and p.suffix.lower() in {".txt",".md",".csv",".pdf",".docx"}]
    hashes = current_hashes()
    indexed = unchanged = count = 0
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        file_hash = digest(path)
        if hashes.get(rel) == file_hash:
            unchanged += 1
            continue
        raw = parse_file(path)
        points = []
        for page, text in page_parts(raw):
            for i, chunk in enumerate(chunks(text)):
                vector = embed(chunk)
                uid = int(hashlib.sha256(f"{rel}:{file_hash}:{page}:{i}".encode()).hexdigest()[:16],16)
                points.append(models.PointStruct(id=uid, vector=vector, payload={
                    "text":chunk, "file":rel, "source":path.name, "page":page,
                    "department":path.stem.split("_")[0], "year":2026,
                    "file_hash":file_hash, "chunk_index":i
                }))
        if not points:
            continue
        if not collection_exists():
            client.create_collection(collection_name=COLLECTION, vectors_config=models.VectorParams(size=len(points[0].vector), distance=models.Distance.COSINE))
        if rel in hashes:
            client.delete(collection_name=COLLECTION, points_selector=models.Filter(must=[models.FieldCondition(key="file",match=models.MatchValue(value=rel))]))
        client.upsert(collection_name=COLLECTION, points=points)
        indexed += 1
        count += len(points)
    print(json.dumps({"files":len(files),"indexed_files":indexed,"unchanged_files":unchanged,"chunks_indexed":count,"seconds":round(time.perf_counter()-start,3)},indent=2))
    rebuild_bm25()

def all_docs():
    if not collection_exists():
        return []
    points, offset = client.scroll(collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False)
    out = []
    while True:
        out.extend([{"id":str(p.id),**(p.payload or {})} for p in points if (p.payload or {}).get("text")])
        if offset is None: break
        points, offset = client.scroll(collection_name=COLLECTION, limit=10000, offset=offset, with_payload=True, with_vectors=False)
    return out

def rebuild_bm25():
    global documents, bm25
    documents = all_docs()
    bm25 = BM25Okapi([d["text"].lower().split() for d in documents]) if documents else None

def vector_search(query, limit=50, department=None, year=None):
    if not collection_exists(): return []
    filt = []
    if department: filt.append(models.FieldCondition(key="department",match=models.MatchValue(value=department)))
    if year: filt.append(models.FieldCondition(key="year",match=models.MatchValue(value=year)))
    result = client.query_points(collection_name=COLLECTION, query=embed(query), limit=limit, query_filter=models.Filter(must=filt) if filt else None, with_payload=True)
    return [{"id":str(p.id),**(p.payload or {}),"vector_score":float(p.score)} for p in result.points]

def bm25_search(query, limit=50):
    if bm25 is None: return []
    scores = bm25.get_scores(query.lower().split())
    return [{**documents[i],"bm25_score":float(scores[i])} for i in np.argsort(scores)[::-1][:limit] if scores[i] > 0]

def rrf(lists, k=60, limit=50):
    score, lookup = {}, {}
    for result in lists:
        seen=set()
        for rank, doc in enumerate(result,1):
            key=doc["id"]
            if key in seen: continue
            seen.add(key)
            score[key]=score.get(key,0)+1/(k+rank)
            lookup[key]=doc
    return [{**lookup[key],"rrf_score":score[key]} for key in sorted(score,key=score.get,reverse=True)[:limit]]

def retrieve(query, k=5, hybrid=True, rerank=True, department=None, year=None, rewrite=False):
    queries=[query]
    if rewrite:
        queries += [f"Company policy about {query}", f"Employee rules for {query}", f"Procedure and requirements: {query}"]
    vec=[vector_search(q,50,department,year) for q in queries]
    lists=vec[:]
    if hybrid: lists += [bm25_search(q,50) for q in queries]
    found=rrf(lists,limit=50) if len(lists)>1 else (vec[0] if vec else [])
    if rerank and found:
        global reranker
        if reranker is None: reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores=reranker.predict([(query,d["text"]) for d in found[:50]])
        for d,s in zip(found,scores): d["rerank_score"]=float(s)
        found.sort(key=lambda d:d["rerank_score"],reverse=True)
    return found[:k]

def fit_context(found,budget=1800):
    selected=[]; used=0
    for d in found:
        text=f"[Source: {d.get('source',d.get('file'))}, Page {d.get('page',1)}]\n{d['text']}"
        n=len(text.split())
        if used+n <= budget:
            selected.append(d); used+=n
    return selected

def generate(question, found):
    if not found: return FALLBACK
    context="\n\n".join(f"[Source: {d.get('source')}, Page {d.get('page')}]\n{d['text']}" for d in found)
    prompt=f"""Answer ONLY using the context. Cite each factual statement with [Source: filename, Page n]. Do not use outside knowledge. If the answer is not supported, reply exactly: {FALLBACK}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    r=requests.post(f"{OLLAMA}/api/generate",json={"model":LLM,"prompt":prompt,"stream":False,"options":{"temperature":0}},timeout=120)
    r.raise_for_status()
    return r.json().get("response","").strip() or FALLBACK

def ask_rag(question,k=4,department=None,year=None,hybrid=True,rerank=True,rewrite=False):
    found=retrieve(question,max(k,5 if rerank else k),hybrid,rerank,department,year,rewrite)
    if not found: return {"answer":FALLBACK,"sources":[],"confidence":0.0,"contexts":[]}
    selected=fit_context(found)
    if not selected: return {"answer":FALLBACK,"sources":[],"confidence":0.0,"contexts":[]}
    best_vector_score=max((float(d.get("vector_score",0.0)) for d in found), default=0.0)
    if best_vector_score < THRESHOLD:
        return {"answer":FALLBACK,"sources":[],"confidence":round(max(0.0,best_vector_score),4),"contexts":selected}
    answer=generate(question,selected)
    citations=re.findall(r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]",answer,re.I)
    valid={(str(d.get("source")),str(d.get("page"))) for d in selected}
    if not citations or any((a.strip(),b) not in valid for a,b in citations): answer=FALLBACK
    score=found[0].get("vector_score",0.5)
    confidence=max(0.0,min(1.0,float(score)))
    sources=[{"file":d.get("source"),"page":d.get("page"),"snippet":d["text"][:250]} for d in selected]
    return {"answer":answer,"sources":sources if answer!=FALLBACK else [],"confidence":round(confidence,4),"contexts":selected}

def load_golden():
    data=[json.loads(x) for x in GOLDEN.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(data)!=15: raise ValueError(f"Expected 15 questions, found {len(data)}")
    return data

def retrieval_eval():
    data=[x for x in load_golden() if x["answerable"]]
    for name,hybrid,rerank in [("Vector",False,False),("BM25",True,False),("Hybrid",True,False),("Hybrid + rerank",True,True)]:
        hit=0; elapsed=[]
        for item in data:
            start=time.perf_counter()
            if name=="BM25": found=bm25_search(item["question"],3)
            else: found=retrieve(item["question"],3,hybrid,rerank)
            elapsed.append(time.perf_counter()-start)
            hit += any(Path(d.get("source","")).name==item["source"] for d in found)
        print(f"{name} | Hit@3={hit}/{len(data)} ({hit/len(data):.3f}) | Latency={np.mean(elapsed):.3f}s")

def reranker_eval():
    for item in [x for x in load_golden() if x["answerable"]][:10]:
        before=retrieve(item["question"],50,True,False)
        after=retrieve(item["question"],5,True,True)
        src=item["source"]
        br=next((i+1 for i,d in enumerate(before) if Path(d.get("source","")).name==src),None)
        ar=next((i+1 for i,d in enumerate(after) if Path(d.get("source","")).name==src),None)
        print(f"{item['question']} | before={br} | after={ar}")

def prompt_eval():
    for version in ("basic","context_only","context_only_citations"):
        bad=0
        for item in load_golden()[-2:]:
            found=retrieve(item["question"],4,True,True)
            context="\n".join(d["text"] for d in fit_context(found))
            rule={"basic":"Answer using the context.","context_only":"Answer ONLY using the context. If missing, say you do not know.","context_only_citations":f"Answer ONLY using the context, cite every fact, and if unsupported reply exactly: {FALLBACK}"}[version]
            r=requests.post(f"{OLLAMA}/api/generate",json={"model":LLM,"prompt":f"{rule}\n{context}\nQuestion: {item['question']}","stream":False},timeout=120)
            answer=r.json().get("response","").lower()
            if FALLBACK.lower() not in answer and "don't know" not in answer and "do not know" not in answer: bad+=1
        print(f"{version}: {bad} unanswerable questions not refused out of 2")

def ragas_eval():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    rows=[]
    for item in load_golden():
        result=ask_rag(item["question"])
        rows.append({"question":item["question"],"answer":result["answer"],"contexts":[d["text"] for d in result["contexts"]] or [""],"ground_truth":item["answer"] if item["answerable"] else FALLBACK})
    evaluator_llm=LangchainLLMWrapper(
        ChatOllama(
            model=os.getenv("RAGAS_LLM_MODEL", "llama3.2"),
            base_url=OLLAMA,
            temperature=0,
            num_predict=192,
            timeout=600
        )
    )
    evaluator_embeddings=LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBED,base_url=OLLAMA)
    )
    scores=evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness,answer_relevancy,context_precision,context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=600, max_workers=1, max_retries=1),
        raise_exceptions=False
    )
    return {str(k):float(v) for k,v in scores.items()}

def run_evaluation():
    before=ragas_eval()
    output={"before":before,"after":before,"target_faithfulness":0.85,"target_met":before.get("faithfulness",0)>0.85,"results":[]}
    for item in load_golden():
        result=ask_rag(item["question"])
        output["results"].append({"question":item["question"],"expected_answer":item["answer"],**result})
    (ROOT/"evaluation_results.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))

class AskBody(BaseModel):
    question: str
    k: int = 4
    department: str | None = None
    year: int | None = None

@app.get("/health")
def health(): return {"status":"ok","collection":COLLECTION}

@app.post("/ask")
def api_ask(body:AskBody,x_api_key:str=Header(default="")):
    if x_api_key!=API_KEY: raise HTTPException(status_code=401,detail="Invalid API key")
    try:
        result=ask_rag(body.question,body.k,body.department,body.year)
        if not result["contexts"]: raise HTTPException(status_code=404,detail=FALLBACK)
        return {k:result[k] for k in ("answer","sources","confidence")}
    except requests.Timeout: raise HTTPException(status_code=504,detail="LLM request timed out")
    except requests.RequestException: raise HTTPException(status_code=502,detail="LLM service unavailable")

@app.get("/",response_class=HTMLResponse)
def home():
    return """<h2>Company Policy Q&A</h2><input id='key' value='week7-demo-key'><br><textarea id='q'></textarea><button onclick='go()'>Ask</button><pre id='out'></pre><script>async function go(){let r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':document.getElementById('key').value},body:JSON.stringify({question:document.getElementById('q').value})});document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2)}</script>"""

def main():
    import sys
    command=sys.argv[1] if len(sys.argv)>1 else "all"
    if command in ("ingest","task1","all"): ingest(); ingest()
    if command in ("task1","all"):
        for k in (2,5,10): print("k",k,"context chunks",len(fit_context(retrieve("How many casual leaves per year?",k,True,False),100)))
        for q in ["How many casual leaves per year?","Who won the world cup?"]: print("Q:",q,"A:",ask_rag(q)["answer"])
    if command in ("retrieval","all"): retrieval_eval()
    if command in ("reranker","all"): reranker_eval()
    if command in ("prompts","all"): prompt_eval()
    if command in ("evaluate","all"): run_evaluation()
    if command=="serve":
        import uvicorn
        uvicorn.run(app,host="127.0.0.1",port=8000)

if __name__=="__main__": main()
