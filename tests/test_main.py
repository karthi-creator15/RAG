from fastapi.testclient import TestClient
import main

def test_golden_count():
    assert len(main.load_golden()) == 15

def test_rrf():
    a=[{"id":"a","text":"A"},{"id":"b","text":"B"}]
    b=[{"id":"b","text":"B"},{"id":"a","text":"A"}]
    result=main.rrf([a,b],limit=2)
    assert len(result)==2
    assert result[0]["id"] in {"a","b"}

def test_chunking():
    assert main.chunks("one two three four five",size=3,overlap=1)

def test_context_budget():
    docs=[{"source":"a.txt","page":1,"text":"one two three"}]
    assert len(main.fit_context(docs,100))==1
    assert len(main.fit_context(docs,1))==0

def test_health():
    client=TestClient(main.app)
    assert client.get("/health").status_code==200

def test_api_key():
    client=TestClient(main.app)
    assert client.post("/ask",json={"question":"hello"}).status_code==401
