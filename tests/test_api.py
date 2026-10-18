from fastapi.testclient import TestClient

def test_health():
    from rltestbed.server.app import create_app
    c = TestClient(create_app())
    r = c.get("/api/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"

def test_envs_and_agents():
    from rltestbed.server.app import create_app
    c = TestClient(create_app())
    assert len(c.get("/api/envs").json())>0
    assert len(c.get("/api/agents").json())>0

def test_create_run_and_compare(tmp_path, monkeypatch):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib, rltestbed.config as cfg, rltestbed.registry as reg
    importlib.reload(cfg); importlib.reload(reg); reg.init_db()
    from rltestbed.server.app import create_app
    c = TestClient(create_app())
    # create two tiny runs via registry then compare via API
    rid1 = reg.create_run("r1","dqn","catch",0,5,{}, {})
    rid2 = reg.create_run("r2","dqn","catch",0,5,{}, {})
    # write minimal history
    import json
    for rid in [rid1, rid2]:
        p = cfg.RUNS_DIR / rid / "history.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p,"w") as f:
            for i in range(3):
                f.write(json.dumps({"episode":i+1,"reward":i,"mean_reward":i})+"\n")
    r = c.get(f"/api/compare?run_ids={rid1},{rid2}")
    assert r.status_code==200
    assert len(r.json()["runs"])==2

def test_preview():
    from rltestbed.server.app import create_app
    c = TestClient(create_app())
    r = c.get("/api/envs/catch/preview")
    assert r.status_code==200
    assert "trajectory" in r.json()
