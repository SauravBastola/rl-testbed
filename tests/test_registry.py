import tempfile, os
from pathlib import Path

def test_registry_crud(monkeypatch, tmp_path):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib
    import rltestbed.config as cfg
    import rltestbed.registry as reg
    importlib.reload(cfg)
    importlib.reload(reg)
    reg.init_db()
    rid = reg.create_run("test-run","dqn","cartpole",42,10,{"lr":0.001},{},tags=["a"])
    assert len(rid)==8
    r = reg.get_run(rid)
    assert r["agent"]=="dqn"
    reg.update_run(rid, status="running")
    assert reg.get_run(rid)["status"]=="running"
    runs = reg.list_runs()
    assert any(x["id"]==rid for x in runs)
    reg.delete_run(rid)
    assert reg.get_run(rid) is None

def test_model_keep(tmp_path, monkeypatch):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib, rltestbed.config as cfg, rltestbed.registry as reg
    importlib.reload(cfg); importlib.reload(reg); reg.init_db()
    rid = reg.create_run("mtest","dqn","cartpole",0,5,{}, {})
    # fake checkpoint
    run_dir = cfg.RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pth"
    ckpt.write_bytes(b"fake")
    reg.update_run(rid, best_eval_reward=10.0, final_mean_reward=5.0)
    m = reg.keep_model(rid, ckpt)
    assert Path(m["path"]).exists()
    assert m["checksum"] != ""
    assert reg.get_model(m["id"]) is not None
    reg.delete_model(m["id"])
    assert reg.get_model(m["id"]) is None
