def test_trainer_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib, rltestbed.config as cfg, rltestbed.registry as reg
    importlib.reload(cfg); importlib.reload(reg); reg.init_db()
    rid = reg.create_run("smoke","dqn","catch",0,5,{"lr":1e-3,"gamma":0.99,"hidden_size":32,"batch_size":8,"buffer_size":200,"target_update_freq":10,"epsilon_start":1.0,"epsilon_end":0.05,"epsilon_decay_steps":100,"lr_schedule":"none"}, {})
    run_dir = cfg.RUNS_DIR / rid
    from rltestbed.core.trainer import Trainer
    t = Trainer(run_id=rid, run_dir=run_dir, env_name="catch", agent_name="dqn", params={"lr":1e-3,"gamma":0.99,"hidden_size":32,"batch_size":8,"buffer_size":200,"target_update_freq":10,"epsilon_start":1.0,"epsilon_end":0.05,"epsilon_decay_steps":100,"lr_schedule":"none"}, seed=0, episodes=5, eval_freq=2, checkpoint_freq=10)
    summary = t.run()
    assert summary["episodes_done"]==5
    assert (run_dir/"history.jsonl").exists()
    assert (run_dir/"progress.json").exists()
