from rltestbed.capabilities import check_compat, validate_params

def _agent(supports=None, status="implemented"):
    return {"name":"dqn","label":"DQN","status":status,"supports": supports or {"obs_types":["vector","onehot"],"action_types":["discrete"],"multi_agent":False,"action_mask":True}}

def _env(obs_type="vector", action_type="discrete", n_agents=1, action_masked=False, category="gym"):
    return {"name":"cartpole","category":category,"obs_type":obs_type,"action_type":action_type,"n_agents":n_agents,"action_masked":action_masked,"tags":[]}

def test_compat_ok():
    assert check_compat(_agent(), _env())["compatible"] is True

def test_compat_blocks_continuous():
    c = check_compat(_agent(), _env(action_type="continuous"))
    assert not c["compatible"]

def test_compat_blocks_action_mask():
    c = check_compat(_agent(supports={"obs_types":["vector"],"action_types":["discrete"],"action_mask":False}), _env(action_masked=True))
    assert not c["compatible"]

def test_validate_params_ok():
    spec = {"params":[{"name":"lr","type":"float","default":0.001,"min":0,"max":1}]}
    assert validate_params(spec, {"lr":0.0005})["lr"]==0.0005

def test_validate_params_unknown():
    spec = {"params":[{"name":"lr","type":"float","default":0.001}]}
    try:
        validate_params(spec, {"bad":1})
        assert False
    except ValueError:
        pass
