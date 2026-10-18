# Custom Environments

Drop Python files in this directory and restart the testbed — each file that
follows the contract below is auto-discovered and appears in the web app and CLI.

## Contract

```python
ENV_SPEC = {
    "name": "my_env",              # unique key (required)
    "label": "My Environment",     # display name
    "description": "What it tests",
    "url": "https://example.com/docs",
    "max_steps": 500,              # episode truncation cap
    "tags": ["my-tag"],
    "difficulty": "medium",        # easy | medium | hard | experimental
    "obs_type": "vector",          # vector | onehot | image
    "action_type": "discrete",     # discrete | continuous
    "lab": "foundations",          # optional: which lab shows it (default: foundations)
    "requires": [],                # optional extras, e.g. ["mujoco"]; missing => listed as unavailable
}

def make():
    return MyEnv()
```

`make()` must return an object with:

- `obs_dim` / `act_dim` attributes (int), **or** observations that can be probed
- `reset()` -> first observation
- `step(action)` -> `(obs, reward, terminated, truncated, info)`
- `close()` (optional)

Observations may be flat lists / numpy arrays, or dicts of the form
`{"obs": [...], "action_mask": [1, 0, 1, ...]}` for games with illegal moves.

See `example_env.py` for a complete working example.
