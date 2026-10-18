"""Learning-path and glossary content served to the frontend.

Centralised here so educational copy is versioned with the code and never
duplicated between CLI, API and UI.
"""

# Both lists are derived from their richer sources so the v3 endpoints keep
# their shape. Edit rltestbed/curriculum.py and rltestbed/glossary.py instead.
from rltestbed.curriculum import legacy_path
from rltestbed.glossary import GLOSSARY as _GLOSSARY

LEARNING_PATH = legacy_path()

# v3 entries had {term, definition, algorithm}; the extra fields are additive.
GLOSSARY = [{**g, "algorithm": (g["algorithms"] or [None])[0]} for g in _GLOSSARY]
