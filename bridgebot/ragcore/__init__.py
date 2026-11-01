"""ragcore: a small, dependency-light retrieval core shared by BridgeBot and Rexetive.

Shared code: the same files live in BridgeBot (`bridgebot/ragcore/`) and Rexetive
(`backend/ragcore/`). They are copied, not installed, so neither repo depends on the other.
Every import inside this package is relative, which is what makes the copy work unchanged.
"""

import os as _os

# Quiet model-loading progress bars in CLI/demo output.
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Use the local Hugging Face cache only (scripts/download_models.py fetches the models once),
# so start-up never waits on the network and the demo works offline.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
