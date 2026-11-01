"""ragcore: a small, dependency-light retrieval core shared by BridgeBot and Rexetive.

Shared code: the same files live in BridgeBot (`bridgebot/ragcore/`) and Rexetive
(`backend/ragcore/`). They are copied, not installed, so neither repo depends on the other.
Every import inside this package is relative, which is what makes the copy work unchanged.
"""
