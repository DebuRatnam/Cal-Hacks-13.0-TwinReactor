"""Prediction backend the agent's tools call.

The agent and the web UI share one `backend.surrogate.CoreService`: the trained FNO surrogate
(checkpoints/fno_softplus.pt) on the v3 dataset core, plus the ndiffusion reference solver for
spot checks. Loaded once per process.
"""

from __future__ import annotations

from backend.surrogate import CoreService, LayoutError
from backend.surrogate import get_service as default_backend

__all__ = ["CoreService", "LayoutError", "default_backend"]
