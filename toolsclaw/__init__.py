"""toolsclaw — Ultra-lightweight tool-calling agent framework."""

__version__ = "0.1.0"

from toolsclaw.hook import AgentHook, CompositeHook, SDKCaptureHook
from toolsclaw.sdk import RunResult, ToolsClaw

__all__ = [
    "ToolsClaw",
    "RunResult",
    "AgentHook",
    "CompositeHook",
    "SDKCaptureHook",
]
