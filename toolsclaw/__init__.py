"""toolsclaw — Ultra-lightweight tool-calling agent framework."""

__version__ = "0.1.0"

from toolsclaw.hook import AgentHook, CompositeHook, MemoryCompressionHook, PersistentMemoryHook, SDKCaptureHook
from toolsclaw.memory import CompressionStrategy, MemoryCompressor
from toolsclaw.persistent_memory import Memory, MemoryStore
from toolsclaw.sdk import RunResult, ToolsClaw

__all__ = [
    "ToolsClaw",
    "RunResult",
    "AgentHook",
    "CompositeHook",
    "MemoryCompressionHook",
    "PersistentMemoryHook",
    "SDKCaptureHook",
    "MemoryCompressor",
    "CompressionStrategy",
    "Memory",
    "MemoryStore",
]
