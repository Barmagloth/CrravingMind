"""Agent interface and abstract LLM provider."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM backend providers."""

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        # TODO: implement
        ...


class AgentInterface:
    """High-level interface for agent–LLM interaction."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
