"""Dataclass used to normalise provider SDK responses."""

from dataclasses import dataclass, field
from typing import Any

from .provider_type import ProviderType

__all__ = ["ModelResponse"]


@dataclass
class ModelResponse:
    """Portable representation of a provider completion."""

    content: str
    usage: dict[str, int] = field(default_factory=dict)
    model_name: str = ""
    friendly_name: str = ""
    provider: ProviderType = ProviderType.GOOGLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Report token usage to the in-flight tool call, if one is being tracked.

        Every provider funnels its completion through this dataclass, which
        makes it the single place utilisation can be captured without
        touching each provider. Recording is a no-op outside a tracked call.
        """
        try:
            from utils.metrics import record_model_usage

            provider_value = self.provider.value if hasattr(self.provider, "value") else str(self.provider)
            record_model_usage(self.model_name, provider_value, self.usage)
        except Exception:
            # Metrics must never interfere with a provider response.
            pass

    @property
    def total_tokens(self) -> int:
        """Return the total token count if the provider reported usage data."""

        return self.usage.get("total_tokens", 0)
