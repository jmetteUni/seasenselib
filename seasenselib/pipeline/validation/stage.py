"""
Validation stage.

Runs validators and records results in context metadata.
"""

from __future__ import annotations

from typing import Dict, Any

from ..base import Stage, StageContext
from .handlers.validation_runner import ValidationRunner
from ..utils import resolve_components
from ..handler_registry import HandlerRegistry, HANDLER_GROUP_VALIDATORS
from ..interfaces import IValidator


class ValidationStage(Stage):
    """Stage 6: Validation."""

    def __init__(self):
        self._runner = ValidationRunner()
        self._handler_order: list[str] = ["cf", "unit"]

    def name(self) -> str:
        return "validation"

    def configure(self, config: Dict[str, Any]) -> None:
        handlers = config.get('validators')
        if handlers is None:
            handlers = config.get('handlers')
        if isinstance(handlers, list) and handlers:
            self._handler_order = list(handlers)
            from .handlers.cf_validator import CFValidator
            from .handlers.unit_validator import UnitValidator

            mapping = {
                'cf': CFValidator,
                'unit': UnitValidator,
            }
            if any(name not in mapping for name in handlers):
                plugin_mapping = HandlerRegistry.get(HANDLER_GROUP_VALIDATORS, IValidator)
                for name, cls in plugin_mapping.items():
                    if name not in mapping:
                        mapping[name] = cls
            validators = resolve_components(handlers, mapping)
            if validators:
                self._runner = ValidationRunner(validators=validators)
        self._runner.configure(config)

    def process(self, context: StageContext) -> StageContext:
        from ..utils import record_handler_applied
        for name in self._handler_order:
            record_handler_applied(context.metadata, self.name(), name)
        return self._runner.process(context)
