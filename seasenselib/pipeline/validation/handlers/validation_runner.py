"""
Validation runner.

Runs one or more dataset validators and stores results in context metadata.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging

from ...base import StageContext
from ...interfaces import IValidator
from .cf_validator import CFValidator
from .unit_validator import UnitValidator

logger = logging.getLogger(__name__)


class ValidationRunner:
    """Runs validators and records validation results."""

    def __init__(self, validators: Optional[List[IValidator]] = None, strict: bool = False):
        self.validators = validators or [CFValidator(), UnitValidator()]
        self.strict = strict

    def configure(self, config: Dict[str, Any]) -> None:
        if 'strict' in config:
            self.strict = bool(config['strict'])

    def process(self, context: StageContext) -> StageContext:
        ds = context.dataset

        results: Dict[str, List[str]] = {"errors": [], "warnings": [], "infos": []}

        for validator in self.validators:
            try:
                issues = validator.validate(ds)
            except Exception:
                logger.exception("Validator '%s' failed", validator.name())
                continue

            for issue in issues:
                if issue.severity == "error":
                    results["errors"].append(str(issue))
                elif issue.severity == "warning":
                    results["warnings"].append(str(issue))
                else:
                    results["infos"].append(str(issue))

        context.metadata['validation'] = results

        if self.strict and results["errors"]:
            raise ValueError(f"Validation failed with errors: {results['errors']}")

        return context
