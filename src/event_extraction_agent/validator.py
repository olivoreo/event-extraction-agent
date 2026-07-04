from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from event_extraction_agent.models import Event


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    event: Event | None
    errors: list[ValidationIssue]


def validate_extraction_result(payload: Any) -> ValidationResult:
    if not isinstance(payload, Mapping):
        return ValidationResult(
            is_valid=False,
            event=None,
            errors=[
                ValidationIssue(
                    field="root",
                    code="invalid_type",
                    message="extraction result must be a mapping compatible with Event",
                )
            ],
        )

    try:
        event = Event(**payload)
    except ValidationError as error:
        return ValidationResult(is_valid=False, event=None, errors=_issues_from_pydantic(error))

    if event.start_at is None and event.end_at is None:
        return ValidationResult(
            is_valid=False,
            event=None,
            errors=[
                ValidationIssue(
                    field="start_at",
                    code="missing_event_date",
                    message="event must have start_at or end_at",
                )
            ],
        )

    return ValidationResult(is_valid=True, event=event, errors=[])


def _issues_from_pydantic(error: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for item in error.errors():
        field = _field_from_location(item.get("loc", ()))
        issue_type = str(item.get("type", "validation_error"))
        message = str(item.get("msg", "Validation error"))
        issues.append(
            ValidationIssue(
                field=field,
                code=_code_from_pydantic_type(issue_type),
                message=_message_from_pydantic(message),
            )
        )
    return issues


def _field_from_location(location: Any) -> str:
    if not location:
        return "root"
    if isinstance(location, tuple):
        return ".".join(str(part) for part in location)
    return str(location)


def _code_from_pydantic_type(issue_type: str) -> str:
    if issue_type == "extra_forbidden":
        return "unexpected_field"
    return "invalid_field"


def _message_from_pydantic(message: str) -> str:
    prefix = "Value error, "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return message
