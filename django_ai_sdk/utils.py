from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.conf import settings

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser


def resolve_setting(setting_name: str, default: Any = None) -> Any:
    """Read a Django setting, falling back to `default` when it is unset."""
    return getattr(settings, setting_name, default)


def serialize(value: Any) -> Any:
    """JSON-safe form of a value, recursing into lists and dicts of pydantic models."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else str(value)


async def aget_active_user(identifier: str) -> AbstractBaseUser | None:
    """The one active user whose USERNAME_FIELD is `identifier`, else None.

    None also covers a USERNAME_FIELD that is not unique, which aget would raise on.
    """
    from django.contrib.auth import get_user_model

    if not identifier:
        return None
    user_model = get_user_model()
    field = user_model.USERNAME_FIELD  # ty: ignore[unresolved-attribute]
    try:
        return await user_model._default_manager.aget(**{field: identifier, "is_active": True})
    except (user_model.DoesNotExist, user_model.MultipleObjectsReturned):
        return None
