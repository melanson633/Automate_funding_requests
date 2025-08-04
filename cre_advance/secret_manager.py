# Temporary stub until real Secret Manager integration
import os


def get_secret(name: str) -> str:
    """Return secret from environment; raise if missing."""
    try:
        return os.environ[name]
    except KeyError as exc:
        raise RuntimeError(f"Missing required secret: {name}") from exc
