"""Credential roles, independent of HTTP transport and simulator backends."""

from dataclasses import dataclass
import hashlib
import secrets


class AccessDeniedError(PermissionError):
    """An authenticated credential is not allowed to perform this operation."""


@dataclass(frozen=True)
class Principal:
    namespace: str
    role: str

    @property
    def restricted(self) -> bool:
        return self.role != "privileged"


INTERNAL = Principal("internal", "privileged")


class APIKeys:
    """Hash credentials once; never expose their values in help or errors."""

    def __init__(self, privileged: str, policy: str | None = None) -> None:
        self._entries = []
        for token, role in ((privileged, "privileged"), (policy, "policy")):
            if token is None and role == "policy":
                continue
            if not isinstance(token, str) or not token or any(c.isspace() for c in token):
                raise ValueError("API tokens must be nonempty strings without whitespace")
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            if any(secrets.compare_digest(digest, old) for old, _ in self._entries):
                raise ValueError("Privileged and policy credentials must be distinct")
            self._entries.append((digest, Principal(digest.hex(), role)))

    def authenticate(self, header: str) -> Principal | None:
        if not header.startswith("Bearer "):
            return None
        digest = hashlib.sha256(header[7:].encode("utf-8")).digest()
        match = None
        for expected, principal in self._entries:
            if secrets.compare_digest(digest, expected):
                match = principal
        return match
