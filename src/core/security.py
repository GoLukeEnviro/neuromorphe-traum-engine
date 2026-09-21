"""Zugriffs- und Eigentumsgrenzen für die API (``shared``-Modus).

Der lokale Betriebsmodus (Default) bleibt unverändert: keine Authentifizierung,
keine Owner-Prüfung. Sobald ``OPERATION_MODE=shared`` gesetzt ist, gilt:

* Jede sensible Route verlangt den Header ``X-API-Token``.
* Ohne konfigurierte Tokens antwortet der Dienst fail-closed (503), statt
  still offen weiterzulaufen.
* Erzeugte Ressourcen werden dem Client-Owner des Erzeugers zugeordnet; ein
  fremder Owner erhält 404 (kein Bestätigen fremder Ressourcen).

Das Modul wird nur von der Anwendung importiert und ist frei von Seiteneffekten
auf Importzeit (keine Env-Variable, kein Logging-Schreibzugriff).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, status

from core.config import settings

#: Owner-Kennung im lokalen Modus (kein Client konfiguriert).
LOCAL_OWNER = "local"


@dataclass(frozen=True)
class ClientIdentity:
    """Identität eines API-Clients (Owner + Nachweisquelle)."""

    owner: str
    token: Optional[str] = None


def _client_tokens() -> dict:
    """Konfigurierte ``{token: owner}``-Paare (frisch gelesen)."""
    tokens = getattr(settings, "client_tokens", None)
    if isinstance(tokens, dict):
        return tokens
    # Fallback, falls ein Testobjekt die Property nicht bereitstellt.
    raw = getattr(settings, "API_CLIENT_TOKENS", "") or ""
    pairs = {}
    for entry in str(raw).split(","):
        token, _, owner = entry.strip().partition(":")
        token, owner = token.strip(), owner.strip()
        if token and owner:
            pairs[token] = owner
    return pairs


def require_client(x_api_token: Optional[str] = Header(None)) -> ClientIdentity:
    """FastAPI-Dependency: prüft den Client-Token und liefert die Identität.

    Im lokalen Modus ist jeder Aufruf der Owner ``local``. Im geteilten Modus
    ohne konfigurierte Tokens ist der Dienst nicht benutzbar (503) - das ist
    die fail-closed-Reaktion, kein stilles Weiterlaufen.
    """
    mode = str(getattr(settings, "OPERATION_MODE", "local")).strip().lower()
    if mode != "shared":
        return ClientIdentity(owner=LOCAL_OWNER)

    tokens = _client_tokens()
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared mode without configured client tokens",
        )

    if not x_api_token or x_api_token not in tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token",
            headers={"WWW-Authenticate": "X-API-Token"},
        )

    return ClientIdentity(owner=tokens[x_api_token], token=x_api_token)


def assert_owner(resource_owner: Optional[str], identity: ClientIdentity) -> None:
    """Prüft den Owner einer Ressource.

    Im lokalen Modus ist jede Ressource zugänglich (bisheriges Verhalten). Im
    geteilten Modus ist ein fremder Owner ein 404 - nicht 403, damit die
    Existenz fremder Ressourcen nicht bestätigt wird. Ressourcen ohne
    hinterlegten Owner gehören dem lokalen Betrieb und sind im geteilten Modus
    für niemanden abrufbar (fail-closed).
    """
    mode = str(getattr(settings, "OPERATION_MODE", "local")).strip().lower()
    if mode != "shared":
        return
    if resource_owner != identity.owner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
