"""Tests für Zugriffs- und Eigentumsgrenzen der API.

Vertrag (Entscheidung dokumentiert in ``docs/security-boundaries.md``):

* CORS gewährt ``Access-Control-Allow-Origin`` ausschließlich für die
  konfigurierten Origins ``settings.CORS_ORIGINS`` — niemals als Wildcard.
* Der Modus ``shared`` verlangt für jede sensible Route einen Token im Header
  ``X-API-Token``. Ohne konfigurierte Tokens antwortet der Dienst fail-closed
  (503) statt still offen weiterzulaufen.
* Ressourcen sind dem Owner ihres Erzeugers zugeordnet; ein Zugriff durch einen
  fremden Owner endet mit 404 (keine Bestätigung fremder Ressourcen).
* Der lokale Betriebsmodus (Default) bleibt ohne Token nutzbar.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.config import settings
from core.security import LOCAL_OWNER, require_client

ALLOWED_ORIGIN = "http://localhost:8501"
FOREIGN_ORIGIN = "http://evil.example"

ALICE_TOKEN = "token-alice-0123456789"
BOB_TOKEN = "token-bob-0123456789"


@pytest.fixture
def shared_mode(monkeypatch):
    """Schaltet die API in den geteilten Modus mit zwei Clients."""
    monkeypatch.setattr(settings, "OPERATION_MODE", "shared")
    monkeypatch.setattr(
        settings,
        "API_CLIENT_TOKENS",
        f"{ALICE_TOKEN}:alice,{BOB_TOKEN}:bob",
    )
    return {"alice": {"X-API-Token": ALICE_TOKEN}, "bob": {"X-API-Token": BOB_TOKEN}}


class TestCorsBoundary:
    """CORS ist auf die konfigurierten Origins begrenzt."""

    def test_allowed_origin_is_echoed(self, test_client):
        response = test_client.options(
            "/api/v1/analyze/text",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN

    def test_foreign_preflight_is_rejected(self, test_client):
        response = test_client.options(
            "/api/v1/analyze/text",
            headers={
                "Origin": FOREIGN_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert "access-control-allow-origin" not in response.headers
        assert response.status_code >= 400

    def test_foreign_plain_options_gets_no_grant(self, test_client):
        response = test_client.options(
            "/api/v1/analyze/text", headers={"Origin": FOREIGN_ORIGIN}
        )

        assert "access-control-allow-origin" not in response.headers
        assert response.status_code == 403

    def test_wildcard_origin_is_never_sent(self, test_client):
        response = test_client.get("/health", headers={"Origin": FOREIGN_ORIGIN})

        assert response.headers.get("access-control-allow-origin") != "*"


class TestLocalMode:
    """Der bestehende lokale Betriebsmodus bleibt unverändert nutzbar."""

    def test_local_request_needs_no_token(self, test_client):
        response = test_client.get("/api/v1/arrangements")

        assert response.status_code == 200

    def test_local_client_is_mapped_to_local_owner(self):
        identity = require_client(None)

        assert identity.owner == LOCAL_OWNER


class TestSharedModeAuth:
    """Im geteilten Modus ist ein gültiger Token Pflicht (fail-closed)."""

    def test_missing_token_is_rejected(self, test_client, shared_mode):
        response = test_client.get("/api/v1/arrangements")

        assert response.status_code == 401

    def test_unknown_token_is_rejected(self, test_client, shared_mode):
        response = test_client.get(
            "/api/v1/arrangements", headers={"X-API-Token": "not-a-configured-token"}
        )

        assert response.status_code == 401

    def test_valid_token_is_accepted(self, test_client, shared_mode):
        response = test_client.get(
            "/api/v1/arrangements", headers=shared_mode["alice"]
        )

        assert response.status_code == 200

    def test_shared_mode_without_tokens_is_fail_closed(self, test_client, monkeypatch):
        monkeypatch.setattr(settings, "OPERATION_MODE", "shared")
        monkeypatch.setattr(settings, "API_CLIENT_TOKENS", {})

        response = test_client.get("/api/v1/arrangements")

        assert response.status_code == 503

    def test_upload_route_is_protected(self, test_client, shared_mode):
        response = test_client.post(
            "/api/v1/stems",
            files={"audio_file": ("kick.wav", b"RIFF0000WAVE", "audio/wav")},
        )

        assert response.status_code == 401


class TestOwnershipBoundary:
    """Erzeugte Ressourcen sind ihrem Owner zugeordnet."""

    def test_arrangement_is_scoped_to_its_owner(self, test_client, shared_mode):
        with patch(
            "src.services.arranger.ArrangerService.create_arrangement"
        ) as mock_create:
            mock_create.return_value = {
                "structure": {"sections": []},
                "stems": [],
                "metadata": {},
            }
            created = test_client.post(
                "/api/v1/arrangements",
                json={"prompt": "driving industrial techno 138 bpm", "duration": 180},
                headers=shared_mode["alice"],
            )

        assert created.status_code == 201

        listed = test_client.get("/api/v1/arrangements", headers=shared_mode["alice"])
        assert listed.status_code == 200
        arrangements = listed.json()["arrangements"]
        assert len(arrangements) == 1, arrangements
        arrangement_id = arrangements[0]["id"]
        assert arrangements[0]["metadata"]["owner"] == "alice"

        own = test_client.get(
            f"/api/v1/arrangements/{arrangement_id}", headers=shared_mode["alice"]
        )
        assert own.status_code == 200

        foreign = test_client.get(
            f"/api/v1/arrangements/{arrangement_id}", headers=shared_mode["bob"]
        )
        assert foreign.status_code == 404

        foreign_list = test_client.get(
            "/api/v1/arrangements", headers=shared_mode["bob"]
        )
        assert foreign_list.json()["arrangements"] == []

    def test_render_download_is_scoped_to_its_owner(
        self, test_client, shared_mode, temp_dir
    ):
        output_file = temp_dir / "render_owner.wav"
        output_file.write_bytes(b"fake audio data")

        with patch(
            "src.services.renderer.RendererService.render_arrangement"
        ) as mock_render:
            mock_render.return_value = {
                "output_path": str(output_file),
                "duration": 1.0,
                "metadata": {},
            }
            created = test_client.post(
                "/api/v1/arrangements/any-arrangement/render",
                json={"format": "wav", "quality": "high"},
                headers=shared_mode["alice"],
            )

        assert created.status_code == 200
        render_id = created.json()["render_id"]

        own = test_client.get(
            f"/api/v1/renders/{render_id}/download", headers=shared_mode["alice"]
        )
        assert own.status_code == 200

        foreign = test_client.get(
            f"/api/v1/renders/{render_id}/download", headers=shared_mode["bob"]
        )
        assert foreign.status_code == 404
