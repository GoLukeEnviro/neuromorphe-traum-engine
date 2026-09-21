# Sicherheitsgrenzen der API (Auth, Ownership, CORS)

Stand: 2026-09-21. Verbindliche Entscheidung für den laufenden Nachtlauf.

## Entscheidung: Dienst bleibt privat

Die Traum-Engine wird **nicht öffentlich exponiert**, solange die hier
beschriebene Schutzschicht nicht durch Tests belegt ist. Es gibt weiterhin
**keine öffentliche Freigabe** (kein Reverse-Proxy-Publish, keine Domain, keine
Portfreigabe), und diese Datei dokumentiert die Grenzen nur — sie ändert die
Deployment-Realität nicht. Ein öffentlicher Betrieb ist eine eigene,
ausdrücklich zu genehmigende Entscheidung des Operators.

## Zwei Betriebsmodi

| Modus | Wert | Verhalten |
|---|---|---|
| lokal (Default) | `OPERATION_MODE=local` | Der bisherige lokale Betrieb. Keine Tokens, keine Owner-Prüfung. Alle Routen verhalten sich wie vorher. |
| geteilt | `OPERATION_MODE=shared` | Jede fachliche Route verlangt den Header `X-API-Token` mit einem konfigurierten Client-Token. Ressourcen sind dem Owner ihres Erzeugers zugeordnet. |

Der Modus ist bewusst ein Schalter und keine automatische Erkennung: „lokal"
ist der Default, damit ein bestehender Checkout ohne Konfiguration genau wie
vorher läuft.

## Grenzen im Modus `shared`

* **Client-Nachweis.** `API_CLIENT_TOKENS` enthält `token:owner`-Paare,
  kommagetrennt (`token-a:alice,token-b:bob`). Fehlt der Header oder ist der
  Token unbekannt → `401`.
* **Fail-closed ohne Tokens.** Ist `OPERATION_MODE=shared` gesetzt und
  `API_CLIENT_TOKENS` leer, antwortet der Dienst `503` — er läuft **nicht**
  still offen weiter.
* **Owner-Bindung.**
  * `POST /api/v1/arrangements` schreibt `metadata.owner` des aufrufenden
    Clients.
  * `GET`/`PUT`/`DELETE /api/v1/arrangements/{id}` prüfen diesen Owner; ein
    fremder Client erhält `404` (kein Bestätigen fremder Ressourcen). Ein
    `PUT` kann den Owner nicht überschreiben.
  * `GET /api/v1/renders/{id}` und `.../download` prüfen `options.owner` des
    Render-Jobs, ebenfalls `404` bei fremdem Owner.
  * `GET /api/v1/arrangements` und `GET /api/v1/renders` listen im
    `shared`-Modus nur die eigenen Ressourcen.
  * Ressourcen **ohne** hinterlegten Owner (vor dieser Änderung erzeugte
    Datensätze) sind im `shared`-Modus für niemanden abrufbar — fail-closed.
* **Health bleibt offen.** `GET /health` und `GET /system/health` verlangen
  keinen Token (Monitoring).
* **WebSocket-Endpunkte** (`/ws`, `/ws/render/progress`) sind noch nicht an die
  Token-Prüfung angebunden; sie sind deshalb ein bekannter offener Punkt für
  einen öffentlichen Betrieb (siehe unten).

## CORS

`CORSMiddleware` bekommt ausschließlich die Origins aus
`settings.CORS_ORIGINS` (`http://localhost:8501`, `http://localhost:3000`).
`"*"` wird beim Konfigurations-Laden abgelehnt (`ValueError`), und der
explizite OPTIONS-Handler antwortet einem fremden Origin mit `403` **ohne**
`Access-Control-Allow-Origin`. `allow_headers` ist auf `Content-Type` und
`X-API-Token` begrenzt, `allow_methods` auf die tatsächlich genutzten Verben.

## Tests

`tests/test_api/test_auth_boundaries.py` (13 Tests) pinnt den Vertrag:
fremder Preflight ohne Grant, Wildcard nie gesendet, lokaler Modus ohne Token
nutzbar, `401` bei fehlendem/unbekanntem Token, `503` fail-closed ohne
konfigurierte Tokens, Upload-Route geschützt, Arrangement und Render-Download
owner-gebunden inkl. Listen-Filterung.

RED-Nachweis (vor der Änderung): `8 failed, 5 passed`.
GREEN-Nachweis: `13 passed`.

## Offene Punkte vor einer öffentlichen Freigabe

1. WebSocket-Routen (`/ws`, `/ws/render/progress`) an dieselbe Token-Prüfung
   binden — der `WebSocket`-Handshake kennt keinen `Depends`-Pfad, das braucht
   eine eigene Prüfung im Accept.
2. Token-Verwaltung: Rotation, Ablauf und Herkunft der Tokens (derzeit nur
   Konfigurationsdatei/Umgebung, keine Persistenz, kein Hashing).
3. `X-API-Token` in Transit: TLS-Terminierung ist Voraussetzung, sonst ist der
   Token im Klartext unterwegs.
4. Rate-Limiting und Audit-Log für authentifizierte Zugriffe.
