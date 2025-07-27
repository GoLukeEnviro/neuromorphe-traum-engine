"""Tests für Security-Service"""

import pytest
import jwt
import bcrypt
import time
import secrets
import hashlib
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.services.security_service import (
    SecurityService, AuthenticationManager, AuthorizationManager,
    TokenManager, PasswordManager, EncryptionManager,
    SessionManager, RateLimiter, SecurityAuditor,
    TwoFactorAuth, CertificateManager, KeyManager,
    SecurityPolicy, AccessControl, PermissionManager,
    SecurityEvent, SecurityAlert, SecurityMetrics,
    User, Role, Permission, Session, Token,
    SecurityConfig, AuthConfig, EncryptionConfig,
    RateLimitConfig, AuditConfig, TwoFactorConfig,
    JWTToken, APIKey, RefreshToken, AccessToken,
    SecurityContext, SecurityPrincipal, SecurityClaim,
    SecurityValidator, SecurityFilter, SecurityMiddleware,
    CryptoProvider, HashProvider, SignatureProvider
)
from src.core.config import SecurityConfig as CoreSecurityConfig
from src.core.exceptions import (
    SecurityError, AuthenticationError, AuthorizationError,
    TokenError, PasswordError, EncryptionError,
    SessionError, RateLimitError, AuditError,
    TwoFactorError, CertificateError, KeyError as SecurityKeyError
)
from src.schemas.security import (
    UserData, RoleData, PermissionData, SessionData,
    TokenData, SecurityEventData, SecurityMetricsData,
    AuthRequestData, AuthResponseData, SecurityConfigData
)


@dataclass
class TestUser:
    """Test-Benutzer"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    permissions: List[str]
    is_active: bool = True
    is_verified: bool = True
    two_factor_enabled: bool = False
    created_at: datetime = None
    last_login: datetime = None


class TestSecurityService:
    """Tests für Security-Service"""
    
    @pytest.fixture
    def security_config(self):
        """Security-Konfiguration für Tests"""
        return CoreSecurityConfig(
            jwt_secret="test_jwt_secret_key_for_testing_purposes_only",
            jwt_algorithm="HS256",
            jwt_expiration=3600,  # 1 Stunde
            refresh_token_expiration=86400,  # 24 Stunden
            password_min_length=8,
            password_require_uppercase=True,
            password_require_lowercase=True,
            password_require_numbers=True,
            password_require_symbols=True,
            password_hash_rounds=12,
            session_timeout=1800,  # 30 Minuten
            max_login_attempts=5,
            lockout_duration=900,  # 15 Minuten
            rate_limit_enabled=True,
            rate_limit_requests=100,
            rate_limit_window=3600,  # 1 Stunde
            two_factor_enabled=True,
            two_factor_issuer="Neuromorphe Traum Engine",
            encryption_enabled=True,
            encryption_algorithm="AES-256-GCM",
            audit_enabled=True,
            audit_log_retention=90,  # 90 Tage
            security_headers_enabled=True,
            cors_enabled=True,
            cors_origins=["http://localhost:3000"],
            ssl_required=False,  # Für Tests deaktiviert
            certificate_validation=False  # Für Tests deaktiviert
        )
    
    @pytest.fixture
    def security_service(self, security_config):
        """Security-Service für Tests"""
        return SecurityService(security_config)
    
    @pytest.fixture
    def test_user(self):
        """Test-Benutzer"""
        return TestUser(
            id="user_12345",
            username="testuser",
            email="test@example.com",
            password_hash=bcrypt.hashpw("TestPassword123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            roles=["user", "stem_uploader"],
            permissions=["read_stems", "upload_stems", "create_arrangements"],
            created_at=datetime.now(),
            last_login=datetime.now() - timedelta(hours=1)
        )
    
    @pytest.mark.unit
    def test_security_service_initialization(self, security_config):
        """Test: Security-Service-Initialisierung"""
        service = SecurityService(security_config)
        
        assert service.config == security_config
        assert isinstance(service.auth_manager, AuthenticationManager)
        assert isinstance(service.authz_manager, AuthorizationManager)
        assert isinstance(service.token_manager, TokenManager)
        assert isinstance(service.password_manager, PasswordManager)
        assert isinstance(service.encryption_manager, EncryptionManager)
        assert isinstance(service.session_manager, SessionManager)
        assert isinstance(service.rate_limiter, RateLimiter)
        assert isinstance(service.security_auditor, SecurityAuditor)
    
    @pytest.mark.unit
    def test_security_service_invalid_config(self):
        """Test: Security-Service mit ungültiger Konfiguration"""
        invalid_config = CoreSecurityConfig(
            jwt_secret="",  # Leerer JWT-Secret
            jwt_expiration=-1,  # Negative Ablaufzeit
            password_min_length=0,  # Zu kurze Passwort-Mindestlänge
            password_hash_rounds=0,  # Keine Hash-Runden
            session_timeout=-1,  # Negative Session-Timeout
            max_login_attempts=0,  # Keine Login-Versuche erlaubt
            rate_limit_requests=0  # Keine Rate-Limit-Requests
        )
        
        with pytest.raises(SecurityError):
            SecurityService(invalid_config)
    
    @pytest.mark.unit
    def test_user_registration(self, security_service):
        """Test: Benutzer-Registrierung"""
        # Benutzer-Daten
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "confirm_password": "SecurePassword123!"
        }
        
        # Benutzer registrieren
        user = security_service.register_user(user_data)
        
        assert user is not None
        assert user.username == "newuser"
        assert user.email == "newuser@example.com"
        assert user.is_active == True
        assert user.is_verified == False  # Sollte zunächst unverifiziert sein
        
        # Passwort sollte gehasht sein
        assert user.password_hash != "SecurePassword123!"
        assert bcrypt.checkpw("SecurePassword123!".encode('utf-8'), user.password_hash.encode('utf-8'))
    
    @pytest.mark.unit
    def test_user_registration_validation(self, security_service):
        """Test: Benutzer-Registrierung-Validierung"""
        # Ungültige Passwörter testen
        invalid_passwords = [
            "short",  # Zu kurz
            "nouppercase123!",  # Keine Großbuchstaben
            "NOLOWERCASE123!",  # Keine Kleinbuchstaben
            "NoNumbers!",  # Keine Zahlen
            "NoSymbols123",  # Keine Symbole
        ]
        
        for password in invalid_passwords:
            user_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": password,
                "confirm_password": password
            }
            
            with pytest.raises(PasswordError):
                security_service.register_user(user_data)
        
        # Passwörter stimmen nicht überein
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePassword123!",
            "confirm_password": "DifferentPassword123!"
        }
        
        with pytest.raises(PasswordError):
            security_service.register_user(user_data)
    
    @pytest.mark.unit
    def test_user_authentication(self, security_service, test_user):
        """Test: Benutzer-Authentifizierung"""
        # Mock: Benutzer aus Datenbank laden
        with patch.object(security_service.auth_manager, 'get_user_by_username', return_value=test_user):
            # Erfolgreiche Authentifizierung
            auth_result = security_service.authenticate_user("testuser", "TestPassword123!")
            
            assert auth_result.success == True
            assert auth_result.user.username == "testuser"
            assert auth_result.access_token is not None
            assert auth_result.refresh_token is not None
            
            # Fehlgeschlagene Authentifizierung (falsches Passwort)
            auth_result = security_service.authenticate_user("testuser", "WrongPassword")
            
            assert auth_result.success == False
            assert auth_result.user is None
            assert auth_result.access_token is None
    
    @pytest.mark.unit
    def test_jwt_token_generation_and_validation(self, security_service, test_user):
        """Test: JWT-Token-Generierung und -Validierung"""
        # Token generieren
        token_data = {
            "user_id": test_user.id,
            "username": test_user.username,
            "roles": test_user.roles,
            "permissions": test_user.permissions
        }
        
        access_token = security_service.token_manager.generate_access_token(token_data)
        refresh_token = security_service.token_manager.generate_refresh_token(token_data)
        
        assert access_token is not None
        assert refresh_token is not None
        assert isinstance(access_token, str)
        assert isinstance(refresh_token, str)
        
        # Token validieren
        decoded_access = security_service.token_manager.validate_access_token(access_token)
        decoded_refresh = security_service.token_manager.validate_refresh_token(refresh_token)
        
        assert decoded_access["user_id"] == test_user.id
        assert decoded_access["username"] == test_user.username
        assert decoded_access["roles"] == test_user.roles
        
        assert decoded_refresh["user_id"] == test_user.id
        assert decoded_refresh["username"] == test_user.username
    
    @pytest.mark.unit
    def test_jwt_token_expiration(self, security_service, test_user):
        """Test: JWT-Token-Ablauf"""
        # Token mit kurzer Ablaufzeit generieren
        token_data = {
            "user_id": test_user.id,
            "username": test_user.username
        }
        
        # Mock: Kurze Ablaufzeit
        with patch.object(security_service.config, 'jwt_expiration', 1):  # 1 Sekunde
            short_token = security_service.token_manager.generate_access_token(token_data)
            
            # Token sollte zunächst gültig sein
            decoded = security_service.token_manager.validate_access_token(short_token)
            assert decoded["user_id"] == test_user.id
            
            # Warten bis Token abläuft
            time.sleep(2)
            
            # Token sollte jetzt ungültig sein
            with pytest.raises(TokenError):
                security_service.token_manager.validate_access_token(short_token)
    
    @pytest.mark.unit
    def test_password_hashing_and_verification(self, security_service):
        """Test: Passwort-Hashing und -Verifizierung"""
        password = "SecureTestPassword123!"
        
        # Passwort hashen
        password_hash = security_service.password_manager.hash_password(password)
        
        assert password_hash != password
        assert isinstance(password_hash, str)
        assert len(password_hash) > 50  # Bcrypt-Hash sollte lang sein
        
        # Passwort verifizieren
        is_valid = security_service.password_manager.verify_password(password, password_hash)
        assert is_valid == True
        
        # Falsches Passwort
        is_valid = security_service.password_manager.verify_password("WrongPassword", password_hash)
        assert is_valid == False
    
    @pytest.mark.unit
    def test_password_strength_validation(self, security_service):
        """Test: Passwort-Stärke-Validierung"""
        # Starke Passwörter
        strong_passwords = [
            "SecurePassword123!",
            "MyVeryStr0ng@Password",
            "C0mpl3x#P@ssw0rd!"
        ]
        
        for password in strong_passwords:
            is_strong = security_service.password_manager.validate_password_strength(password)
            assert is_strong == True
        
        # Schwache Passwörter
        weak_passwords = [
            "password",  # Zu einfach
            "12345678",  # Nur Zahlen
            "PASSWORD",  # Nur Großbuchstaben
            "password123",  # Keine Symbole
            "Pass1!",  # Zu kurz
        ]
        
        for password in weak_passwords:
            is_strong = security_service.password_manager.validate_password_strength(password)
            assert is_strong == False
    
    @pytest.mark.unit
    def test_role_based_authorization(self, security_service, test_user):
        """Test: Rollenbasierte Autorisierung"""
        # Benutzer-Kontext erstellen
        security_context = SecurityContext(
            user_id=test_user.id,
            username=test_user.username,
            roles=test_user.roles,
            permissions=test_user.permissions
        )
        
        # Autorisierung für verschiedene Aktionen testen
        
        # Benutzer sollte Stems lesen können
        can_read = security_service.authz_manager.check_permission(
            security_context, "read_stems"
        )
        assert can_read == True
        
        # Benutzer sollte Stems hochladen können
        can_upload = security_service.authz_manager.check_permission(
            security_context, "upload_stems"
        )
        assert can_upload == True
        
        # Benutzer sollte keine Admin-Aktionen durchführen können
        can_admin = security_service.authz_manager.check_permission(
            security_context, "admin_access"
        )
        assert can_admin == False
        
        # Rollenbasierte Autorisierung
        has_user_role = security_service.authz_manager.check_role(
            security_context, "user"
        )
        assert has_user_role == True
        
        has_admin_role = security_service.authz_manager.check_role(
            security_context, "admin"
        )
        assert has_admin_role == False
    
    @pytest.mark.unit
    def test_session_management(self, security_service, test_user):
        """Test: Session-Management"""
        # Session erstellen
        session = security_service.session_manager.create_session(
            user_id=test_user.id,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert session is not None
        assert session.user_id == test_user.id
        assert session.ip_address == "192.168.1.100"
        assert session.is_active == True
        assert session.expires_at > datetime.now()
        
        # Session validieren
        is_valid = security_service.session_manager.validate_session(session.id)
        assert is_valid == True
        
        # Session aktualisieren
        security_service.session_manager.update_session_activity(session.id)
        
        updated_session = security_service.session_manager.get_session(session.id)
        assert updated_session.last_activity > session.last_activity
        
        # Session beenden
        security_service.session_manager.end_session(session.id)
        
        ended_session = security_service.session_manager.get_session(session.id)
        assert ended_session.is_active == False
    
    @pytest.mark.unit
    def test_rate_limiting(self, security_service):
        """Test: Rate-Limiting"""
        client_ip = "192.168.1.100"
        endpoint = "/api/auth/login"
        
        # Rate-Limiter konfigurieren (5 Requests pro Minute)
        rate_limiter = security_service.rate_limiter
        rate_limiter.set_limit(endpoint, requests=5, window=60)
        
        # 5 Requests sollten erlaubt sein
        for i in range(5):
            is_allowed = rate_limiter.is_allowed(client_ip, endpoint)
            assert is_allowed == True
            rate_limiter.record_request(client_ip, endpoint)
        
        # 6. Request sollte blockiert werden
        is_allowed = rate_limiter.is_allowed(client_ip, endpoint)
        assert is_allowed == False
        
        # Rate-Limit-Info abrufen
        limit_info = rate_limiter.get_limit_info(client_ip, endpoint)
        assert limit_info.requests_made == 5
        assert limit_info.requests_remaining == 0
        assert limit_info.reset_time > datetime.now()
    
    @pytest.mark.unit
    def test_data_encryption_and_decryption(self, security_service):
        """Test: Daten-Verschlüsselung und -Entschlüsselung"""
        sensitive_data = "This is sensitive user data that needs encryption"
        
        # Daten verschlüsseln
        encrypted_data = security_service.encryption_manager.encrypt(sensitive_data)
        
        assert encrypted_data != sensitive_data
        assert isinstance(encrypted_data, str)
        assert len(encrypted_data) > len(sensitive_data)
        
        # Daten entschlüsseln
        decrypted_data = security_service.encryption_manager.decrypt(encrypted_data)
        
        assert decrypted_data == sensitive_data
    
    @pytest.mark.unit
    def test_two_factor_authentication(self, security_service, test_user):
        """Test: Zwei-Faktor-Authentifizierung"""
        # 2FA für Benutzer aktivieren
        secret = security_service.two_factor_auth.generate_secret(test_user.id)
        
        assert secret is not None
        assert isinstance(secret, str)
        assert len(secret) == 32  # Base32-Secret sollte 32 Zeichen haben
        
        # QR-Code-URL generieren
        qr_url = security_service.two_factor_auth.generate_qr_url(
            test_user.username, secret
        )
        
        assert qr_url.startswith("otpauth://totp/")
        assert test_user.username in qr_url
        
        # TOTP-Code generieren (simuliert)
        import pyotp
        totp = pyotp.TOTP(secret)
        current_code = totp.now()
        
        # Code verifizieren
        is_valid = security_service.two_factor_auth.verify_code(
            test_user.id, current_code
        )
        assert is_valid == True
        
        # Ungültiger Code
        is_valid = security_service.two_factor_auth.verify_code(
            test_user.id, "000000"
        )
        assert is_valid == False
    
    @pytest.mark.unit
    def test_security_auditing(self, security_service, test_user):
        """Test: Security-Auditing"""
        # Security-Events loggen
        events = [
            SecurityEvent(
                type="user_login",
                user_id=test_user.id,
                ip_address="192.168.1.100",
                user_agent="Test Browser",
                success=True,
                timestamp=datetime.now()
            ),
            SecurityEvent(
                type="failed_login",
                user_id=None,
                ip_address="192.168.1.200",
                user_agent="Malicious Bot",
                success=False,
                details={"attempted_username": "admin"},
                timestamp=datetime.now()
            ),
            SecurityEvent(
                type="permission_denied",
                user_id=test_user.id,
                ip_address="192.168.1.100",
                resource="/admin/users",
                success=False,
                timestamp=datetime.now()
            )
        ]
        
        for event in events:
            security_service.security_auditor.log_event(event)
        
        # Audit-Log abrufen
        audit_logs = security_service.security_auditor.get_events(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(audit_logs) >= 3
        
        # Nach Event-Typ filtern
        login_events = security_service.security_auditor.get_events(
            event_type="user_login"
        )
        assert len(login_events) >= 1
        
        failed_events = security_service.security_auditor.get_events(
            success=False
        )
        assert len(failed_events) >= 2
        
        # Nach Benutzer filtern
        user_events = security_service.security_auditor.get_events(
            user_id=test_user.id
        )
        assert len(user_events) >= 2
    
    @pytest.mark.unit
    def test_api_key_management(self, security_service, test_user):
        """Test: API-Key-Management"""
        # API-Key generieren
        api_key = security_service.token_manager.generate_api_key(
            user_id=test_user.id,
            name="Test API Key",
            permissions=["read_stems", "upload_stems"],
            expires_at=datetime.now() + timedelta(days=30)
        )
        
        assert api_key is not None
        assert api_key.key is not None
        assert api_key.user_id == test_user.id
        assert api_key.name == "Test API Key"
        assert api_key.is_active == True
        
        # API-Key validieren
        validated_key = security_service.token_manager.validate_api_key(api_key.key)
        
        assert validated_key is not None
        assert validated_key.user_id == test_user.id
        assert validated_key.permissions == ["read_stems", "upload_stems"]
        
        # API-Key deaktivieren
        security_service.token_manager.revoke_api_key(api_key.id)
        
        # Deaktivierter Key sollte ungültig sein
        with pytest.raises(TokenError):
            security_service.token_manager.validate_api_key(api_key.key)
    
    @pytest.mark.unit
    def test_security_headers(self, security_service):
        """Test: Security-Headers"""
        # Security-Headers generieren
        headers = security_service.get_security_headers()
        
        # Standard Security-Headers prüfen
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in headers
        assert headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Strict-Transport-Security" in headers
        assert "max-age=" in headers["Strict-Transport-Security"]
        
        assert "Content-Security-Policy" in headers
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        
        assert "Referrer-Policy" in headers
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    
    @pytest.mark.unit
    def test_cors_configuration(self, security_service):
        """Test: CORS-Konfiguration"""
        # CORS-Headers für erlaubte Origin
        allowed_origin = "http://localhost:3000"
        cors_headers = security_service.get_cors_headers(allowed_origin)
        
        assert "Access-Control-Allow-Origin" in cors_headers
        assert cors_headers["Access-Control-Allow-Origin"] == allowed_origin
        
        assert "Access-Control-Allow-Methods" in cors_headers
        assert "GET" in cors_headers["Access-Control-Allow-Methods"]
        assert "POST" in cors_headers["Access-Control-Allow-Methods"]
        
        assert "Access-Control-Allow-Headers" in cors_headers
        assert "Authorization" in cors_headers["Access-Control-Allow-Headers"]
        
        # CORS-Headers für nicht erlaubte Origin
        disallowed_origin = "http://malicious-site.com"
        cors_headers = security_service.get_cors_headers(disallowed_origin)
        
        # Sollte keine CORS-Headers für nicht erlaubte Origins geben
        assert cors_headers == {}


class TestAuthenticationManager:
    """Tests für Authentication-Manager"""
    
    @pytest.fixture
    def auth_manager(self, security_config):
        """Authentication-Manager für Tests"""
        return AuthenticationManager(security_config)
    
    @pytest.mark.unit
    def test_login_attempt_tracking(self, auth_manager):
        """Test: Login-Versuch-Tracking"""
        username = "testuser"
        
        # Mehrere fehlgeschlagene Login-Versuche
        for i in range(4):
            auth_manager.record_failed_login(username)
            
            # Account sollte noch nicht gesperrt sein
            is_locked = auth_manager.is_account_locked(username)
            assert is_locked == False
        
        # 5. fehlgeschlagener Versuch sollte Account sperren
        auth_manager.record_failed_login(username)
        
        is_locked = auth_manager.is_account_locked(username)
        assert is_locked == True
        
        # Lockout-Info abrufen
        lockout_info = auth_manager.get_lockout_info(username)
        assert lockout_info.is_locked == True
        assert lockout_info.failed_attempts == 5
        assert lockout_info.unlock_time > datetime.now()
    
    @pytest.mark.unit
    def test_successful_login_resets_attempts(self, auth_manager):
        """Test: Erfolgreicher Login setzt Versuche zurück"""
        username = "testuser"
        
        # Einige fehlgeschlagene Versuche
        for i in range(3):
            auth_manager.record_failed_login(username)
        
        # Erfolgreicher Login
        auth_manager.record_successful_login(username)
        
        # Failed-Attempts sollten zurückgesetzt sein
        lockout_info = auth_manager.get_lockout_info(username)
        assert lockout_info.failed_attempts == 0
        assert lockout_info.is_locked == False
    
    @pytest.mark.unit
    def test_account_unlock_after_timeout(self, auth_manager):
        """Test: Account-Entsperrung nach Timeout"""
        username = "testuser"
        
        # Account sperren
        for i in range(5):
            auth_manager.record_failed_login(username)
        
        assert auth_manager.is_account_locked(username) == True
        
        # Mock: Lockout-Zeit überschreiten
        with patch('datetime.datetime') as mock_datetime:
            # Aktuelle Zeit + Lockout-Dauer + 1 Minute
            future_time = datetime.now() + timedelta(minutes=16)
            mock_datetime.now.return_value = future_time
            
            # Account sollte jetzt entsperrt sein
            is_locked = auth_manager.is_account_locked(username)
            assert is_locked == False


class TestTokenManager:
    """Tests für Token-Manager"""
    
    @pytest.fixture
    def token_manager(self, security_config):
        """Token-Manager für Tests"""
        return TokenManager(security_config)
    
    @pytest.mark.unit
    def test_token_blacklisting(self, token_manager):
        """Test: Token-Blacklisting"""
        # Token generieren
        token_data = {"user_id": "12345", "username": "testuser"}
        access_token = token_manager.generate_access_token(token_data)
        
        # Token sollte zunächst gültig sein
        decoded = token_manager.validate_access_token(access_token)
        assert decoded["user_id"] == "12345"
        
        # Token auf Blacklist setzen
        token_manager.blacklist_token(access_token)
        
        # Token sollte jetzt ungültig sein
        with pytest.raises(TokenError):
            token_manager.validate_access_token(access_token)
    
    @pytest.mark.unit
    def test_refresh_token_rotation(self, token_manager):
        """Test: Refresh-Token-Rotation"""
        token_data = {"user_id": "12345", "username": "testuser"}
        
        # Ursprüngliches Refresh-Token
        refresh_token = token_manager.generate_refresh_token(token_data)
        
        # Token verwenden um neues Access-Token zu erhalten
        new_tokens = token_manager.refresh_access_token(refresh_token)
        
        assert new_tokens.access_token is not None
        assert new_tokens.refresh_token is not None
        assert new_tokens.refresh_token != refresh_token  # Neues Refresh-Token
        
        # Altes Refresh-Token sollte ungültig sein
        with pytest.raises(TokenError):
            token_manager.refresh_access_token(refresh_token)
    
    @pytest.mark.unit
    def test_token_introspection(self, token_manager):
        """Test: Token-Introspection"""
        token_data = {
            "user_id": "12345",
            "username": "testuser",
            "roles": ["user", "premium"],
            "permissions": ["read", "write"]
        }
        
        access_token = token_manager.generate_access_token(token_data)
        
        # Token-Informationen abrufen
        token_info = token_manager.introspect_token(access_token)
        
        assert token_info.active == True
        assert token_info.user_id == "12345"
        assert token_info.username == "testuser"
        assert token_info.roles == ["user", "premium"]
        assert token_info.permissions == ["read", "write"]
        assert token_info.expires_at > datetime.now()


class TestSecurityServiceIntegration:
    """Integrationstests für Security-Service"""
    
    @pytest.mark.integration
    def test_full_authentication_flow(self):
        """Test: Vollständiger Authentifizierungs-Workflow"""
        config = CoreSecurityConfig(
            jwt_secret="integration_test_secret_key",
            jwt_expiration=3600,
            password_min_length=8,
            max_login_attempts=3,
            two_factor_enabled=True
        )
        
        service = SecurityService(config)
        
        # 1. Benutzer registrieren
        user_data = {
            "username": "integrationuser",
            "email": "integration@example.com",
            "password": "SecurePassword123!",
            "confirm_password": "SecurePassword123!"
        }
        
        user = service.register_user(user_data)
        assert user.username == "integrationuser"
        
        # 2. Benutzer authentifizieren
        with patch.object(service.auth_manager, 'get_user_by_username', return_value=user):
            auth_result = service.authenticate_user("integrationuser", "SecurePassword123!")
            
            assert auth_result.success == True
            assert auth_result.access_token is not None
            
            # 3. Token validieren
            decoded_token = service.token_manager.validate_access_token(auth_result.access_token)
            assert decoded_token["username"] == "integrationuser"
            
            # 4. Session erstellen
            session = service.session_manager.create_session(
                user_id=user.id,
                ip_address="192.168.1.100",
                user_agent="Integration Test Browser"
            )
            
            assert session.user_id == user.id
            assert session.is_active == True
            
            # 5. Autorisierung prüfen
            security_context = SecurityContext(
                user_id=user.id,
                username=user.username,
                roles=user.roles,
                permissions=user.permissions
            )
            
            # Benutzer sollte grundlegende Berechtigungen haben
            can_read = service.authz_manager.check_permission(security_context, "read_stems")
            assert can_read == True
            
            # 6. Security-Event loggen
            security_event = SecurityEvent(
                type="integration_test_login",
                user_id=user.id,
                ip_address="192.168.1.100",
                success=True
            )
            
            service.security_auditor.log_event(security_event)
            
            # 7. Session beenden
            service.session_manager.end_session(session.id)
            
            ended_session = service.session_manager.get_session(session.id)
            assert ended_session.is_active == False
    
    @pytest.mark.performance
    def test_security_service_performance(self):
        """Test: Security-Service-Performance"""
        config = CoreSecurityConfig(
            jwt_secret="performance_test_secret_key",
            password_hash_rounds=10  # Reduziert für Performance-Test
        )
        
        service = SecurityService(config)
        
        # Performance-Test: Passwort-Hashing
        start_time = time.time()
        
        for i in range(100):
            password = f"TestPassword{i}!"
            password_hash = service.password_manager.hash_password(password)
            is_valid = service.password_manager.verify_password(password, password_hash)
            assert is_valid == True
        
        hash_time = time.time() - start_time
        
        # Sollte unter 5 Sekunden für 100 Hash-Operationen dauern
        assert hash_time < 5.0
        
        # Performance-Test: Token-Generierung
        start_time = time.time()
        
        tokens = []
        for i in range(1000):
            token_data = {"user_id": f"user_{i}", "username": f"user{i}"}
            access_token = service.token_manager.generate_access_token(token_data)
            tokens.append(access_token)
        
        generation_time = time.time() - start_time
        
        # Sollte unter 2 Sekunden für 1000 Token dauern
        assert generation_time < 2.0
        
        # Performance-Test: Token-Validierung
        start_time = time.time()
        
        for token in tokens:
            decoded = service.token_manager.validate_access_token(token)
            assert "user_id" in decoded
        
        validation_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 1000 Validierungen dauern
        assert validation_time < 1.0
        
        # Performance-Test: Rate-Limiting
        start_time = time.time()
        
        rate_limiter = service.rate_limiter
        rate_limiter.set_limit("/api/test", requests=10000, window=3600)
        
        for i in range(1000):
            client_ip = f"192.168.1.{i % 255}"
            is_allowed = rate_limiter.is_allowed(client_ip, "/api/test")
            if is_allowed:
                rate_limiter.record_request(client_ip, "/api/test")
        
        rate_limit_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 1000 Rate-Limit-Checks dauern
        assert rate_limit_time < 1.0