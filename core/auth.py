"""
core/auth.py

Google OAuth 2.0 helper for Streamlit.
This manages Google authentication redirects, exchanges code for verified user info,
and supports simple email fallback for local testing when Google Client ID is not set.
"""

import streamlit as st
import urllib.parse
import requests
import json
import base64

# Configure OAuth keys via environment variables or Streamlit secrets
try:
    GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
except Exception:
    GOOGLE_CLIENT_ID = ""
    GOOGLE_CLIENT_SECRET = ""

def get_redirect_uri() -> str:
    """
    Get the redirect URI (current application base URL).
    We try to extract it dynamically or fall back to localhost.
    """
    # Streamlit query params can help us determine redirect location
    # Standard redirect target is the main page
    return st.secrets.get("OAUTH_REDIRECT_URI", "http://localhost:8501/")

def get_google_auth_url() -> str:
    """
    Generate Google's OAuth consent screen link.
    """
    base_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": get_redirect_uri(),
        "response_type": "code",
        "scope": "openid email",
        "state": "sentio_auth_state",
        "prompt": "select_account"
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def exchange_code_for_email(auth_code: str) -> str:
    """
    Exchange the OAuth authorization code for the user's verified email.
    """
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": auth_code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": get_redirect_uri(),
        "grant_type": "authorization_code"
    }
    
    try:
        response = requests.post(token_url, data=data, timeout=10.0)
        if response.status_code != 200:
            return ""
        
        token_data = response.json()
        id_token = token_data.get("id_token", "")
        if not id_token:
            return ""
            
        # Decode the JWT ID Token payload (2nd part of JWT)
        parts = id_token.split(".")
        if len(parts) < 2:
            return ""
            
        # Add padding if necessary
        payload_b64 = parts[1]
        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
        
        payload_json = base64.b64decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
        
        # Return verified email
        return payload.get("email", "").lower().strip()
    except Exception as e:
        st.error(f"Authentication exchange failed: {e}")
        return ""

def is_google_auth_configured() -> bool:
    """
    Check if Google OAuth credentials exist.
    """
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
