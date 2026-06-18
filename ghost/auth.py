"""Lightweight password gate for public deployments.

Set a password via Streamlit secrets (``.streamlit/secrets.toml`` locally, or
the Secrets UI on Streamlit Cloud)::

    app_password = "your-strong-password"

If no password is configured the app runs open (convenient for local use).
The comparison is constant-time and the password is never stored in session
state — only a boolean 'authenticated' flag.
"""

from __future__ import annotations

import hmac

import streamlit as st


def _configured_password() -> str | None:
    try:
        pw = st.secrets.get("app_password", None)
    except Exception:
        pw = None
    return pw or None


def require_password() -> bool:
    """Render a password prompt and return True only once authenticated.

    Returns True immediately if no password is configured (local mode).
    """
    password = _configured_password()
    if password is None:
        return True  # open mode (no secret set)

    if st.session_state.get("authenticated"):
        return True

    st.markdown("### 🔒 Parallax — restricted")
    entered = st.text_input("Password", type="password",
                            help="Set by the app owner via Streamlit secrets.")
    if not entered:
        st.stop()
    if hmac.compare_digest(entered, str(password)):
        st.session_state["authenticated"] = True
        st.rerun()
    else:
        st.error("Incorrect password.")
        st.stop()
    return False
