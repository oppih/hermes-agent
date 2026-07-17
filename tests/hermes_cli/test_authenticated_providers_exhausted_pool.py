"""Regression test for #45759 + model-level exhaustion horizon.

An all-exhausted credential pool holds entries but no *usable* credential.
``list_authenticated_providers`` must NOT hide a provider whose pool is
exhausted — rate limits are per-model for many providers (Google Gemini,
etc.), and switching to a different model under the same provider should
work immediately.  See the companion Layer-2 fix in
``try_activate_fallback`` for the runtime recovery path.
"""

import pytest


class _FakePool:
    def __init__(self, available: bool):
        self._available = available

    def has_credentials(self) -> bool:
        # The pool still holds entries...
        return True

    def has_available(self) -> bool:
        # ...but none of them are usable when exhausted/dead.
        return self._available


def _patch_opencode_pool(monkeypatch, *, available: bool):
    """Make the opencode-go aggregator look configured but with a pool whose
    only credential is (un)available, depending on ``available``."""
    import hermes_cli.auth as auth
    import agent.credential_pool as cp

    monkeypatch.setattr(
        auth,
        "_load_auth_store",
        lambda: {
            "version": 1,
            "providers": {},
            "active_provider": None,
            "credential_pool": {"opencode-go": {"entries": [{"id": "x"}]}},
        },
    )
    monkeypatch.setattr(
        cp,
        "load_pool",
        lambda provider: _FakePool(available if provider == "opencode-go" else True),
    )


@pytest.fixture(autouse=True)
def _strip_provider_env(monkeypatch):
    """Don't let real provider keys in the environment authenticate providers
    through a different code path than the pool gate under test."""
    import os

    for key in list(os.environ):
        if "OPENCODE" in key or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)


def test_exhausted_pool_provider_is_authenticated(monkeypatch):
    """With credentials visible, an exhausted pool is still authenticated.
    The provider stays visible so the user can switch to a different model
    under the same provider (rate limits are per-model not per-key)."""
    from hermes_cli.model_switch import get_authenticated_provider_slugs

    _patch_opencode_pool(monkeypatch, available=False)
    slugs = get_authenticated_provider_slugs(current_provider="alibaba")
    assert "opencode-go" in slugs


def test_pool_provider_with_available_credential_is_authenticated(monkeypatch):
    """Control: with a usable credential the provider IS authenticated, proving
    the test drives the credential gate rather than excluding it for some other
    reason."""
    from hermes_cli.model_switch import get_authenticated_provider_slugs

    _patch_opencode_pool(monkeypatch, available=True)
    slugs = get_authenticated_provider_slugs(current_provider="alibaba")
    assert "opencode-go" in slugs


def test_opaque_legacy_pool_value_stays_visible(monkeypatch):
    """Legacy token-style auth-store values have no parsed pool entries."""
    from hermes_cli.model_switch import _credential_pool_is_usable

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda _provider: type(
            "EmptyPool",
            (),
            {
                "has_credentials": lambda self: False,
                "has_available": lambda self: False,
            },
        )(),
    )

    assert _credential_pool_is_usable("opencode-go", raw_pool_present=True)
