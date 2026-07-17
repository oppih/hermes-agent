"""Test that fallback to a different model under the *same* provider clears
credential-pool exhaustion.

Rate limits are per-model for many providers (Google Gemini, etc.). A 429
on model A should not prevent the same key from being tried on model B.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(primary_runtime: dict | None = None, fallback_model=None):
    """Create a minimal AIAgent with optional primary_runtime and fallback."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        agent._primary_runtime = primary_runtime or {}
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._unavailable_fallback_keys = set()
        return agent


def _make_mock_pool(has_credentials=True, has_available=False):
    """Return a mock credential pool in all-exhausted state."""
    pool = MagicMock()
    pool.has_credentials.return_value = has_credentials
    pool.has_available.return_value = has_available
    pool.entries.return_value = [
        MagicMock(last_status="exhausted"),
        MagicMock(last_status="exhausted"),
    ]
    pool.reset_statuses.return_value = 2
    return pool


class TestFallbackSameProviderResetsPool:
    """When the fallback entry targets the same provider (after alias
    normalisation) but a *different* model, exhausted credentials must be
    reset — the new model has its own independent quota pool."""

    def _run_test(self, monkeypatch, primary_runtime, fallback_entries, expected_reset=True):
        agent = _make_agent(
            primary_runtime=primary_runtime,
            fallback_model=fallback_entries,
        )
        pool = _make_mock_pool()
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: pool,
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(MagicMock(base_url="https://generativelanguage.googleapis.com/v1beta"), "gemini-3.1-flash-lite"),
        ):
            result = agent._try_activate_fallback()

        assert result is True, "Fallback should succeed"
        if expected_reset:
            pool.reset_statuses.assert_called_once()
        else:
            pool.reset_statuses.assert_not_called()
        return agent, pool

    def test_same_provider_different_model_clears_exhaustion(self, monkeypatch):
        """Fallback from gemini/gemini-3.5-flash → google/gemini-3.1-flash-lite
        should clear pool exhaustion because google aliases to the same gemini
        provider but the model is different."""
        primary = {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
        }
        fb = [{"provider": "google", "model": "gemini-3.1-flash-lite"}]
        self._run_test(monkeypatch, primary, fb, expected_reset=True)

    def test_different_provider_does_not_clear_exhaustion(self, monkeypatch):
        """Fallback from gemini → opencode-go is a *different* provider.
        Model-level quota does not apply; exhausted pool stays exhausted."""
        primary = {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
        }
        fb = [{"provider": "opencode-go", "model": "deepseek-v4-flash"}]
        self._run_test(monkeypatch, primary, fb, expected_reset=False)

    def test_same_provider_same_model_does_not_clear(self, monkeypatch):
        """Fallback entry that matches the current (provider, model) should
        be deduped by ``try_activate_fallback`` itself (it compares
        ``agent.provider/model``, not ``_primary_runtime``). The pool should
        never be touched because our code runs after dedup."""
        agent = _make_agent(
            primary_runtime={
                "provider": "gemini",
                "model": "gemini-3.5-flash",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
            },
            fallback_model=[{"provider": "gemini", "model": "gemini-3.5-flash"}],
        )
        # Set the live agent attributes so dedup fires
        agent.provider = "gemini"
        agent.model = "gemini-3.5-flash"
        pool = _make_mock_pool()
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: pool,
        )
        # The function should skip the same-model entry entirely and advance
        # past it; with no more entries the chain is exhausted.
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(MagicMock(base_url="https://generativelanguage.googleapis.com/v1beta"), "gemini-3.5-flash"),
        ):
            result = agent._try_activate_fallback()
        # Same model is deduped before our code runs -> skip to next -> exhausted chain -> False
        assert result is False, "Same-model entry should be deduped, chain exhausted"
        pool.reset_statuses.assert_not_called()

    def test_no_primary_runtime_skips_clear(self, monkeypatch):
        """When ``_primary_runtime`` is empty, the guard should be safe and
        never call reset_statuses."""
        agent = _make_agent(primary_runtime={}, fallback_model=[
            {"provider": "google", "model": "gemini-3.1-flash-lite"},
        ])
        pool = _make_mock_pool()
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: pool,
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(MagicMock(), "gemini-3.1-flash-lite"),
        ):
            result = agent._try_activate_fallback()
        assert result is True
        pool.reset_statuses.assert_not_called()

    def test_pool_already_has_available_skips_reset(self, monkeypatch):
        """When the pool already has available credentials, no reset is
        needed — avoid unnecessary write traffic to auth.json."""
        primary = {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
        }
        fb = [{"provider": "google", "model": "gemini-3.1-flash-lite"}]
        agent = _make_agent(primary_runtime=primary, fallback_model=fb)
        # Pool that already has available credentials
        pool = _make_mock_pool(has_available=True)
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: pool,
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(MagicMock(base_url="https://generativelanguage.googleapis.com/v1beta"), "gemini-3.1-flash-lite"),
        ):
            result = agent._try_activate_fallback()
        assert result is True
        pool.reset_statuses.assert_not_called()
