from __future__ import annotations

from src.retrieval.query_expander import LocalQueryExpander


class TestLocalQueryExpander:
    def test_injects_synonyms(self, expander: LocalQueryExpander) -> None:
        result = expander.expand("How does scaling work?")
        assert "autoscaling" in result.lower()
        assert "horizontal scaling" in result.lower()

    def test_unknown_terms_pass_through(self, expander: LocalQueryExpander) -> None:
        query = "What is quantum entanglement?"
        result = expander.expand(query)
        assert result == query

    def test_case_insensitive(self, expander: LocalQueryExpander) -> None:
        result = expander.expand("SCALING and CACHING strategies")
        assert "autoscaling" in result.lower()
        assert "redis" in result.lower()

    def test_multiple_triggers(self, expander: LocalQueryExpander) -> None:
        result = expander.expand("scaling and caching")
        assert "autoscaling" in result.lower()
        assert "redis" in result.lower()

    def test_deduplication(self, expander: LocalQueryExpander) -> None:
        result = expander.expand("cache and caching")
        words = result.lower().split()
        # "redis" should appear only once despite both triggers matching
        assert words.count("redis") == 1
