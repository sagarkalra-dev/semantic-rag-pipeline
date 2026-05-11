from __future__ import annotations

from abc import ABC, abstractmethod


class BaseQueryExpander(ABC):
    @abstractmethod
    def expand(self, query: str) -> str:
        ...


class LocalQueryExpander(BaseQueryExpander):
    """
    Rule-based query expansion simulating what a GenerativeModel would do.

    Applies synonym injection: when a trigger term appears in the query,
    related technical terms are appended to improve embedding retrieval.
    """

    SYNONYM_MAP: dict[str, list[str]] = {
        "scaling": ["horizontal scaling", "autoscaling", "elasticity", "provisioning"],
        "scale": ["horizontal scaling", "autoscaling", "elasticity", "provisioning"],
        "traffic": ["load balancing", "request routing", "throughput", "connections"],
        "spike": ["peak load", "burst", "surge", "saturation", "autoscaling"],
        "peak load": ["traffic spike", "burst capacity", "autoscaling", "saturation"],
        "caching": ["cache", "Redis", "cache invalidation", "TTL", "LRU"],
        "cache": ["caching", "Redis", "cache invalidation", "TTL", "LRU"],
        "stale": ["cache invalidation", "TTL", "expiry", "consistency"],
        "monitoring": ["observability", "metrics", "tracing", "logging", "alerting"],
        "observability": ["metrics", "tracing", "logging", "Prometheus", "alerting"],
        "deployment": ["blue-green", "canary", "rolling update", "CI/CD", "release"],
        "deploy": ["blue-green deployment", "canary release", "rolling update", "CI/CD"],
        "downtime": ["zero-downtime", "blue-green", "rolling update", "failover"],
        "failure": ["fault tolerance", "circuit breaker", "retry", "fallback", "resilience"],
        "unresponsive": ["timeout", "circuit breaker", "fault tolerance", "fallback", "cascading failure"],
        "load balancing": ["round-robin", "least connections", "consistent hashing", "routing"],
        "messaging": ["event-driven", "Kafka", "pub/sub", "message queue", "async"],
        "database": ["sharding", "replication", "partitioning", "PostgreSQL", "consistency"],
        "security": ["mTLS", "service mesh", "authentication", "encryption", "certificates"],
        "throttling": ["rate limiting", "backpressure", "throughput control", "load shedding"],
        "rate limit": ["throttling", "backpressure", "token bucket", "load shedding"],
        "performance": ["latency", "throughput", "p99", "degradation", "saturation", "SLO"],
        "degradation": ["latency", "performance", "SLO", "error budget", "alerting"],
    }

    def expand(self, query: str) -> str:
        query_lower = query.lower()
        expansions: list[str] = [query]
        for trigger, synonyms in self.SYNONYM_MAP.items():
            if trigger in query_lower:
                expansions.extend(synonyms)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for term in expansions:
            lower = term.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(term)
        return " ".join(unique)


class VertexQueryExpander(BaseQueryExpander):
    """
    Query expander using Vertex AI GenerativeModel (legacy SDK).

    NOTE: vertexai.generative_models was deprecated on June 24, 2025 and will
    be removed on June 24, 2026. This class exists for mock testing as required
    by the assessment spec. For new production code, use GenAIQueryExpander below.
    """

    SYSTEM_PROMPT = (
        "You are a search query optimizer. Given a user question about distributed systems, "
        "rewrite it to include relevant technical synonyms and context that would improve "
        "semantic search retrieval. Return only the expanded query, no explanation."
    )

    def __init__(self, model_name: str = "gemini-2.5-flash-001") -> None:
        from vertexai.generative_models import GenerativeModel

        self._model = GenerativeModel(
            model_name=model_name,
            system_instruction=self.SYSTEM_PROMPT,
        )

    def expand(self, query: str) -> str:
        response = self._model.generate_content(query)
        return response.text


class GenAIQueryExpander(BaseQueryExpander):
    """
    Production query expander using the recommended Google Gen AI SDK (google-genai).

    This is the forward-looking replacement for VertexQueryExpander.
    """

    SYSTEM_PROMPT = (
        "You are a search query optimizer. Given a user question about distributed systems, "
        "rewrite it to include relevant technical synonyms and context that would improve "
        "semantic search retrieval. Return only the expanded query, no explanation."
    )

    def __init__(
        self,
        project: str = "my-project",
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash-001",
    ) -> None:
        from google import genai

        self._client = genai.Client(
            vertexai=True, project=project, location=location
        )
        self._model_name = model_name

    def expand(self, query: str) -> str:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=query,
            config={"system_instruction": self.SYSTEM_PROMPT},
        )
        return response.text
