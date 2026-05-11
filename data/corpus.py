from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)


CORPUS_DOCUMENTS: list[Document] = [
    Document(
        id="doc_scaling",
        title="Horizontal Scaling and Autoscaling",
        content=(
            "ArcScale employs horizontal scaling as its primary strategy for handling increased workloads. "
            "The platform runs stateless application instances behind an autoscaler that monitors CPU utilization, "
            "request queue depth, and p99 latency. When any metric breaches its threshold for 60 seconds, the "
            "autoscaler provisions additional instances within the same availability zone. Scale-down follows a "
            "cooldown period of 5 minutes to prevent thrashing. Each instance registers with the service registry "
            "on startup and deregisters gracefully during shutdown, ensuring zero-downtime scaling events."
        ),
        tags=["scaling", "autoscaling", "elasticity"],
    ),
    Document(
        id="doc_load_balancing",
        title="Load Balancing Strategies",
        content=(
            "Traffic distribution in ArcScale uses a tiered load balancing approach. The edge layer employs "
            "round-robin DNS across regional entry points. Behind each entry point, an L7 load balancer applies "
            "least-connections routing to distribute requests among healthy instances. For stateful operations "
            "such as WebSocket sessions, consistent hashing pins clients to specific backends based on a session "
            "token. Health checks run every 10 seconds; instances failing two consecutive checks are removed from "
            "the rotation. The balancer also supports weighted routing for canary traffic splitting."
        ),
        tags=["load-balancing", "routing", "traffic"],
    ),
    Document(
        id="doc_caching",
        title="Distributed Caching and Cache Invalidation",
        content=(
            "ArcScale uses a two-tier caching architecture with an in-process LRU cache and a shared Redis cluster. "
            "Frequently accessed configuration and reference data is cached locally with a 30-second TTL, while "
            "session state and computed results reside in Redis with configurable expiry. Cache invalidation follows "
            "a publish-subscribe pattern: when a source record changes, an invalidation event is broadcast to all "
            "nodes, which then evict the stale entry. Write-through caching is used for critical paths to ensure "
            "consistency, while read-heavy endpoints use cache-aside with lazy repopulation."
        ),
        tags=["caching", "redis", "invalidation"],
    ),
    Document(
        id="doc_circuit_breaker",
        title="Circuit Breaker and Fault Tolerance",
        content=(
            "To prevent cascading failures, ArcScale implements the circuit breaker pattern on all inter-service "
            "calls. Each circuit tracks failure rates over a sliding 30-second window. When failures exceed 50%, "
            "the circuit opens and subsequent requests receive an immediate fallback response instead of waiting "
            "for timeouts. After a configurable recovery interval, the circuit enters a half-open state, allowing "
            "a single probe request through. If the probe succeeds, the circuit closes; otherwise, it reopens. "
            "Bulkhead isolation limits concurrency per downstream dependency to prevent thread pool exhaustion."
        ),
        tags=["fault-tolerance", "circuit-breaker", "resilience"],
    ),
    Document(
        id="doc_observability",
        title="Observability: Metrics, Tracing, and Logging",
        content=(
            "ArcScale's observability stack consists of three pillars. Structured JSON logs are emitted by every "
            "service and shipped to a centralized log aggregator with correlation IDs for request tracing. "
            "Distributed traces propagate context across service boundaries using W3C Trace Context headers, "
            "enabling end-to-end latency analysis. Prometheus-compatible metrics expose request rates, error "
            "rates, and saturation indicators per service. Alert rules trigger when error budgets are depleted "
            "or when p99 latency exceeds SLO thresholds, notifying the on-call engineer via PagerDuty."
        ),
        tags=["observability", "monitoring", "logging", "tracing"],
    ),
    Document(
        id="doc_service_mesh",
        title="Service Mesh and Mutual TLS",
        content=(
            "Inter-service communication in ArcScale is mediated by a service mesh that provides transparent "
            "mTLS encryption, traffic management, and observability. Each pod runs a sidecar proxy that handles "
            "TLS termination, certificate rotation, and request-level authorization policies. The mesh control "
            "plane manages service identity through SPIFFE-compatible certificates issued by an internal CA. "
            "Traffic policies enforce retry budgets, request timeouts, and circuit breaking at the mesh layer, "
            "decoupling resilience logic from application code."
        ),
        tags=["service-mesh", "security", "mtls"],
    ),
    Document(
        id="doc_event_driven",
        title="Event-Driven Architecture with Kafka",
        content=(
            "ArcScale's asynchronous workflows are built on an event-driven architecture using Apache Kafka as "
            "the message backbone. Domain events are published to topic partitions keyed by aggregate ID, "
            "ensuring ordered processing within a partition. Consumer groups enable parallel processing with "
            "automatic partition rebalancing. Dead-letter topics capture failed messages after three retry "
            "attempts, with automated alerting on DLQ growth. Schema evolution is managed through a schema "
            "registry enforcing backward compatibility, preventing producers from breaking existing consumers."
        ),
        tags=["event-driven", "kafka", "messaging"],
    ),
    Document(
        id="doc_database",
        title="Database Sharding and Replication",
        content=(
            "ArcScale's persistence layer uses range-based sharding across a PostgreSQL cluster to distribute "
            "write load. Each shard maintains a primary with two synchronous replicas for durability and one "
            "asynchronous replica for read scaling. Cross-shard queries are minimized through careful data "
            "modeling that co-locates related entities on the same shard key. Schema migrations run as "
            "non-blocking online DDL operations to avoid table locks. A connection pooler sits between "
            "application instances and database shards, limiting total connection count and providing "
            "transparent failover when a primary becomes unavailable."
        ),
        tags=["database", "sharding", "replication"],
    ),
    Document(
        id="doc_deployments",
        title="Blue-Green Deployments and Canary Releases",
        content=(
            "ArcScale follows a blue-green deployment model where two identical production environments "
            "exist in parallel. New releases are deployed to the inactive environment, validated through "
            "automated smoke tests and synthetic monitoring, then traffic is switched via the load balancer. "
            "For high-risk changes, a canary release strategy routes 5% of production traffic to the new "
            "version while monitoring error rates and latency. Automatic rollback triggers if the canary's "
            "error rate exceeds the baseline by more than 1%. The entire deployment pipeline is codified "
            "in a CI/CD system with mandatory approval gates for production promotions."
        ),
        tags=["deployment", "blue-green", "canary", "ci-cd"],
    ),
    Document(
        id="doc_rate_limiting",
        title="Rate Limiting and Backpressure",
        content=(
            "ArcScale applies rate limiting at multiple layers to protect system stability. The API gateway "
            "enforces per-client token bucket rate limits based on subscription tier. Internally, services "
            "implement adaptive concurrency limits that automatically adjust based on measured latency: as "
            "latency increases, the concurrency window shrinks, applying backpressure to upstream callers. "
            "When the system approaches saturation, non-critical background tasks are shed first via priority "
            "queues. Load shedding decisions are propagated through response headers, allowing clients to "
            "implement exponential backoff with jitter."
        ),
        tags=["rate-limiting", "backpressure", "throttling"],
    ),
]
