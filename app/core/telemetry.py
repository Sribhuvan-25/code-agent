"""
Telemetry and observability utilities for the Backspace Coding Agent.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import structlog

from app.core.config import settings

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


# Configure structured logging
def configure_logging() -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Configure OpenTelemetry
def configure_tracing() -> None:
    """Configure OpenTelemetry tracing."""
    if not OTEL_AVAILABLE:
        return
        
    if not settings.enable_tracing:
        return
    
    resource = Resource(attributes={SERVICE_NAME: settings.app_name})
    provider = TracerProvider(resource=resource)
    
    if settings.jaeger_endpoint:
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name=settings.jaeger_endpoint.split(":")[0],
                agent_port=int(settings.jaeger_endpoint.split(":")[1]) if ":" in settings.jaeger_endpoint else 14268,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            provider.add_span_processor(span_processor)
        except Exception:
            pass
    
    trace.set_tracer_provider(provider)


# Initialize telemetry
configure_logging()
configure_tracing()

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__) if OTEL_AVAILABLE and trace else None


class TelemetryManager:
    """Manages telemetry for the application."""
    
    def __init__(self):
        self.logger = logger
        self.tracer = tracer
        self.metrics = {}
    
    def log_event(
        self,
        event: str,
        level: str = "info",
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log an event with structured logging.
        
        Args:
            event: The event name
            level: Log level
            correlation_id: Correlation ID for request tracing
            **kwargs: Additional context
        """
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(
            event,
            correlation_id=correlation_id,
            timestamp=time.time(),
            **kwargs,
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Log an error with full context.
        
        Args:
            error: The exception to log
            context: Additional context
            correlation_id: Correlation ID for request tracing
        """
        self.logger.error(
            "Error occurred",
            error=str(error),
            error_type=type(error).__name__,
            context=context or {},
            correlation_id=correlation_id,
            timestamp=time.time(),
            exc_info=True,
        )
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: The operation name
            duration: Duration in seconds
            correlation_id: Correlation ID for request tracing
            **kwargs: Additional metrics
        """
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration=duration,
            correlation_id=correlation_id,
            timestamp=time.time(),
            **kwargs,
        )
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes: Any):
        """
        Context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            **attributes: Additional attributes for the span
        """
        start_time = time.time()
        
        if self.tracer and OTEL_AVAILABLE:
            with self.tracer.start_as_current_span(operation_name) as span:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                
                try:
                    yield span
                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("duration", duration)
                    self.log_performance(operation_name, duration)
        else:
            try:
                yield None
            finally:
                duration = time.time() - start_time
                self.log_performance(operation_name, duration)
    
    def increment_counter(self, metric_name: str, value: int = 1, **labels: Any) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
            **labels: Additional labels
        """
        key = f"{metric_name}_{hash(frozenset(labels.items()))}"
        self.metrics[key] = self.metrics.get(key, 0) + value
        
        self.logger.info(
            "Metric incremented",
            metric=metric_name,
            value=value,
            labels=labels,
            timestamp=time.time(),
        )
    
    def record_histogram(self, metric_name: str, value: float, **labels: Any) -> None:
        """
        Record a histogram metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            **labels: Additional labels
        """
        self.logger.info(
            "Histogram recorded",
            metric=metric_name,
            value=value,
            labels=labels,
            timestamp=time.time(),
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

telemetry = TelemetryManager()


def get_telemetry() -> TelemetryManager:
    """Get the global telemetry manager."""
    return telemetry 