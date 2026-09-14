from __future__ import annotations

from typing import Any

from .contracts import DiagnosticRequest
from .domains import DOMAIN_PACKS
from .domains.runners import WindScadaDiagnosticPipeline
from .execution import DomainInputSchema, InputField, StepExecutionError, WorkflowExecutor
from .models import DetectionResult, DiagnosticResult, ToolTraceStep
from .registry import domain_registry, model_registry, policy_registry
from .result_contract import standardize_result

PIPELINE_VERSION = "1.0.0"

DOMAIN_INPUT_SCHEMAS = {
    "bearing": DomainInputSchema("bearing", (InputField("signal"), InputField("sampling_rate_hz"), InputField("fault_frequencies", False))),
    "process": DomainInputSchema("process", (InputField("signal_matrix"), InputField("normal_reference"), InputField("channel_names"), InputField("sampling_rate_hz", False))),
    "wind_scada": DomainInputSchema("wind_scada", (InputField("signal_matrix"), InputField("channel_names"), InputField("timestamps"), InputField("normal_reference", False))),
    "battery": DomainInputSchema("battery", (InputField("cell_voltage"), InputField("cell_temperature"), InputField("cell_ids"), InputField("timestamps"))),
    "turbofan": DomainInputSchema("turbofan", (InputField("signal_matrix"), InputField("channel_names"), InputField("cycle_index"))),
    "transformer": DomainInputSchema("transformer", (InputField("signal_matrix"), InputField("sampling_rate_hz"), InputField("sensor_positions"), InputField("trained_image_model", False))),
}


def get_input_schema(domain: str) -> DomainInputSchema:
    try:
        return DOMAIN_INPUT_SCHEMAS[domain]
    except KeyError as exc:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_INPUT_SCHEMAS)}") from exc


def _abstain(domain: str, task: str, reason: str, trace: list[ToolTraceStep] | None = None) -> DiagnosticResult:
    return standardize_result(DiagnosticResult(
        domain=domain,
        task=task,
        decision="abstain",
        detection=DetectionResult(abnormal=None, method="pipeline_validation"),
        confidence=0.0,
        uncertainty=1.0,
        abstained=True,
        abstain_reason=reason,
        tool_trace=trace or [ToolTraceStep("input_validation", status="warning", details={"reason": reason})],
        metadata={"pipeline_version": PIPELINE_VERSION},
    ), pipeline_version=PIPELINE_VERSION)


def _with_task(result: DiagnosticResult, task: str | None) -> DiagnosticResult:
    if task is not None:
        result.task = task
    return standardize_result(result, pipeline_version=PIPELINE_VERSION)


def _ensure_plugins() -> None:
    from .domains.battery_plugin import BatteryPlugin
    from .domains.compat_plugins import BearingPlugin, ProcessPlugin, TransformerPlugin, TurbofanPlugin
    from .domains.wind_scada_plugin import WindScadaPlugin

    plugins = (
        BearingPlugin(),
        ProcessPlugin(),
        WindScadaPlugin(),
        BatteryPlugin(),
        TurbofanPlugin(),
        TransformerPlugin(),
    )
    for plugin in plugins:
        if domain_registry.get(plugin.name) is None:
            domain_registry.register(plugin)

        probe = plugin.policy(DiagnosticRequest(domain=plugin.name, task=None, inputs={}))
        factory = lambda request, p=plugin: p.policy(request)
        policy_registry.register(plugin.name, "default", factory)
        policy_registry.register(plugin.name, str(probe.version), factory)

    # Battery exposes a second versioned policy for the capacity-history
    # prognosis task. Register it explicitly because the default probe above is
    # intentionally the pack-diagnostic policy.
    battery_plugin = domain_registry.resolve("battery")
    prognosis_probe_request = DiagnosticRequest(domain="battery", task="prognosis", inputs={})
    prognosis_probe = battery_plugin.policy(prognosis_probe_request)
    policy_registry.register(
        "battery",
        str(prognosis_probe.version),
        lambda request, p=battery_plugin: p.policy(request),
    )


def _diagnose_request(request: DiagnosticRequest) -> DiagnosticResult:
    _ensure_plugins()
    task = str(request.task or "diagnosis")
    try:
        plugin = domain_registry.resolve(request.domain)
        validated = dict(plugin.validate(request))
        execution, trace = WorkflowExecutor().run(plugin.workflow(request), validated)
        if request.policy_ref:
            policy = policy_registry.resolve(request.domain, request.policy_ref)(request)
        else:
            policy = plugin.policy(request)
        result = policy.decide(execution, trace)
        if request.task is not None:
            result.task = request.task
        result.metadata.setdefault("pipeline_version", PIPELINE_VERSION)
        if request.run_context is not None:
            result.metadata.setdefault("run_context", {
                "run_id": request.run_context.run_id,
                "source": request.run_context.source,
                "metadata": dict(request.run_context.metadata),
            })
        if request.model_refs:
            result.metadata.setdefault("model_refs", dict(request.model_refs))
            resolved_versions = {
                slot: record.version
                for slot, ref in request.model_refs.items()
                if (record := model_registry.get(ref)) is not None
            }
            if resolved_versions:
                result.metadata.setdefault("resolved_model_versions", resolved_versions)
        if request.policy_ref:
            result.metadata.setdefault("policy_ref", request.policy_ref)
        return standardize_result(result, pipeline_version=PIPELINE_VERSION)
    except StepExecutionError as exc:
        return _abstain(request.domain, task, f"Pipeline step failed: {exc}", exc.trace)
    except (TypeError, ValueError, KeyError) as exc:
        return _abstain(request.domain, task, f"Input validation failed: {exc}")
    except Exception as exc:
        return _abstain(request.domain, task, f"Pipeline execution failed in {request.domain}: {exc}")


class DiagnosticPipeline:
    """Stable public dispatcher for all supported industrial domain pipelines.

    Five domains now use the structured plugin boundary internally. The original
    Wind SCADA call shape remains a compatibility adapter because the richer Wind
    workflow requires separate healthy-training and prediction matrices. CARE and
    new callers use ``DiagnosticRequest`` and therefore exercise the canonical
    Wind plugin path.
    """

    version = PIPELINE_VERSION

    def run(self, domain: str, *, task: str | None = None, metadata: dict[str, Any] | None = None, **inputs) -> DiagnosticResult:
        domain = str(domain).strip().lower()
        if domain not in DOMAIN_PACKS:
            raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_PACKS)}")
        values = dict(metadata or {})
        values.update(inputs)
        default_task = str(task or DOMAIN_PACKS[domain].tasks[0].value)
        supported_tasks = {row.value for row in DOMAIN_PACKS[domain].tasks}
        if task is not None and task not in supported_tasks:
            return _abstain(domain, default_task, f"Unsupported task {task!r}; available tasks: {sorted(supported_tasks)}")

        missing = [key for key in DOMAIN_PACKS[domain].required_metadata if values.get(key) is None]
        # Capacity-history prognosis is a distinct battery input contract and
        # therefore does not require pack cell_ids/timestamps.
        if domain == "battery" and default_task == "prognosis" and values.get("cycle_index") is not None:
            missing = []
        if missing:
            return _abstain(domain, default_task, f"Missing required metadata: {', '.join(missing)}")

        schema_values = dict(values)
        if domain == "bearing" and schema_values.get("signal") is None:
            schema_values["signal"] = schema_values.get("signal_matrix")
        if not (domain == "battery" and default_task == "prognosis"):
            try:
                get_input_schema(domain).validate(schema_values)
            except (TypeError, ValueError) as exc:
                return _abstain(domain, default_task, f"Input validation failed: {exc}")

        if domain == "wind_scada":
            try:
                return _with_task(WindScadaDiagnosticPipeline().run(**values), task)
            except StepExecutionError as exc:
                return _abstain(domain, default_task, f"Pipeline step failed: {exc}", exc.trace)
            except (TypeError, ValueError) as exc:
                return _abstain(domain, default_task, f"Input validation failed: {exc}")
            except Exception as exc:
                return _abstain(domain, default_task, f"Pipeline execution failed in {domain}: {exc}")

        return _diagnose_request(DiagnosticRequest(
            domain=domain,
            task=default_task,
            inputs=values,
        ))


def diagnose(
    request_or_domain: DiagnosticRequest | str,
    *,
    task: str | None = None,
    metadata: dict[str, Any] | None = None,
    **inputs,
) -> DiagnosticResult:
    """Run a structured plugin request or the backwards-compatible legacy API."""
    if isinstance(request_or_domain, DiagnosticRequest):
        if task is not None or metadata is not None or inputs:
            raise TypeError("task/metadata/inputs cannot be combined with DiagnosticRequest")
        return _diagnose_request(request_or_domain)
    return DiagnosticPipeline().run(str(request_or_domain), task=task, metadata=metadata, **inputs)
