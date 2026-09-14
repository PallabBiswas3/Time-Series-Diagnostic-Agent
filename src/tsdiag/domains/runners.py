from __future__ import annotations

from typing import Any

import numpy as np

from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    PrognosisResult,
    ToolTraceStep,
    VerificationResult,
)
from ..tools.change import change_point_detection, rolling_statistics
from ..tools.wavelet import (
    correlation_sensor_weighting,
    cross_correlation_analysis,
    multisensor_fusion,
    wavelet_denoising,
)


def _matrix(value, name: str, *, minimum_rows: int = 8) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < minimum_rows:
        raise ValueError(f"{name} must be [samples, channels] with at least {minimum_rows} samples")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return x


def _names(values, width: int, name: str) -> list[str]:
    rows = [str(v) for v in values]
    if len(rows) != width:
        raise ValueError(f"{name} length must match the number of channels")
    return rows


def _finish(result: DiagnosticResult) -> DiagnosticResult:
    result.metadata.setdefault("pipeline_version", "1.0.0")
    result.validate()
    return result


class WindScadaDiagnosticPipeline:
    """Deterministic SCADA baseline using reference residuals and persistence."""

    def run(self, signal_matrix, channel_names, timestamps, *, normal_reference=None, **context) -> DiagnosticResult:
        x = _matrix(signal_matrix, "signal_matrix")
        names = _names(channel_names, x.shape[1], "channel_names")
        ts = np.asarray(timestamps)
        if ts.ndim != 1 or len(ts) != len(x):
            raise ValueError("timestamps must contain one value per sample")
        ref = _matrix(normal_reference, "normal_reference") if normal_reference is not None else x[: max(8, len(x) // 4)]
        if ref.shape[1] != x.shape[1]:
            raise ValueError("normal_reference channels must match signal_matrix")

        center = np.median(ref, axis=0)
        scale = np.median(np.abs(ref - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-9, scale, np.where(ref.std(0) > 1e-9, ref.std(0), 1.0))
        physics_model = context.get("physics_model")
        if callable(physics_model):
            predicted = physics_model(signal_matrix=x, channel_names=names, timestamps=ts)
            expected = np.asarray(predicted.get("expected_signal") if isinstance(predicted, dict) else predicted, dtype=float)
            if expected.shape != x.shape:
                raise ValueError("physics_model expected_signal must match signal_matrix")
            residual = (x - expected) / scale
            reference_source = "physics_model"
        else:
            residual = (x - center) / scale
            reference_source = "provided" if normal_reference is not None else "initial_segment"
        sample_score = np.max(np.abs(residual), axis=1)
        alarm = sample_score > float(context.get("z_threshold", 4.0))
        alarm_fraction = float(np.mean(alarm))
        channel_scores = np.quantile(np.abs(residual), 0.95, axis=0)
        top = int(np.argmax(channel_scores))
        abnormal = alarm_fraction >= float(context.get("minimum_alarm_fraction", 0.05))
        confidence = float(np.clip(max(alarm_fraction * 4.0, channel_scores[top] / 8.0), 0.05, 0.95))
        stats = rolling_statistics(residual, ts, window=min(max(3, len(x) // 10), 20))
        changes = change_point_detection(residual, ts, min_size=max(3, min(20, len(x) // 4)))
        ev = Evidence("scada_residuals", f"{alarm_fraction:.1%} of samples exceed the residual limit; strongest channel is {names[top]}.", confidence,
                      {"channel_scores": dict(zip(names, channel_scores.tolist())), "change_points": changes["change_points"]},
                      "wind-residual-evidence", "statistical")
        return _finish(DiagnosticResult(
            domain="wind_scada", task="condition_monitoring", decision="diagnose" if abnormal else "monitor",
            detection=DetectionResult(abnormal=abnormal, score=confidence, method="robust_reference_residual", details={"alarm_fraction": alarm_fraction}),
            localization=LocalizationResult(channels=[names[top]] if abnormal else [], scores={names[top]: confidence} if abnormal else {}),
            hypotheses=[DiagnosticHypothesis("persistent_scada_anomaly", confidence, evidence_ids=[ev.evidence_id])] if abnormal else [],
            evidence=[ev], confidence=confidence, uncertainty=1.0-confidence,
            recommended_actions=["Inspect the localized SCADA channel and its operating regime."] if abnormal else [],
            tool_trace=[ToolTraceStep("scada_quality_check"), ToolTraceStep("operating_regime_detection"), ToolTraceStep("normal_behavior_model"), ToolTraceStep("residual_analysis", evidence_ids=[ev.evidence_id]), ToolTraceStep("rolling_statistics"), ToolTraceStep("cross_sensor_relationships"), ToolTraceStep("scada_anomaly_detection"), ToolTraceStep("change_point_detection"), ToolTraceStep("physics_consistency_check", status="ok" if physics_model else "skipped"), ToolTraceStep("scada_risk_assessment")],
            verification=[VerificationResult("persistent_scada_anomaly", "SUPPORTED", "Residual anomaly persisted relative to the expected behavior model.", [ev.evidence_id], "scada_residual_check")] if abnormal else [],
            metadata={"rolling_statistics": stats, "change_points": changes, "reference_source": reference_source},
        ))


class BatteryDiagnosticPipeline:
    """Cell-to-pack deviation baseline for battery localization and risk."""

    def run(self, cell_voltage, cell_temperature, cell_ids, timestamps, **context) -> DiagnosticResult:
        voltage = _matrix(cell_voltage, "cell_voltage")
        temperature = _matrix(cell_temperature, "cell_temperature")
        if voltage.shape != temperature.shape:
            raise ValueError("cell_voltage and cell_temperature must have matching shapes")
        ids = _names(cell_ids, voltage.shape[1], "cell_ids")
        if len(np.asarray(timestamps)) != len(voltage):
            raise ValueError("timestamps must contain one value per sample")

        def deviations(x):
            delta = x - np.median(x, axis=1, keepdims=True)
            offset = np.median(delta, axis=0)
            centered = delta - offset
            per_cell_noise = np.median(np.abs(centered), axis=0) * 1.4826
            noise = float(np.median(per_cell_noise[per_cell_noise > 1e-9])) if np.any(per_cell_noise > 1e-9) else float(np.std(centered) + 1e-6)
            scores = np.abs(offset) / max(noise, 1e-6)
            return delta, scores
        dv, sv = deviations(voltage)
        dt, st = deviations(temperature)
        scores = np.maximum(sv, st)
        top = int(np.argmax(scores))
        threshold = float(context.get("cell_anomaly_threshold", 3.5))
        abnormal = bool(scores[top] >= threshold)
        confidence = float(np.clip(scores[top] / (threshold * 1.5), 0.05, 0.95))
        prognostic_model = context.get("trained_prognostic_model")
        prediction: dict[str, Any] = {}
        if callable(prognostic_model):
            raw_prediction = prognostic_model(cell_voltage=voltage, cell_temperature=temperature, timestamps=np.asarray(timestamps))
            prediction = raw_prediction if isinstance(raw_prediction, dict) else {"risk": float(raw_prediction)}
        risk = float(np.clip(prediction.get("risk", np.max(np.maximum(temperature - 50.0, 0.0)) / 20.0 + confidence * 0.5), 0.0, 1.0))
        ev = Evidence("cell_deviation", f"Largest cell-to-pack deviation is {ids[top]} (score={scores[top]:.2f}).", confidence,
                      {"cell_scores": dict(zip(ids, scores.tolist()))}, "battery-cell-evidence", "statistical")
        return _finish(DiagnosticResult(
            domain="battery", task="anomaly_localization", decision="diagnose" if abnormal else "monitor",
            detection=DetectionResult(abnormal=abnormal, score=confidence, method="robust_cell_to_pack_deviation"),
            localization=LocalizationResult(components=[ids[top]] if abnormal else [], channels=[ids[top]] if abnormal else [], scores={ids[top]: confidence} if abnormal else {}),
            hypotheses=[DiagnosticHypothesis("cell_imbalance", confidence, evidence_ids=[ev.evidence_id])] if abnormal else [],
            evidence=[ev], prognosis=PrognosisResult(risk=risk, horizon=prediction.get("horizon"), uncertainty=float(prediction.get("uncertainty", 1.0-confidence)), details={"model": "trained" if prediction else "deterministic_risk_baseline"}),
            confidence=confidence, uncertainty=1.0-confidence,
            recommended_actions=[f"Inspect {ids[top]} voltage and temperature sensing and balance state."] if abnormal else [],
            tool_trace=[ToolTraceStep("battery_data_quality"), ToolTraceStep("operating_state_segmentation"), ToolTraceStep("cell_deviation_features", evidence_ids=[ev.evidence_id]), ToolTraceStep("battery_temporal_features"), ToolTraceStep("cell_spatial_consistency"), ToolTraceStep("adaptive_window_builder"), ToolTraceStep("battery_spatiotemporal_model", status="ok" if prognostic_model else "skipped"), ToolTraceStep("battery_anomaly_localization"), ToolTraceStep("battery_failure_prognosis"), ToolTraceStep("battery_decision")],
            metadata={"delta_voltage": dv, "delta_temperature": dt, "cell_scores": dict(zip(ids, scores.tolist()))},
        ))


class TurbofanDiagnosticPipeline:
    """Interpretable degradation and RUL baseline for run-to-failure series."""

    def run(self, signal_matrix, channel_names, cycle_index, **context) -> DiagnosticResult:
        x = _matrix(signal_matrix, "signal_matrix")
        names = _names(channel_names, x.shape[1], "channel_names")
        cycles = np.asarray(cycle_index, dtype=float)
        if cycles.ndim != 1 or len(cycles) != len(x) or np.any(np.diff(cycles) <= 0):
            raise ValueError("cycle_index must be strictly increasing with one value per sample")
        span = max(float(np.ptp(cycles)), 1.0)
        scaled_x = (x - np.median(x, axis=0)) / (np.std(x, axis=0) + 1e-9)
        slopes = np.array([np.polyfit(cycles, scaled_x[:, j], 1)[0] * span for j in range(x.shape[1])])
        selected = np.flatnonzero(np.abs(slopes) >= max(0.1, np.quantile(np.abs(slopes), 0.5)))
        if selected.size == 0:
            selected = np.array([int(np.argmax(np.abs(slopes)))])
        direction = np.sign(slopes[selected])
        health = np.mean(scaled_x[:, selected] * direction, axis=1)
        health -= health[0]
        health_slope, intercept = np.polyfit(cycles, health, 1)
        current_health = float(np.mean(health[-max(3, len(health)//10):]))
        threshold = float(context.get("failure_threshold", 3.0))
        predictor = context.get("trained_rul_model")
        if callable(predictor):
            prediction = predictor(signal_matrix=x, cycle_index=cycles, health_index=health)
            rul = float(prediction["rul_cycles"] if isinstance(prediction, dict) else prediction)
            method = "trained_rul_model"
        elif health_slope > 1e-9:
            rul = max(0.0, float((threshold - intercept) / health_slope - cycles[-1]))
            method = "linear_health_index"
        else:
            rul = None
            method = "linear_health_index"
        residual = health - (health_slope * cycles + intercept)
        error = float(np.sqrt(np.mean(residual**2)))
        confidence = float(np.clip(1.0 / (1.0 + error), 0.1, 0.9))
        degraded = bool(current_health >= float(context.get("degradation_threshold", 1.0)) and health_slope * span >= 1.0)
        ev = Evidence("health_index", f"Health index={current_health:.2f}; trend={health_slope:.4g} per cycle.", confidence,
                      {"selected_channels": [names[i] for i in selected], "rul_cycles": rul}, "turbofan-health-evidence", "prognostic")
        return _finish(DiagnosticResult(
            domain="turbofan", task="remaining_useful_life", decision="diagnose" if degraded else "monitor",
            detection=DetectionResult(abnormal=degraded, score=float(np.clip(max(current_health, 0)/threshold, 0, 1)), method="health_index_trend"),
            localization=LocalizationResult(channels=[names[i] for i in selected], scores={names[i]: float(min(abs(slopes[i]), 1.0)) for i in selected}),
            hypotheses=[DiagnosticHypothesis("degradation", confidence, evidence_ids=[ev.evidence_id])] if degraded else [], evidence=[ev],
            prognosis=PrognosisResult(remaining_useful_life=rul, horizon="cycles", uncertainty=1.0-confidence, details={"method": method}),
            confidence=confidence, uncertainty=1.0-confidence,
            recommended_actions=["Track the health-index trend and inspect the critical sensors."] if degraded else [],
            tool_trace=[ToolTraceStep("sensor_screening"), ToolTraceStep("operating_condition_identification"), ToolTraceStep("regime_normalization"), ToolTraceStep("degradation_smoothing"), ToolTraceStep("sequence_windowing"), ToolTraceStep("health_index_estimation", evidence_ids=[ev.evidence_id]), ToolTraceStep("rul_prediction"), ToolTraceStep("rul_uncertainty"), ToolTraceStep("prognostic_explanation"), ToolTraceStep("turbofan_decision")],
            metadata={"health_index": health, "sensor_slopes": dict(zip(names, slopes.tolist()))},
        ))


class TransformerDiagnosticPipeline:
    """Multisensor fusion baseline with an optional fault classifier."""

    def run(self, signal_matrix, sampling_rate_hz, sensor_positions, **context) -> DiagnosticResult:
        x = _matrix(signal_matrix, "signal_matrix", minimum_rows=32)
        positions = _names(sensor_positions, x.shape[1], "sensor_positions")
        fs = float(sampling_rate_hz)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        clean = wavelet_denoising(x, context.get("wavelet", "db4"))
        corr = cross_correlation_analysis(clean["denoised_signal_matrix"])
        weights = correlation_sensor_weighting(corr["correlation_energy"])["sensor_weights"]
        fused = multisensor_fusion(clean["denoised_signal_matrix"], weights)["fused_waveform"]
        centered = fused - fused.mean()
        variance = float(np.mean(centered**2))
        kurtosis = float(np.mean(centered**4) / (variance**2 + 1e-12))
        score = float(np.clip(max(kurtosis - 3.0, 0.0) / 7.0, 0.0, 1.0))
        abnormal = score >= float(context.get("anomaly_threshold", 0.25))
        width = min(256, len(centered))
        hop = max(1, width // 2)
        windows = [centered[i:i+width] for i in range(0, len(centered)-width+1, hop)]
        spectrum = np.stack([np.abs(np.fft.rfft(row * np.hanning(width))) for row in windows], axis=1)
        spectral_correlation = np.abs(np.diff(spectrum, axis=1, prepend=spectrum[:, :1]))
        feature_image = np.log1p(spectral_correlation)
        feature_image /= max(float(np.max(feature_image)), 1e-12)
        predictor = context.get("trained_image_model")
        output: dict[str, Any] = predictor(feature_image) if callable(predictor) else {}
        label = output.get("label")
        model_confidence = float(output.get("confidence", score))
        abstained = bool(abnormal and not label)
        decision = "abstain" if abstained else ("diagnose" if abnormal and label else "monitor")
        confidence = float(np.clip(model_confidence if label else max(0.1, score), 0.0, 1.0))
        ev = Evidence("multisensor_fusion", f"Fused-waveform kurtosis={kurtosis:.2f}; classifier label={label!r}.", confidence,
                      {"sensor_weights": dict(zip(positions, weights.tolist())), "model_output": output}, "transformer-fusion-evidence", "signal")
        return _finish(DiagnosticResult(
            domain="transformer", task="fault_diagnosis", decision=decision,
            detection=DetectionResult(abnormal=abnormal, score=score, method="wavelet_multisensor_impulsiveness"),
            localization=LocalizationResult(channels=[positions[int(np.argmax(weights))]], scores=dict(zip(positions, weights.tolist()))),
            hypotheses=[DiagnosticHypothesis(str(label), confidence, evidence_ids=[ev.evidence_id])] if label else [], evidence=[ev],
            confidence=confidence, uncertainty=1.0-confidence, abstained=abstained,
            abstain_reason="Abnormal multisensor evidence requires a trained fault classifier for labeling." if abstained else None,
            recommended_actions=["Supply a validated transformer fault classifier to label the detected condition."] if abstained else [],
            tool_trace=[ToolTraceStep("multisensor_sync_check"), ToolTraceStep("wavelet_denoising"), ToolTraceStep("cross_correlation_analysis"), ToolTraceStep("correlation_sensor_weighting"), ToolTraceStep("multisensor_fusion", evidence_ids=[ev.evidence_id]), ToolTraceStep("envelope_sanity_check"), ToolTraceStep("fast_spectral_correlation"), ToolTraceStep("spectral_correlation_representation"), ToolTraceStep("transformer_fault_classifier", status="ok" if label else "skipped"), ToolTraceStep("transformer_decision")],
            metadata={"kurtosis": kurtosis, "sensor_weights": dict(zip(positions, weights.tolist())), "feature_image_shape": list(feature_image.shape), "model_output": output},
        ))
