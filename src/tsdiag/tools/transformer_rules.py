from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class TransformerRuleReference:
    center: dict[str, float]
    scale: dict[str, float]


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2, axis=0))


def _fundamental_phasors(abc: np.ndarray) -> np.ndarray:
    x = np.asarray(abc, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    if len(fft) <= 1:
        return np.zeros(3, dtype=complex)
    energy = np.sum(np.abs(fft[1:]) ** 2, axis=1)
    k = int(np.argmax(energy)) + 1
    return fft[k] / max(len(x), 1)


def _sequence(phasors: np.ndarray) -> tuple[complex, complex, complex]:
    a = np.exp(1j * 2 * np.pi / 3)
    aa, ab, ac = phasors
    zero = (aa + ab + ac) / 3
    positive = (aa + a * ab + a**2 * ac) / 3
    negative = (aa + a**2 * ab + a * ac) / 3
    return zero, positive, negative


def _thd(abc: np.ndarray) -> float:
    x = np.asarray(abc, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    fft = np.abs(np.fft.rfft(x, axis=0))
    if len(fft) <= 2:
        return 0.0
    joint = np.sum(fft[1:] ** 2, axis=1)
    k = int(np.argmax(joint)) + 1
    total = np.sum(fft[1:] ** 2, axis=0)
    fundamental = fft[k] ** 2
    harmonic = np.maximum(total - fundamental, 0)
    return float(np.mean(np.sqrt(harmonic / (fundamental + _EPS))))


def transformer_electrical_features(signal_matrix) -> dict[str, float]:
    x = np.asarray(signal_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] < 6 or len(x) < 16 or not np.all(np.isfinite(x)):
        raise ValueError("transformer electrical rules require finite [samples, >=6] Ua,Ub,Uc,Ia,Ib,Ic data")
    v = x[:, :3]
    i = x[:, 3:6]

    vrms = _rms(v)
    irms = _rms(i)
    vp = _fundamental_phasors(v)
    ip = _fundamental_phasors(i)
    v0, v1, v2 = _sequence(vp)
    i0, i1, i2 = _sequence(ip)

    v_mean = float(np.mean(vrms))
    i_mean = float(np.mean(irms))
    v_imb = float(np.std(vrms) / (v_mean + _EPS))
    i_imb = float(np.std(irms) / (i_mean + _EPS))
    v0_ratio = float(abs(v0) / (abs(v1) + _EPS))
    v2_ratio = float(abs(v2) / (abs(v1) + _EPS))
    i0_ratio = float(abs(i0) / (abs(i1) + _EPS))
    i2_ratio = float(abs(i2) / (abs(i1) + _EPS))

    dv = np.diff(v, axis=0)
    di = np.diff(i, axis=0)
    v_transient = float(np.max(np.sqrt(np.mean(dv**2, axis=1))) / (v_mean + _EPS))
    i_transient = float(np.max(np.sqrt(np.mean(di**2, axis=1))) / (i_mean + _EPS))

    impedance = vrms / (irms + _EPS)
    impedance_spread = float(np.std(impedance) / (np.mean(impedance) + _EPS))
    apparent_impedance = float(np.mean(impedance))

    power_phase = np.mean(v * i, axis=0)
    power_imbalance = float(np.std(power_phase) / (np.mean(np.abs(power_phase)) + _EPS))

    return {
        "voltage_rms": v_mean,
        "current_rms": i_mean,
        "voltage_imbalance": v_imb,
        "current_imbalance": i_imb,
        "voltage_zero_sequence_ratio": v0_ratio,
        "voltage_negative_sequence_ratio": v2_ratio,
        "current_zero_sequence_ratio": i0_ratio,
        "current_negative_sequence_ratio": i2_ratio,
        "voltage_transient_ratio": v_transient,
        "current_transient_ratio": i_transient,
        "voltage_thd": _thd(v),
        "current_thd": _thd(i),
        "apparent_impedance": apparent_impedance,
        "impedance_phase_spread": impedance_spread,
        "power_phase_imbalance": power_imbalance,
    }


def fit_transformer_rule_reference(feature_rows: Iterable[dict[str, float]]) -> TransformerRuleReference:
    rows = list(feature_rows)
    if not rows:
        raise ValueError("normal reference requires at least one feature row")
    keys = list(rows[0])
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)) * 1.4826)
        fallback = float(np.std(values))
        center[key] = med
        scale[key] = max(mad, fallback * 0.25, 1e-6)
    return TransformerRuleReference(center=center, scale=scale)


def transformer_rule_diagnosis(signal_matrix, reference: TransformerRuleReference, *, threshold: float = 3.0) -> dict:
    f = transformer_electrical_features(signal_matrix)
    z = {key: (float(value) - reference.center[key]) / reference.scale[key] for key, value in f.items()}
    absz = {key: abs(value) for key, value in z.items()}

    disturbance = max(
        absz["current_rms"], absz["voltage_rms"], absz["current_transient_ratio"],
        absz["voltage_transient_ratio"], absz["apparent_impedance"], absz["current_thd"],
        absz["voltage_thd"],
    )
    asymmetry = max(
        absz["current_negative_sequence_ratio"], absz["current_zero_sequence_ratio"],
        absz["voltage_negative_sequence_ratio"], absz["voltage_zero_sequence_ratio"],
        absz["current_imbalance"], absz["voltage_imbalance"], absz["power_phase_imbalance"],
    )
    internal_signature = max(
        absz["apparent_impedance"], absz["impedance_phase_spread"], absz["current_thd"],
        absz["voltage_thd"], absz["current_transient_ratio"],
    )

    # Protection-inspired logic: a transformer candidate needs a strong electrical
    # disturbance plus an internal/impedance/harmonic signature. Very strong
    # sequence/asymmetry evidence is treated as an external grid-fault signature.
    external_fault = bool(asymmetry >= max(4.0, 1.35 * internal_signature))
    transformer_candidate = bool(
        disturbance >= threshold
        and internal_signature >= threshold
        and not external_fault
    )
    score = float(internal_signature + 0.35 * disturbance - 0.45 * max(asymmetry - 2.0, 0.0))
    confidence = float(np.clip((score - threshold + 2.0) / 6.0, 0.05, 0.95))

    reasons = []
    ranked = sorted(absz.items(), key=lambda kv: kv[1], reverse=True)
    for key, value in ranked[:5]:
        if value >= 2.0:
            reasons.append(f"{key} deviates {value:.2f} robust-sigma from normal")
    if external_fault:
        reasons.append("sequence/phase asymmetry dominates the internal transformer signature")

    return {
        "features": f,
        "robust_z": z,
        "disturbance_score": float(disturbance),
        "asymmetry_score": float(asymmetry),
        "internal_signature_score": float(internal_signature),
        "rule_score": score,
        "external_fault_signature": external_fault,
        "transformer_fault": transformer_candidate,
        "confidence": confidence,
        "reasons": reasons,
        "method": "deterministic_transformer_protection_rules_v1",
    }
