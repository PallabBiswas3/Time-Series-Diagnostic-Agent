from __future__ import annotations

from ..contracts import ToolRegistry
from .change import change_point_detection, cross_sensor_relationships, rolling_statistics
from .multivariate import (
    contribution_analysis,
    data_quality_check,
    detect_operating_regimes,
    granger_causality,
    pca_monitoring,
    regime_normalization,
    residual_analysis,
    robust_anomaly_detection,
    standardize_against_normal,
    stationarity_analysis,
)
from .process import (
    causal_graph_filter,
    fault_onset_timing,
    process_diagnosis,
    root_cause_rank,
)
from .signal import (
    bandpass_filter,
    envelope_spectrum,
    hilbert_envelope,
    signal_integrity,
    spectral_kurtosis,
    time_domain_features,
    time_frequency_analysis,
    welch_psd,
)
from .wavelet import (
    correlation_sensor_weighting,
    cross_correlation_analysis,
    multisensor_fusion,
    wavelet_denoising,
)


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    tools = {
        "signal_integrity": signal_integrity,
        "process_data_quality": data_quality_check,
        "scada_quality_check": data_quality_check,
        "time_domain_features": time_domain_features,
        "welch_psd": welch_psd,
        "time_frequency_analysis": time_frequency_analysis,
        "spectral_kurtosis": spectral_kurtosis,
        "bandpass_filter": bandpass_filter,
        "hilbert_envelope": hilbert_envelope,
        "envelope_spectrum": envelope_spectrum,
        "standardize_against_normal": standardize_against_normal,
        "pca_monitoring": pca_monitoring,
        "contribution_analysis": contribution_analysis,
        "stationarity_analysis": stationarity_analysis,
        "granger_causality": granger_causality,
        "causal_graph_filter": causal_graph_filter,
        "fault_onset_timing": fault_onset_timing,
        "root_cause_rank": root_cause_rank,
        "process_diagnosis": process_diagnosis,
        "operating_regime_detection": detect_operating_regimes,
        "operating_condition_identification": detect_operating_regimes,
        "regime_normalization": regime_normalization,
        "residual_analysis": residual_analysis,
        "scada_anomaly_detection": robust_anomaly_detection,
        "rolling_statistics": rolling_statistics,
        "cross_sensor_relationships": cross_sensor_relationships,
        "change_point_detection": change_point_detection,
        "wavelet_denoising": wavelet_denoising,
        "cross_correlation_analysis": cross_correlation_analysis,
        "correlation_sensor_weighting": correlation_sensor_weighting,
        "multisensor_fusion": multisensor_fusion,
    }
    for name, fn in tools.items():
        registry.register(name, fn)
    return registry
