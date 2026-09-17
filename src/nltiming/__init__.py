"""Nonlinear timing transforms, engine support, and likelihood interfaces."""

# Importing log_config installs the default loguru sink (WARNING and above;
# loguru's own default is DEBUG). Keep it the first package import.
from . import hybrid, sampling
from .coordinates import TimingCoordinatePolicy, TimingExpansionSpec
from .decentering import (
    MarginalProducts,
    NumpyMarginalTransport,
    decode_decentered_chain,
)
from .evaluator import (
    TimingCapabilities,
    TimingEvaluation,
    TimingEvaluator,
    TimingFitResult,
    TimingParameter,
    TimingParameters,
    TimingScan,
    TimingZFitResult,
)
from .expansion import ExpansionRefinementResult, refine_timing_expansion
from .fw10_absorbed import (
    FW10AbsorbedChart,
    fw10_decode,
    fw10_encode,
    fw10_jacobian,
)
from .geometry import (
    GeometryCertificationError,
    GeometryDiagnosticWarning,
    GeometryThresholds,
    JointGeometryReport,
    TransportCenterAxis,
    box_hyper_probe_points,
    certify_decentered_geometry,
    certify_joint_geometry,
    read_geometry_report,
    transport_center_report,
    write_geometry_report,
)
from .inference import (
    InferencePreset,
    Marginalize,
    TimingInference,
    TimingParameterPlan,
    coerce_timing_inference,
)
from .linearization import (
    ExpansionOutsidePriorInteriorError,
    TimingLinearization,
)
from .log_config import configure_logging
from .metric import (
    DynamicTransportRecord,
    LocalPosteriorMetric,
    OneAffineLayerError,
    StaticTransportRecord,
    WhiteningConfig,
    assert_static_layer_identity,
    dynamic_transport_record,
    frozen_white_metric,
    toa_errors_metric,
)
from .nonlinear_timing_model import TimingSignal, TimingSpec
from .physical_charts import (
    KeplerLaplaceChart,
    KeplerLaplacePolicy,
    MarginalBasisFrame,
    kepler_from_laplace,
    kepler_from_laplace_vec,
    laplace_from_kepler,
)
from .protocols import (
    BinaryChartCapability,
    EnterprisePulsarLike,
    EphemerisExtras,
    JaxTimingEngine,
    PulsarData,
    TimingEngine,
    TimingParameterMappingProvider,
    TimingPulsar,
)
from .run_io import (
    RunIOError,
    RunManifest,
    RunResults,
    build_run_manifest,
    decode_physical,
    derived_fw10_columns,
    derived_kepler_columns,
    derived_param_name,
    load_run,
    save_discovery_checkpoint,
    save_dynamic_checkpoint,
)
from .space import ParameterSpace

# Capability gate for MetaPulsar Case-D conversion metadata (§8.5a): declares
# that `for_pulsar` probes `pulsar.conversion_metadata()` and enforces the
# required_sampling contract. Defined after the imports so it does not push
# every module-level import past the module docstring (E402).
SUPPORTS_CONVERSION_METADATA = True

__all__ = [
    "SUPPORTS_CONVERSION_METADATA",
    "BinaryChartCapability",
    "DynamicTransportRecord",
    "EnterprisePulsarLike",
    "EphemerisExtras",
    "ExpansionOutsidePriorInteriorError",
    "ExpansionRefinementResult",
    "FW10AbsorbedChart",
    "GeometryCertificationError",
    "GeometryDiagnosticWarning",
    "GeometryThresholds",
    "InferencePreset",
    "JaxTimingEngine",
    "JointGeometryReport",
    "KeplerLaplaceChart",
    "KeplerLaplacePolicy",
    "LocalPosteriorMetric",
    "MarginalBasisFrame",
    "MarginalProducts",
    "Marginalize",
    "NumpyMarginalTransport",
    "OneAffineLayerError",
    "ParameterSpace",
    "PulsarData",
    "RunIOError",
    "RunManifest",
    "RunResults",
    "StaticTransportRecord",
    "TimingCapabilities",
    "TimingCoordinatePolicy",
    "TimingEngine",
    "TimingEvaluation",
    "TimingEvaluator",
    "TimingExpansionSpec",
    "TimingFitResult",
    "TimingInference",
    "TimingLinearization",
    "TimingParameter",
    "TimingParameterMappingProvider",
    "TimingParameterPlan",
    "TimingParameters",
    "TimingPulsar",
    "TimingScan",
    "TimingSignal",
    "TimingSpec",
    "TimingZFitResult",
    "TransportCenterAxis",
    "WhiteningConfig",
    "assert_static_layer_identity",
    "box_hyper_probe_points",
    "build_run_manifest",
    "certify_decentered_geometry",
    "certify_joint_geometry",
    "coerce_timing_inference",
    "configure_logging",
    "decode_decentered_chain",
    "decode_physical",
    "derived_fw10_columns",
    "derived_kepler_columns",
    "derived_param_name",
    "dynamic_transport_record",
    "frozen_white_metric",
    "fw10_decode",
    "fw10_encode",
    "fw10_jacobian",
    "hybrid",
    "kepler_from_laplace",
    "kepler_from_laplace_vec",
    "laplace_from_kepler",
    "load_run",
    "read_geometry_report",
    "refine_timing_expansion",
    "sampling",
    "save_discovery_checkpoint",
    "save_dynamic_checkpoint",
    "toa_errors_metric",
    "transport_center_report",
    "write_geometry_report",
]
