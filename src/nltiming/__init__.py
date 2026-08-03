"""Nonlinear timing transforms, engine support, and likelihood interfaces."""

from . import sampling
from .run_io import (
    RunIOError,
    RunManifest,
    RunResults,
    build_run_manifest,
    derived_param_name,
    derived_fw10_columns,
    derived_kepler_columns,
    decode_physical,
    load_run,
    save_discovery_checkpoint,
    save_dynamic_checkpoint,
)
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
from .nonlinear_timing_model import NonLinearTimingModel, TimingContext
from .inference import (
    InferencePreset,
    Marginalize,
    TimingInference,
    TimingParameterPlan,
    coerce_timing_inference,
)
from .coordinates import TimingCoordinatePolicy, TimingExpansionSpec
from .linearization import (
    ExpansionOutsidePriorInteriorError,
    TimingLinearization,
)
from .expansion import ExpansionRefinementResult, refine_timing_expansion
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
from .protocols import (
    BinaryChartCapability,
    EnterprisePulsarLike,
    EphemerisExtras,
    JaxTimingEngine,
    PulsarData,
    TimingParameterMappingProvider,
    TimingPulsar,
    TimingEngine,
)
from .fw10_absorbed import (
    FW10AbsorbedChart,
    fw10_decode,
    fw10_encode,
    fw10_jacobian,
)
from .physical_charts import (
    KeplerLaplaceChart,
    KeplerLaplacePolicy,
    MarginalBasisFrame,
    kepler_from_laplace,
    kepler_from_laplace_vec,
    laplace_from_kepler,
)
from .space import ParameterSpace
from .decentering import (
    MarginalProducts,
    NumpyMarginalTransport,
    decode_decentered_chain,
)

# Capability gate for MetaPulsar Case-D conversion metadata (§8.5a): declares
# that `for_pulsar` probes `pulsar.conversion_metadata()` and enforces the
# required_sampling contract. Defined after the imports so it does not push
# every module-level import past the module docstring (E402).
SUPPORTS_CONVERSION_METADATA = True

__all__ = [
    "SUPPORTS_CONVERSION_METADATA",
    "NonLinearTimingModel",
    "TimingContext",
    "InferencePreset",
    "Marginalize",
    "TimingInference",
    "TimingParameterPlan",
    "coerce_timing_inference",
    "TimingCoordinatePolicy",
    "TimingExpansionSpec",
    "TimingLinearization",
    "ExpansionOutsidePriorInteriorError",
    "refine_timing_expansion",
    "ExpansionRefinementResult",
    "GeometryThresholds",
    "JointGeometryReport",
    "TransportCenterAxis",
    "GeometryCertificationError",
    "GeometryDiagnosticWarning",
    "certify_joint_geometry",
    "certify_decentered_geometry",
    "MarginalProducts",
    "NumpyMarginalTransport",
    "decode_decentered_chain",
    "transport_center_report",
    "box_hyper_probe_points",
    "write_geometry_report",
    "read_geometry_report",
    "WhiteningConfig",
    "LocalPosteriorMetric",
    "StaticTransportRecord",
    "DynamicTransportRecord",
    "OneAffineLayerError",
    "assert_static_layer_identity",
    "dynamic_transport_record",
    "toa_errors_metric",
    "frozen_white_metric",
    "ParameterSpace",
    "sampling",
    "RunIOError",
    "RunManifest",
    "RunResults",
    "build_run_manifest",
    "derived_param_name",
    "derived_fw10_columns",
    "derived_kepler_columns",
    "decode_physical",
    "load_run",
    "save_discovery_checkpoint",
    "save_dynamic_checkpoint",
    "EnterprisePulsarLike",
    "EphemerisExtras",
    "TimingEngine",
    "JaxTimingEngine",
    "TimingParameterMappingProvider",
    "TimingPulsar",
    "PulsarData",
    "TimingCapabilities",
    "TimingEvaluation",
    "TimingEvaluator",
    "TimingFitResult",
    "TimingParameter",
    "TimingParameters",
    "TimingScan",
    "TimingZFitResult",
    "BinaryChartCapability",
    "KeplerLaplacePolicy",
    "KeplerLaplaceChart",
    "FW10AbsorbedChart",
    "fw10_encode",
    "fw10_decode",
    "fw10_jacobian",
    "MarginalBasisFrame",
    "laplace_from_kepler",
    "kepler_from_laplace",
    "kepler_from_laplace_vec",
]
