"""Explicit package/platform expectations for the existing performance CLIs."""

CPU_FIELDS = (
    "amrex_source_commit", "amrex_prefix", "compiler", "mpi",
    "ucx_version", "ucx_tls", "partition", "qos", "binding_policy",
)
GPU_FIELDS = (
    "amrex_source_commit", "amrex_prefix", "amrex_config_sha256",
    "compiler", "nvcc", "cuda_toolkit", "mpi", "mpi_cuda_build_version",
    "partition", "qos", "mig_profile",
)


def add_expected_arguments(parser, fields):
    for field in fields:
        parser.add_argument(
            "--expected-" + field.replace("_", "-"), required=True,
            help="independent expected " + field + " (not inferred from the input CSV)",
        )


def expected_values(args, fields):
    return {field: getattr(args, "expected_" + field) for field in fields}
