from tokenspeed_kernel.registrations._vendor import export_vendor_symbols

__all__ = ["argmax", "argmax_pair", "is_available"]

globals().update(
    export_vendor_symbols(
        "nvidia", "tokenspeed_kernel_nvidia.sampling.cute_dsl", __all__
    )
)
