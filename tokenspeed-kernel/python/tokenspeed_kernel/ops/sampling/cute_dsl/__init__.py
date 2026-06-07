from tokenspeed_kernel._vendor import export_vendor_symbols

__all__ = ["argmax", "argmax_pair", "is_available"]

globals().update(
    export_vendor_symbols("nvidia", "tokenspeed_kernel.ops.sampling.cute_dsl", __all__)
)
