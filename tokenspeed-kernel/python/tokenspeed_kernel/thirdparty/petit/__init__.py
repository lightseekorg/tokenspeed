# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Optional Petit kernel dependency boundary."""


def import_petit_kernel():
    """Import and return the optional :mod:`petit_kernel` module.

    Returns:
        The imported ``petit_kernel`` module.

    Raises:
        RuntimeError: If ``petit_kernel`` is not installed.
    """
    try:
        import petit_kernel
    except ImportError as exc:
        raise RuntimeError(
            "Petit MegaMoE was selected, but petit_kernel is not installed. "
            "Install a petit_kernel build with MegaMoeConfig support in this environment."
        ) from exc
    return petit_kernel


__all__ = ["import_petit_kernel"]
