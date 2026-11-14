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

"""Shared input cases for real child-process memory-saver launcher tests."""


def preload_cases(*, prepend_hook: bool):
    hook = "/test/torch_memory_saver_hook_mode_preload.so"
    explicit = "/custom/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
    extra, other = "/test/context_only.so", "/test/other_hook.so"
    cases = [
        ("no_override", None, None),
        ("ordinary_preload", [other], None),
        (
            "explicit_hook",
            [other, explicit, "/test/last.so"],
            [other, explicit, "/test/last.so", extra],
        ),
        (
            "legacy_cpp_hook",
            [other, "/custom/torch_memory_saver_cpp.some-postfix.so"],
            [other, "/custom/torch_memory_saver_cpp.some-postfix.so", extra],
        ),
        ("directory_name", ["/test/torch_memory_saver/libhelper.so"], None),
        ("other_library", ["/test/torch_memory_saver_helper.so"], None),
        ("cpp_helper", ["/test/torch_memory_saver_cpp_helper.so"], None),
        ("similar_name", ["/test/not_torch_memory_saver_hook_mode_preload.so"], None),
        (
            "repeated_explicit_hook",
            [explicit, other, explicit],
            [explicit, other, extra],
        ),
        ("repeated_context_hook", None, None),
    ]
    for name, inherited, override in cases:
        configured = [hook, extra, *(inherited if prepend_hook and inherited else [])]
        expected = override if override is not None else configured.copy()
        if name == "repeated_context_hook":
            configured.append(hook)
        for separator in (":", " \t"):
            yield (
                f"{name}-{'prepend' if prepend_hook else 'overwrite'}-{separator!r}",
                separator.join(inherited) if inherited is not None else None,
                separator.join(configured),
                ":".join(expected),
            )


def conflicting_preload_cases():
    first = "/one/torch_memory_saver_hook_mode_preload.so"
    second = "/two/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
    for separator in (":", " \t"):
        yield f"parent-{separator!r}", separator.join([first, second]), first
        yield f"context-{separator!r}", None, separator.join([first, second])
