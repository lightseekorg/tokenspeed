import unittest
from types import SimpleNamespace

try:
    from tokenspeed.runtime.models.kimi_k25 import KimiK25ForConditionalGeneration
except (ImportError, ModuleNotFoundError) as exc:
    KimiK25ForConditionalGeneration = None
    _IMPORT_SKIP_REASON = f"KimiK25 import dependencies are not available: {exc}"
else:
    _IMPORT_SKIP_REASON = None


class TestKimiK25Delegation(unittest.TestCase):

    def setUp(self):
        if _IMPORT_SKIP_REASON is not None:
            self.skipTest(_IMPORT_SKIP_REASON)

    def test_logits_processor_delegates_to_language_model(self):
        model = object.__new__(KimiK25ForConditionalGeneration)
        processor = object()
        object.__setattr__(
            model,
            "language_model",
            SimpleNamespace(logits_processor=processor),
        )

        self.assertIs(model.logits_processor, processor)

    def test_logits_processor_missing_raises_attribute_error(self):
        model = object.__new__(KimiK25ForConditionalGeneration)
        object.__setattr__(model, "language_model", SimpleNamespace())

        with self.assertRaisesRegex(AttributeError, "logits_processor"):
            _ = model.logits_processor


if __name__ == "__main__":
    unittest.main()
