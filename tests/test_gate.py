import unittest
from unittest.mock import patch

from koko.engine.bus import Bus
from koko.engine.config import Config
from koko.engine.events import Event, Kind
from koko.engine.gate import InterpretationGate
from koko.engine.monitor import Monitor


class InterpretationGateTests(unittest.TestCase):
    def test_timeout_is_measured_from_first_buffered_final(self):
        cfg = Config()
        cfg.gate.release_words = 8
        cfg.gate.no_new_words_s = 5.0
        gate = InterpretationGate(cfg, Bus(), Monitor(cfg))

        with patch("koko.engine.gate.time.monotonic", side_effect=(100.0, 104.9)):
            gate.on_asr_text(Event(kind=Kind.FINAL_TEXT, text="one"))
            gate.on_asr_text(Event(kind=Kind.FINAL_TEXT, text="two"))

        self.assertIsNone(gate.maybe_fire(104.99))
        event = gate.maybe_fire(105.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.text, "one two")
