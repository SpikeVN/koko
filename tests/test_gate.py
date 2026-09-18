import unittest
from unittest.mock import patch

from engine.bus import Bus
from engine.config import Config
from engine.events import Event, Kind
from engine.gate import InterpretationGate
from engine.monitor import Monitor


class InterpretationGateTests(unittest.TestCase):
    def test_timeout_is_measured_from_first_buffered_final(self):
        cfg = Config()
        cfg.gate.release_words = 8
        cfg.gate.no_new_words_s = 5.0
        gate = InterpretationGate(cfg, Bus(), Monitor(cfg))

        with patch("engine.gate.time.monotonic", side_effect=(100.0, 104.9)):
            gate.on_asr_text(Event(kind=Kind.FINAL_TEXT, text="one"))
            gate.on_asr_text(Event(kind=Kind.FINAL_TEXT, text="two"))

        self.assertIsNone(gate.maybe_fire(104.99))
        event = gate.maybe_fire(105.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.text, "one two")
