import unittest

from koko.engine.bus import Bus
from koko.engine.config import Config
from koko.engine.llm import LlmStage


class TargetLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_switching_target_clears_translation_context(self):
        stage = LlmStage(Config(), Bus(), None)
        stage._history.append({"role": "assistant", "content": "xin chao"})

        stage.set_target_language("English")

        self.assertEqual(stage.target_language, "English")
        self.assertEqual(stage._history, [])
        await stage.close()

    async def test_reselecting_target_preserves_translation_context(self):
        stage = LlmStage(Config(), Bus(), None)
        stage._history.append({"role": "assistant", "content": "xin chao"})

        stage.set_target_language("tiếng Việt")

        self.assertEqual(len(stage._history), 1)
        await stage.close()
