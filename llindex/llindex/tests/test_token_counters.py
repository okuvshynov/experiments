import unittest
from llindex.token_counters import token_counter_claude

class TestTokenCounterClaude(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(token_counter_claude(""), 0)

    def test_single_token(self):
        self.assertEqual(token_counter_claude("hello"), 1)

    def test_multiple_tokens(self):
        self.assertEqual(token_counter_claude("hello world"), 2)

    def test_long_text(self):
        long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        self.assertGreater(token_counter_claude(long_text), 10)

    def test_non_ascii_text(self):
        non_ascii_text = "¡Hola, mundo! 👋"
        self.assertGreater(token_counter_claude(non_ascii_text), 0)

if __name__ == "__main__":
    unittest.main()

