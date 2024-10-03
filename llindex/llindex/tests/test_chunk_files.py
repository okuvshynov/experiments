import unittest

from llindex.crawler import Crawler
from typing import List, Dict, Any


class TestChunkFiles(unittest.TestCase):
    def test_empty_list(self):
        c = Crawler('', {}) 
        files = []
        token_limit = 100
        expected_output = ([], [])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

    def test_single_file_within_limit(self):
        c = Crawler('', {}) 
        files = [{"approx_tokens": 50}]
        token_limit = 100
        expected_output = ([[{"approx_tokens": 50}]], [])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

    def test_single_file_exceeds_limit(self):
        c = Crawler('', {}) 
        files = [{"approx_tokens": 150}]
        token_limit = 100
        expected_output = ([[{"approx_tokens": 150}]], [])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

    def test_multiple_files_within_limit(self):
        c = Crawler('', {}) 
        files = [{"approx_tokens": 30}, {"approx_tokens": 70}]
        token_limit = 100
        expected_output = ([[{"approx_tokens": 30}, {"approx_tokens": 70}]], [])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

    def test_multiple_files_exceeds_limit(self):
        c = Crawler('', {}) 
        files = [{"approx_tokens": 70}, {"approx_tokens": 70}]
        token_limit = 100
        expected_output = ([[{"approx_tokens": 70}], [{"approx_tokens": 70}]], [])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

    def test_files_with_processing_result(self):
        c = Crawler('', {}) 
        files = [{"approx_tokens": 50, "processing_result": "done"}, {"approx_tokens": 70}]
        token_limit = 100
        expected_output = ([[{"approx_tokens": 70}]], [{"approx_tokens": 50, "processing_result": "done"}])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

    def test_large_input(self):
        c = Crawler('', {}) 
        files = [{"approx_tokens": 50}] * 100
        token_limit = 5000
        expected_output = ([], [])
        for i in range(0, len(files), 100):
            expected_output[0].append(files[i:i+100])
        self.assertEqual(c.chunk_files(files, token_limit), expected_output)

if __name__ == '__main__':
    unittest.main()

