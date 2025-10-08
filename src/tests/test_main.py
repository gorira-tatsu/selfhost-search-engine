import math
import unittest
from pprint import pprint

import main


DEBUG_OUTPUT = True


def debug(label, value):
    """Emit human-friendly output for intermediate objects when debugging is enabled."""
    if DEBUG_OUTPUT:
        print(f"=== {label} ===")
        pprint(value)


class TestMainModule(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_create_inverted_index_tracks_positions(self):
        tokens = ["alpha", "beta", "alpha", "gamma"]

        inverted = main.create_inverted_index(tokens)

        debug("tokens", tokens)
        debug("inverted.index", inverted.index)

        self.assertEqual(inverted.index["alpha"], [0, 2])
        self.assertEqual(inverted.index["beta"], [1])
        self.assertEqual(inverted.index["gamma"], [3])

    def test_token_to_ngram_three_character_windows(self):
        tokens = ["alpha", "beta"]

        ngrams = main.token_to_ngram(tokens, n=3)

        debug("tokens", tokens)
        debug("ngrams", ngrams)

        self.assertEqual(ngrams, [["alp", "lph", "pha"], ["bet", "eta"]])

    def test_create_document_inverted_index_builds_metadata(self):
        docs = {
            0: ["hello", "world"],
            1: ["hello", "test"],
        }

        doc_index = main.create_document_inverted_index(docs)

        debug("docs", docs)
        debug("doc_index.document_count", doc_index.document_count)
        debug("doc_index.index", doc_index.index)

        self.assertEqual(doc_index.document_count, 2)
        self.assertEqual(doc_index.average_length, 2)
        self.assertEqual(doc_index.index["hello"], [(0, 0), (1, 0)])
        self.assertEqual(doc_index.index["world"], [(0, 1)])

    def test_add_document_inverted_index_appends_sorted_postings(self):
        base_index = main.create_document_inverted_index({0: ["foo", "bar"]})

        updated = main.addDocumentInvertedIndex(base_index, ["foo baz"], docId=1)

        debug("base_index.index", base_index.index)
        debug("updated.document_count", updated.document_count)
        debug("updated.index", updated.index)

        self.assertEqual(updated.document_count, 2)
        self.assertTrue(math.isclose(updated.average_length, 2.0))
        self.assertEqual(updated.index["foo"], [(0, 0), (1, 0)])
        self.assertEqual(updated.index["bar"], [(0, 1)])
        self.assertEqual(updated.index["baz"], [(1, 1)])

    def test_rank_bm25_document_scores_more_frequent_docs_higher(self):
        docs = {
            0: ["term", "term", "term"],
            1: ["term", "alpha", "beta"],
        }
        doc_index = main.create_document_inverted_index(docs)

        scores = main.rankBM25_DocumentAtATime(doc_index, docs, "term", k_1=1.5, b=0.75)

        debug("docs", docs)
        debug("doc_index.index", doc_index.index)
        debug("scores", scores)

        self.assertGreater(scores[0], scores[1])

    def test_search_word_returns_top_document_and_frequency(self):
        docs = {
            0: ["alpha", "beta", "alpha"],
            1: ["alpha", "beta"],
        }
        doc_index = main.create_document_inverted_index(docs)

        result = main.searchWord("alpha", doc_index)

        debug("docs", docs)
        debug("doc_index.index", doc_index.index)
        debug("search result", result)

        self.assertEqual(result, {"id": 0, "count": 2})


if __name__ == "__main__":
    unittest.main()
