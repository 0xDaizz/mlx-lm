"""Tests for block-level KV cache bridge functions (P7.5).

Tests decompose_cache_to_blocks() and reconstruct_cache_from_blocks()
using mock cache objects (no real model needed).
"""

import threading
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mlx_lm_server.kv_cache_manager import (
    compute_block_hash,
    decompose_cache_to_blocks,
    extract_block,
    inject_blocks,
    reconstruct_cache_from_blocks,
)


class MockKVCacheLayer:
    """Mock of a single KVCache layer for testing.

    Simulates the mlx_lm KVCache interface:
    - state property returns (keys, values)
    - keys/values have shape [1, n_heads, seq_len, head_dim]
    """

    def __init__(self, seq_len: int, n_heads: int = 4, head_dim: int = 8):
        self.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.offset = seq_len

    @property
    def state(self):
        return (self.keys, self.values)


def _make_mock_cache(num_layers: int = 2, seq_len: int = 32, n_heads: int = 4, head_dim: int = 8):
    """Create a mock List[KVCache] with random data."""
    return [MockKVCacheLayer(seq_len, n_heads, head_dim) for _ in range(num_layers)]


class TestDecomposeCacheToBlocks:
    """Tests for decompose_cache_to_blocks()."""

    def test_basic_decomposition(self):
        """Decompose a 32-token cache into 2 blocks of 16."""
        block_size = 16
        num_layers = 2
        seq_len = 32
        cache = _make_mock_cache(num_layers=num_layers, seq_len=seq_len)
        token_ids = list(range(seq_len))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        assert len(blocks) == 2
        for block in blocks:
            assert 'block_hash' in block
            assert 'token_ids' in block
            assert 'kv_data_per_layer' in block
            assert len(block['token_ids']) == block_size
            assert len(block['kv_data_per_layer']) == num_layers

    def test_token_ids_correct(self):
        """Block token_ids match the corresponding slice of input tokens."""
        block_size = 8
        token_ids = list(range(24))
        cache = _make_mock_cache(seq_len=24)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        assert len(blocks) == 3
        assert blocks[0]['token_ids'] == list(range(0, 8))
        assert blocks[1]['token_ids'] == list(range(8, 16))
        assert blocks[2]['token_ids'] == list(range(16, 24))

    def test_remainder_tokens_ignored(self):
        """Tokens that don't fill a complete block are ignored."""
        block_size = 16
        token_ids = list(range(30))  # 1 full block + 14 remainder
        cache = _make_mock_cache(seq_len=30)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        assert len(blocks) == 1
        assert blocks[0]['token_ids'] == list(range(16))

    def test_hash_consistency(self):
        """Block hashes match manually computed hashes."""
        block_size = 8
        token_ids = list(range(16))
        cache = _make_mock_cache(seq_len=16)

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        expected_hash_0 = compute_block_hash([], list(range(0, 8)))
        expected_hash_1 = compute_block_hash(list(range(0, 8)), list(range(8, 16)))

        assert blocks[0]['block_hash'] == expected_hash_0
        assert blocks[1]['block_hash'] == expected_hash_1

    def test_kv_data_shape(self):
        """KV data slices have correct shape."""
        block_size = 8
        n_heads = 4
        head_dim = 8
        seq_len = 16
        cache = _make_mock_cache(num_layers=2, seq_len=seq_len, n_heads=n_heads, head_dim=head_dim)
        token_ids = list(range(seq_len))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        for block in blocks:
            for layer_data in block['kv_data_per_layer']:
                assert layer_data['keys'].shape == (1, n_heads, block_size, head_dim)
                assert layer_data['values'].shape == (1, n_heads, block_size, head_dim)

    def test_kv_data_values_correct(self):
        """Decomposed KV data matches the original cache data at correct positions."""
        block_size = 8
        seq_len = 16
        cache = _make_mock_cache(num_layers=1, seq_len=seq_len)
        token_ids = list(range(seq_len))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        # Block 0 should match positions 0:8
        original_keys = cache[0].keys
        block0_keys = blocks[0]['kv_data_per_layer'][0]['keys']
        assert mx.allclose(block0_keys, original_keys[:, :, 0:8, :]).item()

        # Block 1 should match positions 8:16
        block1_keys = blocks[1]['kv_data_per_layer'][0]['keys']
        assert mx.allclose(block1_keys, original_keys[:, :, 8:16, :]).item()

    def test_empty_tokens(self):
        """Empty token list produces no blocks."""
        cache = _make_mock_cache(seq_len=0)
        blocks = decompose_cache_to_blocks(cache, [], 16)
        assert blocks == []


class TestReconstructCacheFromBlocks:
    """Tests for reconstruct_cache_from_blocks()."""

    def test_empty_blocks(self):
        """Empty block list returns empty list."""
        result = reconstruct_cache_from_blocks([], None)
        assert result == []


class TestDecomposeReconstructRoundtrip:
    """End-to-end tests: decompose then reconstruct."""

    def test_roundtrip_preserves_data(self):
        """Decompose -> inject_blocks roundtrip preserves KV data per layer."""
        block_size = 8
        seq_len = 16
        n_heads = 4
        head_dim = 8
        cache = _make_mock_cache(num_layers=2, seq_len=seq_len, n_heads=n_heads, head_dim=head_dim)
        token_ids = list(range(seq_len))

        blocks = decompose_cache_to_blocks(cache, token_ids, block_size)

        # Manually reconstruct by injecting blocks per layer
        for layer_idx in range(2):
            layer_blocks = [b['kv_data_per_layer'][layer_idx] for b in blocks]
            reconstructed = inject_blocks(layer_blocks)

            original_keys = cache[layer_idx].keys
            original_values = cache[layer_idx].values

            assert mx.allclose(reconstructed['keys'], original_keys).item()
            assert mx.allclose(reconstructed['values'], original_values).item()


class TestBlockLevelPrefixSharing:
    """Tests for block-level prefix sharing between requests."""

    def test_shared_system_prompt_blocks(self):
        """Two requests with same system prompt produce identical block hashes."""
        block_size = 8
        system_prompt = list(range(16))  # 2 blocks of system prompt

        req1_tokens = system_prompt + [100, 101, 102, 103, 104, 105, 106, 107]
        req2_tokens = system_prompt + [200, 201, 202, 203, 204, 205, 206, 207]

        cache1 = _make_mock_cache(seq_len=len(req1_tokens))
        cache2 = _make_mock_cache(seq_len=len(req2_tokens))

        blocks1 = decompose_cache_to_blocks(cache1, req1_tokens, block_size)
        blocks2 = decompose_cache_to_blocks(cache2, req2_tokens, block_size)

        # System prompt blocks (first 2) should have same hash
        assert blocks1[0]['block_hash'] == blocks2[0]['block_hash']
        assert blocks1[1]['block_hash'] == blocks2[1]['block_hash']

        # User message blocks (block 2) should have different hashes
        assert blocks1[2]['block_hash'] != blocks2[2]['block_hash']

    def test_partial_prefix_sharing(self):
        """Requests sharing only part of prefix share only those blocks."""
        block_size = 8
        shared = list(range(8))  # 1 shared block

        req1_tokens = shared + list(range(100, 108))  # Different second block
        req2_tokens = shared + list(range(200, 208))

        cache1 = _make_mock_cache(seq_len=len(req1_tokens))
        cache2 = _make_mock_cache(seq_len=len(req2_tokens))

        blocks1 = decompose_cache_to_blocks(cache1, req1_tokens, block_size)
        blocks2 = decompose_cache_to_blocks(cache2, req2_tokens, block_size)

        # First block shared
        assert blocks1[0]['block_hash'] == blocks2[0]['block_hash']
        # Second block different
        assert blocks1[1]['block_hash'] != blocks2[1]['block_hash']


class TestConcurrentDecomposeReconstruct:
    """Thread safety tests."""

    def test_concurrent_decompose(self):
        """Multiple threads decomposing simultaneously doesn't crash."""
        block_size = 8
        errors = []

        def worker(tid):
            try:
                for _ in range(10):
                    cache = _make_mock_cache(seq_len=32)
                    tokens = list(range(32))
                    blocks = decompose_cache_to_blocks(cache, tokens, block_size)
                    assert len(blocks) == 4
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
