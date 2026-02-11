# Speculative Decoding: Implementation Blueprint (Phase 1–4)

> **목적**: 이 문서는 Claude Code가 코드 작성 플랜을 수립하고 구현을 시작하기 위한 완전한 참조 문서입니다.
> mlx-lm 위의 커스텀 API 레이어에 speculative decoding을 통합합니다.
> Continuous batching은 이미 구현 완료된 상태입니다.

---

## 0. 전제 조건 및 현재 상태

| Component | Status | 비고 |
|-----------|--------|------|
| OpenAI-compat API Layer | ✅ Done | FastAPI 기반 |
| Continuous Batching Engine | ✅ Done | vLLM-MLX 로직 이식, token-level batched forward |
| Speculative Decoding | 🔧 구현 대상 | 이 문서의 범위 |

**기술 스택**: Python 3.12+, mlx, mlx-lm, Apple Silicon (M-series), Metal

**핵심 설계 원칙**:
- `--spec-decode={none|ngram|draft|eagle}` flag로 모드 선택
- Proposer는 Strategy Pattern으로 교체 가능
- 기존 continuous batching 엔진의 `step()` 루프에 자연스럽게 삽입
- Dynamic control로 배치 크기에 따라 자동 ON/OFF

---

## 1. 파일 구조

```
hwquant/
├── serve.py                          # CLI entrypoint (기존)
├── engine/
│   ├── engine.py                     # 메인 엔진 루프 (기존, 수정)
│   ├── scheduler.py                  # Unified scheduler (기존, 수정)
│   └── request.py                    # Request/Sequence 데이터 (기존)
├── spec_decode/                      # ★ 신규 모듈
│   ├── __init__.py
│   ├── config.py                     # SpecDecodeConfig
│   ├── proposer/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseProposer (ABC)
│   │   ├── ngram.py                  # Phase 1: NGramProposer
│   │   ├── draft_model.py            # Phase 2: DraftModelProposer
│   │   └── eagle.py                  # Phase 3: EAGLEProposer
│   ├── verifier.py                   # BatchedVerifier
│   ├── rejection_sampler.py          # BatchedRejectionSampler
│   ├── dynamic_controller.py         # Phase 4: DynamicSpecController
│   └── kv_manager.py                 # SpecDecodeKVManager
└── tests/
    └── spec_decode/
        ├── test_ngram_proposer.py
        ├── test_rejection_sampler.py
        ├── test_draft_model.py
        ├── test_eagle.py
        ├── test_dynamic_controller.py
        └── test_engine_integration.py
```

---

## 2. Config (spec_decode/config.py)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SpecDecodeConfig:
    """
    Speculative decoding 전체 설정.
    CLI arg, config file, API request-level override 모두 이 구조체로 통합.
    """

    # ─── 모드 선택 ───
    spec_decode_mode: Literal["none", "ngram", "draft", "eagle"] = "none"

    # ─── 공통 ───
    num_speculative_tokens: int = 5           # draft 토큰 수 k
    disable_by_batch_size: int = 8            # batch >= 이 값이면 자동 OFF
    acceptance_rate_threshold: float = 0.3    # EMA가 이 미만이면 spec 중단

    # ─── Phase 1: N-gram ───
    ngram_max: int = 4
    ngram_min: int = 1
    ngram_prompt_lookup: bool = True          # prompt 토큰에서도 매칭 탐색

    # ─── Phase 2: Draft Model ───
    draft_model_path: Optional[str] = None
    draft_model_quantize: Optional[str] = None  # e.g. "4bit", "8bit"

    # ─── Phase 3: EAGLE ───
    eagle_head_path: Optional[str] = None     # 추가 prediction head 경로
    eagle_num_layers: int = 1                 # prediction head 레이어 수

    # ─── Phase 4: Dynamic Control ───
    dynamic_spec_decode: bool = True
    acceptance_rate_ema_alpha: float = 0.1    # EMA smoothing factor
    adaptive_k: bool = True                   # acceptance rate에 따라 k 동적 조절

    def validate(self):
        if self.spec_decode_mode == "draft" and not self.draft_model_path:
            raise ValueError("--draft-model required when --spec-decode=draft")
        if self.spec_decode_mode == "eagle" and not self.eagle_head_path:
            raise ValueError("--eagle-head-path required when --spec-decode=eagle")
```

**CLI 매핑** (serve.py에 추가):

```python
# argparse 그룹
spec_group = parser.add_argument_group("Speculative Decoding")
spec_group.add_argument("--spec-decode", choices=["none","ngram","draft","eagle"], default="none")
spec_group.add_argument("--num-speculative-tokens", type=int, default=5)
spec_group.add_argument("--disable-by-batch-size", type=int, default=8)
spec_group.add_argument("--ngram-max", type=int, default=4)
spec_group.add_argument("--ngram-min", type=int, default=1)
spec_group.add_argument("--ngram-prompt-lookup", action="store_true", default=True)
spec_group.add_argument("--draft-model", type=str, default=None)
spec_group.add_argument("--draft-model-quantize", type=str, default=None)
spec_group.add_argument("--eagle-head-path", type=str, default=None)
spec_group.add_argument("--dynamic-spec-decode", action="store_true", default=True)
spec_group.add_argument("--no-dynamic-spec-decode", dest="dynamic_spec_decode", action="store_false")
```

**Request-level override** (OpenAI extra_body):

```json
{
    "model": "qwen3-32b",
    "messages": [],
    "extra_body": {
        "spec_decode": "ngram",
        "num_speculative_tokens": 3
    }
}
```

---

## 3. Proposer Interface (spec_decode/proposer/base.py)

모든 proposer가 구현하는 통일된 인터페이스. Strategy Pattern으로 엔진에서 교체 가능.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class ProposalResult:
    """Proposer의 출력."""
    draft_tokens: mx.array          # [batch, k] — proposal된 토큰 ID
    draft_probs: Optional[mx.array] # [batch, k, vocab] — draft model일 때만, n-gram은 None
    proposal_lens: mx.array         # [batch] — 시퀀스별 실제 proposal 길이 (패딩 제외)


class BaseProposer(ABC):
    """모든 proposer의 기반 클래스."""

    @abstractmethod
    def propose(
        self,
        sequences: List["Sequence"],
        k: int,
    ) -> Optional[ProposalResult]:
        """
        배치 내 모든 시퀀스에 대해 draft token 생성.

        Args:
            sequences: decode 상태의 시퀀스 리스트
            k: 시퀀스당 생성할 draft 토큰 수

        Returns:
            ProposalResult 또는 None (전체 배치 proposal 실패 시)
        """
        ...

    @property
    @abstractmethod
    def needs_draft_probs(self) -> bool:
        """
        True: rejection sampling 사용 (draft model, EAGLE)
        False: greedy/threshold verification 사용 (n-gram)
        """
        ...

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """True면 Metal GPU 사용 (draft model, EAGLE), False면 CPU (n-gram)"""
        ...


def create_proposer(config: "SpecDecodeConfig", target_model=None) -> Optional[BaseProposer]:
    """Factory function. config.spec_decode_mode에 따라 proposer 인스턴스 생성."""
    match config.spec_decode_mode:
        case "none":
            return None
        case "ngram":
            from .ngram import NGramProposer
            return NGramProposer(
                ngram_max=config.ngram_max,
                ngram_min=config.ngram_min,
                prompt_lookup=config.ngram_prompt_lookup,
            )
        case "draft":
            from .draft_model import DraftModelProposer
            return DraftModelProposer(
                model_path=config.draft_model_path,
                quantize=config.draft_model_quantize,
            )
        case "eagle":
            from .eagle import EAGLEProposer
            return EAGLEProposer(
                target_model=target_model,
                head_path=config.eagle_head_path,
                num_layers=config.eagle_num_layers,
            )
```

---

## 4. Phase 1: N-gram Proposer (spec_decode/proposer/ngram.py)

### 4.1 설계 근거

- vLLM의 `vllm/spec_decode/ngram_worker.py` 패턴 이식
- 추가 모델 로딩 없음 → 메모리 오버헤드 0
- CPU에서 순수 Python 실행 → Metal GPU는 target model에 전념
- 코드 생성, 번역, 요약 등 반복 패턴이 많은 task에서 높은 acceptance rate

### 4.2 구현

```python
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from .base import BaseProposer, ProposalResult


class NGramProposer(BaseProposer):
    """
    Context(prompt + generated) 내 n-gram 매칭으로 draft token 생성.

    알고리즘:
    1. 현재 context 끝 n개 토큰을 key로 설정
    2. context 앞부분에서 같은 n-gram 탐색
    3. 매칭 시 그 뒤 k개 토큰을 draft로 제안
    4. 큰 n부터 탐색 (4-gram → 3-gram → ... → 1-gram)

    vLLM ngram_worker 대비 차이:
    - Batched: 여러 시퀀스에 대해 한 번에 proposal
    - suffix index 캐싱으로 긴 context에서 O(1) 탐색
    - 매칭 실패 시퀀스는 proposal_len=0으로 표시 → 해당 시퀀스만 일반 decode
    """

    def __init__(self, ngram_max: int = 4, ngram_min: int = 1,
                 prompt_lookup: bool = True):
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.prompt_lookup = prompt_lookup

    @property
    def needs_draft_probs(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False  # 순수 CPU 연산

    def propose(
        self,
        sequences: List["Sequence"],
        k: int,
    ) -> Optional[ProposalResult]:
        batch_proposals: List[List[int]] = []
        proposal_lens: List[int] = []
        any_found = False

        for seq in sequences:
            tokens = self._propose_single(seq, k)
            batch_proposals.append(tokens)
            proposal_lens.append(len(tokens))
            if tokens:
                any_found = True

        if not any_found:
            return None  # 전체 배치 실패 → 일반 decode fallback

        # 가장 긴 proposal에 맞춰 0-padding
        max_len = max(proposal_lens) if proposal_lens else 0
        if max_len == 0:
            return None

        padded = []
        for p in batch_proposals:
            padded.append(p + [0] * (max_len - len(p)))

        return ProposalResult(
            draft_tokens=mx.array(padded, dtype=mx.int32),
            draft_probs=None,
            proposal_lens=mx.array(proposal_lens, dtype=mx.int32),
        )

    def _propose_single(self, seq: "Sequence", k: int) -> List[int]:
        """
        단일 시퀀스에 대한 n-gram 매칭.

        큰 n-gram부터 탐색 → 첫 매칭에서 다음 k개 토큰 반환.
        suffix index가 있으면 사용, 없으면 선형 탐색.
        """
        if self.prompt_lookup:
            context = seq.prompt_tokens + seq.generated_tokens
        else:
            context = seq.generated_tokens

        if len(context) < self.ngram_min + 1:
            return []

        # suffix index 사용 (캐싱)
        if hasattr(seq, '_ngram_index') and not seq._ngram_dirty:
            return self._propose_with_index(seq._ngram_index, context, k)

        # 선형 탐색 fallback
        return self._propose_linear(context, k)

    def _propose_linear(self, context: List[int], k: int) -> List[int]:
        """선형 탐색. context가 짧을 때 (< ~1000 토큰) 충분히 빠름."""
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(context) < n + 1:
                continue

            key = tuple(context[-n:])
            search_end = len(context) - n  # 현재 위치 자신은 제외

            # 역순 탐색: 최근 매칭이 더 관련성 높음
            for i in range(search_end - 1, -1, -1):
                if tuple(context[i:i + n]) == key:
                    start = i + n
                    end = min(start + k, len(context))
                    proposals = list(context[start:end])
                    if len(proposals) >= 1:
                        return proposals[:k]

        return []

    def _propose_with_index(
        self, index: Dict[tuple, List[int]], context: List[int], k: int
    ) -> List[int]:
        """
        Suffix index 기반 O(1) 탐색.
        index는 {n-gram_tuple: [position_list]} 형태.
        """
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(context) < n + 1:
                continue

            key = tuple(context[-n:])
            if key not in index:
                continue

            positions = index[key]
            # 역순: 마지막(최근) 위치부터 탐색
            for pos in reversed(positions):
                if pos + n >= len(context):
                    continue  # 현재 위치 자신 제외
                start = pos + n
                end = min(start + k, len(context))
                proposals = list(context[start:end])
                if len(proposals) >= 1:
                    return proposals[:k]

        return []

    @staticmethod
    def build_suffix_index(tokens: List[int], ngram_max: int = 4) -> Dict[tuple, List[int]]:
        """
        시퀀스에 대한 n-gram suffix index 구축.
        호출 시점: prefill 완료 후 1회, 이후 생성 토큰 추가 시 incremental update.

        Returns:
            {(token_1, ..., token_n): [pos_0, pos_1, ...]}
        """
        index: Dict[tuple, List[int]] = {}
        for n in range(1, ngram_max + 1):
            for i in range(len(tokens) - n):
                key = tuple(tokens[i:i + n])
                if key not in index:
                    index[key] = []
                index[key].append(i)
        return index

    @staticmethod
    def update_suffix_index(
        index: Dict[tuple, List[int]],
        tokens: List[int],
        new_token_count: int,
        ngram_max: int = 4,
    ):
        """
        새 토큰 추가 시 incremental index 업데이트.
        전체 재구축 대신 새로 추가된 영역만 갱신.
        """
        start = max(0, len(tokens) - new_token_count - ngram_max)
        for n in range(1, ngram_max + 1):
            for i in range(start, len(tokens) - n):
                key = tuple(tokens[i:i + n])
                if key not in index:
                    index[key] = []
                if not index[key] or index[key][-1] != i:
                    index[key].append(i)
```

### 4.3 테스트 케이스 (tests/spec_decode/test_ngram_proposer.py)

```python
def test_exact_match():
    """반복 패턴이 있으면 정확히 proposal해야 함."""
    # context: "A B C D E A B C D E F G"
    # 현재 끝이 [D, E] → 앞에서 [D, E] 매칭 → [A, B, C] proposal (if k=3)
    seq = make_sequence(prompt=[1,2,3,4,5], generated=[1,2,3,4,5,6,7])
    proposer = NGramProposer(ngram_max=2)
    result = proposer._propose_single(seq, k=3)
    # context[-2:] = [6,7], 앞에서 매칭되지 않을 수 있음
    # context[-2:] = [4,5] 두 번 나타남 → 두 번째 뒤의 [6,7] 또는 첫 번째 뒤의 [1,2,3]

def test_no_match():
    """매칭 없으면 빈 리스트 반환."""
    seq = make_sequence(prompt=[1,2,3], generated=[4,5,6])
    proposer = NGramProposer(ngram_max=4)
    result = proposer._propose_single(seq, k=3)
    assert result == []

def test_batch_partial_match():
    """배치 내 일부만 매칭되면, 매칭된 것만 proposal 있어야 함."""
    proposer = NGramProposer(ngram_max=2)
    seqs = [make_sequence_with_repeat(), make_sequence_without_repeat()]
    result = proposer.propose(seqs, k=3)
    assert result is not None
    assert result.proposal_lens[0] > 0
    assert result.proposal_lens[1] == 0

def test_suffix_index_correctness():
    """Suffix index가 선형 탐색과 동일한 결과를 내야 함."""
    tokens = [1,2,3,4,5,1,2,3,4,5,6]
    index = NGramProposer.build_suffix_index(tokens, ngram_max=4)
    # 선형 탐색과 index 탐색 결과 비교
```

---

## 5. Batched Verifier (spec_decode/verifier.py)

Draft token들을 target model로 일괄 검증하는 모듈. 모든 Phase에서 공유.

```python
from typing import List

import mlx.core as mx


class BatchedVerifier:
    """
    Target model로 draft token 일괄 검증.

    CUDA와 달리 batch expansion(시퀀스 복제) 없이
    padding + attention mask로 가변 길이 처리.
    Apple Silicon unified memory → zero-copy.
    """

    def __init__(self, target_model, tokenizer):
        self.model = target_model
        self.tokenizer = tokenizer

    def verify(
        self,
        sequences: List["Sequence"],
        draft_tokens: mx.array,       # [batch, k]
        proposal_lens: mx.array,      # [batch] — 시퀀스별 실제 proposal 길이
    ) -> mx.array:
        """
        Target model forward pass로 draft token들의 확률 분포 계산.

        각 시퀀스에 대해 context 끝 토큰 + draft tokens를 입력으로 넣고,
        single forward pass로 target_probs를 얻음.

        Args:
            sequences: decode 중인 시퀀스 리스트
            draft_tokens: [batch, k] — proposer가 생성한 draft
            proposal_lens: [batch] — 시퀀스별 유효한 proposal 길이

        Returns:
            target_probs: [batch, max_k+1, vocab_size]
            max_k+1인 이유: k개 draft token verification + 1개 bonus/resample 위치
        """
        batch_size = len(sequences)
        k = draft_tokens.shape[1]

        # 각 시퀀스별 verification 입력 구성
        # 입력 = [last_token_of_context, draft_token_0, ..., draft_token_{k-1}]
        # → target model이 각 위치에서의 next-token 확률을 출력
        verify_inputs = []
        for i, seq in enumerate(sequences):
            plen = int(proposal_lens[i])
            if plen == 0:
                # proposal 없는 시퀀스: 일반 decode (1 토큰만)
                verify_inputs.append(mx.array([seq.last_token_id], dtype=mx.int32))
            else:
                # last_token + draft_tokens[:plen]
                tokens = mx.concatenate([
                    mx.array([seq.last_token_id], dtype=mx.int32),
                    draft_tokens[i, :plen],
                ])
                verify_inputs.append(tokens)

        # Padding + batching
        max_len = max(v.shape[0] for v in verify_inputs)
        padded_input = mx.zeros((batch_size, max_len), dtype=mx.int32)
        attention_mask = mx.zeros((batch_size, max_len), dtype=mx.bool_)

        for i, v in enumerate(verify_inputs):
            padded_input[i, :v.shape[0]] = v
            attention_mask[i, :v.shape[0]] = True

        # ★ Single target model forward pass (배치 전체)
        # 기존 continuous batching 엔진의 forward 메서드 활용
        # KV cache는 각 시퀀스의 기존 cache를 extend하는 방식
        target_logits = self._forward_with_kv_cache(
            sequences, padded_input, attention_mask
        )
        # target_logits: [batch, max_len, vocab_size]

        target_probs = mx.softmax(target_logits, axis=-1)
        mx.eval(target_probs)

        return target_probs

    def _forward_with_kv_cache(self, sequences, input_ids, mask):
        """
        기존 continuous batching 엔진의 batched forward를 호출.
        각 시퀀스의 KV cache를 extend하면서 forward pass 수행.

        ★ 구현 시 기존 엔진의 forward 인터페이스에 맞춰 조정 필요.
        """
        # TODO: 기존 엔진의 batched forward 메서드에 위임
        # 핵심: input_ids가 1개(일반 decode)가 아니라 k+1개이므로
        #        KV cache를 k+1 위치만큼 extend
        raise NotImplementedError("기존 continuous batching 엔진의 forward에 위임")
```

**구현 시 주의점**:
- 기존 continuous batching 엔진의 forward pass는 `input_ids`가 시퀀스당 1개 토큰을 가정할 수 있음
- Spec decode 시에는 시퀀스당 `k+1`개 토큰을 입력하므로, forward에 가변 길이 입력을 받을 수 있도록 확장 필요
- KV cache도 `k+1` 위치만큼 pre-allocate → reject 시 truncate

---

## 6. Rejection Sampler (spec_decode/rejection_sampler.py)

### 6.1 핵심 알고리즘 (vLLM V1 기반)

```
For each sequence in batch:
    For position i = 0, 1, ..., k-1 (left to right):
        if target_prob[draft_token[i]] / draft_prob[draft_token[i]] >= uniform_random():
            ACCEPT draft_token[i]
        else:
            REJECT → resample from normalized max(0, target_prob - draft_prob)
            BREAK (이후 위치는 모두 reject)

    if all k tokens accepted:
        sample BONUS token from target_prob[k]

Output: [batch, k+1] tensor, rejected positions = -1 (PLACEHOLDER)
```

### 6.2 구현

```python
import mlx.core as mx


PLACEHOLDER_TOKEN_ID = -1


class BatchedRejectionSampler:
    """
    Batched speculative decoding rejection sampler.

    Source: vLLM V1 `vllm/v1/sample/rejection_sampler.py`
    핵심 이식 패턴: -1 padding으로 가변 길이 acceptance 처리.
    """

    def __call__(
        self,
        target_probs: mx.array,    # [batch, k+1, vocab]
        draft_probs: mx.array,     # [batch, k, vocab]
        draft_tokens: mx.array,    # [batch, k]
        proposal_lens: mx.array,   # [batch] — 시퀀스별 유효 proposal 길이
    ) -> mx.array:
        """
        Returns:
            output_tokens: [batch, k+1]
            accepted positions have token IDs, rejected = -1
        """
        return self._forward_vectorized(
            target_probs, draft_probs, draft_tokens, proposal_lens
        )

    def _forward_vectorized(
        self,
        target_probs: mx.array,
        draft_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> mx.array:
        """
        벡터화 버전. 배치 전체를 루프 없이 한 번에 처리.
        """
        batch_size, k = draft_tokens.shape
        output = mx.full((batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32)

        # 1) Draft token에 대한 target/draft probability 추출
        # target_probs[:, :k, :] 에서 draft_tokens 위치의 확률
        idx = draft_tokens[:, :, None]  # [batch, k, 1]
        target_p = mx.take_along_axis(target_probs[:, :k, :], idx, axis=2).squeeze(-1)  # [batch, k]
        draft_p = mx.take_along_axis(draft_probs, idx, axis=2).squeeze(-1)              # [batch, k]

        # 2) Acceptance criterion: p_target / p_draft >= uniform
        rand = mx.random.uniform(shape=(batch_size, k))
        ratio = target_p / mx.maximum(draft_p, 1e-10)
        accepted = ratio >= rand  # [batch, k] bool

        # 3) proposal_lens mask: proposal 없는 위치는 reject
        position_indices = mx.arange(k)[None, :]  # [1, k]
        valid_mask = position_indices < proposal_lens[:, None]  # [batch, k]
        accepted = accepted & valid_mask

        # 4) Left-to-right masking: 첫 rejection 이후는 모두 reject
        # cumprod trick: [T, T, F, T] → [T, T, F, F]
        accepted_cumulative = mx.cumprod(accepted.astype(mx.float32), axis=1)
        accepted_mask = accepted_cumulative.astype(mx.bool_)  # [batch, k]

        # 5) Accept된 위치에 draft token 채우기
        output[:, :k] = mx.where(accepted_mask, draft_tokens, PLACEHOLDER_TOKEN_ID)

        # 6) 첫 rejection 위치에서 corrected distribution으로 resample
        num_accepted = accepted_mask.astype(mx.int32).sum(axis=1)  # [batch]
        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])
            if plen == 0:
                # proposal 없었음 → 일반 decode: target의 top token
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                # 첫 rejection 위치에서 corrected distribution으로 resample
                corrected = mx.maximum(
                    target_probs[b, n, :] - draft_probs[b, n, :],
                    0.0
                )
                total = corrected.sum()
                if total > 1e-10:
                    corrected = corrected / total
                    output[b, n] = mx.random.categorical(mx.log(corrected + 1e-10))
                else:
                    output[b, n] = mx.argmax(target_probs[b, n, :])
            else:
                # 전부 accept → bonus token from target[k]
                output[b, k] = mx.random.categorical(
                    mx.log(target_probs[b, plen, :] + 1e-10)
                )

        return output

    def _forward_loop(
        self,
        target_probs: mx.array,
        draft_probs: mx.array,
        draft_tokens: mx.array,
        proposal_lens: mx.array,
    ) -> mx.array:
        """
        루프 버전. 디버깅/검증용.
        벡터화 버전과 동일한 결과를 내야 함.
        """
        batch_size, k = draft_tokens.shape
        output = mx.full((batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32)

        for b in range(batch_size):
            plen = int(proposal_lens[b])
            if plen == 0:
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
                continue

            accepted_count = 0
            for i in range(plen):
                token = int(draft_tokens[b, i])
                p_target = float(target_probs[b, i, token])
                p_draft = float(draft_probs[b, i, token])

                r = float(mx.random.uniform())
                if p_draft > 1e-10 and (p_target / p_draft) >= r:
                    output[b, i] = token
                    accepted_count += 1
                else:
                    corrected = mx.maximum(
                        target_probs[b, i, :] - draft_probs[b, i, :], 0.0
                    )
                    total = float(corrected.sum())
                    if total > 1e-10:
                        corrected = corrected / total
                        output[b, i] = mx.random.categorical(mx.log(corrected + 1e-10))
                    else:
                        output[b, i] = mx.argmax(target_probs[b, i, :])
                    break
            else:
                # All accepted → bonus
                output[b, plen] = mx.random.categorical(
                    mx.log(target_probs[b, plen, :] + 1e-10)
                )

        return output
```

### 6.3 N-gram 전용 Greedy Verifier

N-gram은 draft probability가 없으므로 rejection sampling 대신 greedy/threshold 기반 verification 사용.

```python
class NGramVerifier:
    """
    N-gram proposer 전용 verifier.
    Draft probability 없이 target model output만으로 accept/reject 결정.
    """

    def __init__(self, mode: str = "greedy", threshold: float = 0.1):
        """
        Args:
            mode: "greedy" (target argmax == draft) 또는
                  "threshold" (target_prob[draft] >= threshold)
            threshold: threshold 모드에서 acceptance 기준 확률
        """
        self.mode = mode
        self.threshold = threshold

    def __call__(
        self,
        target_probs: mx.array,   # [batch, k+1, vocab]
        draft_tokens: mx.array,   # [batch, k]
        proposal_lens: mx.array,  # [batch]
    ) -> mx.array:
        batch_size, k = draft_tokens.shape
        output = mx.full((batch_size, k + 1), PLACEHOLDER_TOKEN_ID, dtype=mx.int32)

        if self.mode == "greedy":
            return self._greedy(target_probs, draft_tokens, proposal_lens, output)
        else:
            return self._threshold(target_probs, draft_tokens, proposal_lens, output)

    def _greedy(self, target_probs, draft_tokens, proposal_lens, output):
        """Target model의 argmax와 draft token이 일치하면 accept."""
        batch_size, k = draft_tokens.shape
        target_argmax = mx.argmax(target_probs[:, :k, :], axis=-1)  # [batch, k]

        match = (draft_tokens == target_argmax)  # [batch, k]

        # proposal_lens mask
        pos = mx.arange(k)[None, :]
        valid = pos < proposal_lens[:, None]
        match = match & valid

        # Left-to-right: 첫 불일치 이후 전부 reject
        match_cum = mx.cumprod(match.astype(mx.float32), axis=1).astype(mx.bool_)

        output[:, :k] = mx.where(match_cum, draft_tokens, PLACEHOLDER_TOKEN_ID)

        # 첫 rejection 위치에서 target argmax로 대체, 또는 bonus
        num_accepted = match_cum.astype(mx.int32).sum(axis=1)
        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])
            if plen == 0:
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                output[b, n] = target_argmax[b, n]
            else:
                output[b, plen] = mx.argmax(target_probs[b, plen, :])

        return output

    def _threshold(self, target_probs, draft_tokens, proposal_lens, output):
        """Target model이 draft token에 부여한 확률이 threshold 이상이면 accept."""
        batch_size, k = draft_tokens.shape
        target_p = mx.take_along_axis(
            target_probs[:, :k, :], draft_tokens[:, :, None], axis=2
        ).squeeze(-1)  # [batch, k]

        accepted = target_p >= self.threshold

        pos = mx.arange(k)[None, :]
        valid = pos < proposal_lens[:, None]
        accepted = accepted & valid

        accepted_cum = mx.cumprod(accepted.astype(mx.float32), axis=1).astype(mx.bool_)
        output[:, :k] = mx.where(accepted_cum, draft_tokens, PLACEHOLDER_TOKEN_ID)

        num_accepted = accepted_cum.astype(mx.int32).sum(axis=1)
        for b in range(batch_size):
            n = int(num_accepted[b])
            plen = int(proposal_lens[b])
            if plen == 0:
                output[b, 0] = mx.argmax(target_probs[b, 0, :])
            elif n < plen:
                output[b, n] = mx.argmax(target_probs[b, n, :])
            else:
                output[b, plen] = mx.argmax(target_probs[b, plen, :])

        return output
```

---

## 7. Phase 2: Draft Model Proposer (spec_decode/proposer/draft_model.py)

### 7.1 설계 근거

- mlx-lm의 `speculative_generate_step`에서 single-stream 로직 추출
- 이를 batch 모드로 확장: 모든 시퀀스에 대해 draft model을 동시에 실행
- MLX lazy evaluation으로 k step의 draft를 하나의 계산 그래프로 fusion
- Draft model은 target model과 별개로 로딩 (mlx-lm의 `load` 함수 사용)

### 7.2 구현

```python
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load

from .base import BaseProposer, ProposalResult


class DraftModelProposer(BaseProposer):
    """
    소형 draft model로 k개 토큰을 batch 생성.

    mlx-lm의 speculative_generate_step (batch=1)을 batch 모드로 확장.
    Draft model과 target model은 별도 KV cache를 유지.

    MLX lazy evaluation 활용:
    - k step의 draft를 mx.eval() 없이 연속 실행
    - 마지막에 한 번에 evaluate → 자동 operation fusion
    """

    def __init__(self, model_path: str, quantize: Optional[str] = None):
        self.model_path = model_path
        self.quantize = quantize
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Draft model 로딩. 엔진 초기화 시 1회 호출."""
        if self._loaded:
            return
        # mlx-lm의 load 함수로 모델 + 토크나이저 로딩
        self.model, self.tokenizer = mlx_load(self.model_path)
        self._loaded = True

    @property
    def needs_draft_probs(self) -> bool:
        return True  # rejection sampling에 draft probability 필요

    @property
    def requires_gpu(self) -> bool:
        return True  # Metal GPU 사용

    def propose(
        self,
        sequences: List["Sequence"],
        k: int,
    ) -> Optional[ProposalResult]:
        if not self._loaded:
            self.load()

        batch_size = len(sequences)
        all_draft_tokens: List[mx.array] = []
        all_draft_probs: List[mx.array] = []

        # ─── k step autoregressive draft generation ───
        for step in range(k):
            # 배치 입력 준비: 각 시퀀스의 현재 마지막 토큰
            if step == 0:
                input_ids = mx.array(
                    [seq.last_token_id for seq in sequences],
                    dtype=mx.int32,
                )[:, None]  # [batch, 1]
            else:
                input_ids = all_draft_tokens[-1][:, None]  # [batch, 1]

            # Draft model forward (배치)
            # ★ 기존 continuous batching의 forward와 동일한 인터페이스
            #    단, draft model의 KV cache 사용
            logits = self._batched_forward(sequences, input_ids, step)
            # logits: [batch, 1, vocab] → squeeze
            logits = logits[:, -1, :]  # [batch, vocab]

            probs = mx.softmax(logits, axis=-1)    # [batch, vocab]
            tokens = mx.random.categorical(mx.log(probs + 1e-10))  # [batch]

            all_draft_tokens.append(tokens)
            all_draft_probs.append(probs)

            # ⚠️ mx.eval() 호출하지 않음 → lazy evaluation으로 fusion

        # ─── 한 번에 evaluate ───
        draft_tokens = mx.stack(all_draft_tokens, axis=1)  # [batch, k]
        draft_probs = mx.stack(all_draft_probs, axis=1)    # [batch, k, vocab]
        mx.eval(draft_tokens, draft_probs)

        proposal_lens = mx.full((batch_size,), k, dtype=mx.int32)

        return ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            proposal_lens=proposal_lens,
        )

    def _batched_forward(
        self,
        sequences: List["Sequence"],
        input_ids: mx.array,
        step: int,
    ) -> mx.array:
        """
        Draft model의 batched forward pass.
        각 시퀀스별 draft KV cache를 유지하면서 forward.

        ★ 구현 시 기존 엔진의 batched forward 인터페이스에 맞춰 조정.
           draft model 전용 KV cache를 별도로 관리해야 함.
        """
        # TODO: draft model KV cache management + batched forward
        # 핵심: step=0이면 target model의 KV cache 상태에서 시작
        #        step>0이면 이전 draft step의 KV에 이어서 진행
        raise NotImplementedError("Draft model batched forward")

    def reset_draft_cache(self, request_ids: List[str]):
        """매 engine step 시작 시 draft KV cache 리셋."""
        # Draft cache는 일시적 → 매 step 새로 생성
        pass
```

### 7.3 Draft Model 메모리 관리

```
Draft model 메모리 레이아웃:
- Target model: 메인 KV cache (persistent, 요청 수명 동안 유지)
- Draft model: 임시 KV cache (매 engine step마다 리셋)

Step 흐름:
1. Draft model KV cache를 target model의 현재 상태에서 fork
   - Apple Silicon: unified memory → view/shallow copy 가능
   - CUDA에서는 deep copy 필요하지만 MLX에서는 불필요
2. Draft model이 k step 동안 자체 KV cache extend
3. Verification 완료 후 draft KV cache 폐기
4. Target model KV cache는 accepted token 수만큼만 extend
```

---

## 8. Phase 3: EAGLE Proposer (spec_decode/proposer/eagle.py)

### 8.1 설계 근거

- EAGLE: target model의 hidden states를 입력으로 받아 multi-token prediction
- 별도 모델이 아닌 추가 prediction head만 필요 → 메모리 효율적
- vLLM V1 roadmap에서 우선 지원 예정
- Draft model 대비 장점: vocab projection 이미 완료된 hidden states 활용 → 정확도 높음

### 8.2 구현

```python
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseProposer, ProposalResult


class EAGLEHead(nn.Module):
    """
    EAGLE prediction head.
    Target model의 hidden states를 입력으로 받아
    다음 여러 토큰의 확률 분포를 예측.

    구조: target hidden state → FC layers → vocab projection
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size

        layers = []
        for i in range(num_layers):
            in_size = hidden_size if i == 0 else intermediate_size
            out_size = intermediate_size if i < num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_size, out_size))
            if i < num_layers - 1:
                layers.append(nn.SiLU())

        self.fc = nn.Sequential(*layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Args:
            hidden_states: [batch, hidden_size] — target model의 마지막 hidden state

        Returns:
            logits: [batch, vocab_size]
        """
        h = self.fc(hidden_states)
        return self.lm_head(h)


class EAGLEProposer(BaseProposer):
    """
    EAGLE-style speculative decoding.

    Target model의 hidden states + 추가 prediction head로 draft token 생성.
    별도 모델 불필요 → prediction head만 로딩.

    autoregressive하게 k step 실행:
    1. target의 last hidden state → EAGLE head → 토큰 1 예측
    2. 예측된 토큰을 target model에 넣어 다음 hidden state 획득
    3. 그 hidden state로 EAGLE head → 토큰 2 예측
    4. ... k회 반복

    ★ 주의: step 2에서 target model의 forward가 필요하므로
       draft model보다 computation이 크지만, 정확도가 훨씬 높음.
       "self-speculation" 방식.

    대안 (MTP): target model 자체에 multi-token prediction head가 있는 경우
    (DeepSeek V3 등), 별도 EAGLE head 없이 모델 내장 MTP head 활용.
    """

    def __init__(
        self,
        target_model: nn.Module,
        head_path: Optional[str] = None,
        num_layers: int = 1,
    ):
        self.target_model = target_model
        self.head_path = head_path
        self.num_layers = num_layers
        self.eagle_head: Optional[EAGLEHead] = None
        self._loaded = False

    def load(self):
        """EAGLE head 로딩. 없으면 새로 초기화 (학습 필요)."""
        if self._loaded:
            return

        # Target model에서 hidden_size, vocab_size 추출
        # ★ 모델 아키텍처에 따라 접근 방식 다를 수 있음
        config = self.target_model.config if hasattr(self.target_model, 'config') else None
        hidden_size = getattr(config, 'hidden_size', 4096)
        vocab_size = getattr(config, 'vocab_size', 128256)

        self.eagle_head = EAGLEHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=self.num_layers,
        )

        if self.head_path:
            # 사전 학습된 EAGLE head 가중치 로딩
            weights = mx.load(self.head_path)
            self.eagle_head.load_weights(list(weights.items()))

        self._loaded = True

    @property
    def needs_draft_probs(self) -> bool:
        return True

    @property
    def requires_gpu(self) -> bool:
        return True

    def propose(
        self,
        sequences: List["Sequence"],
        k: int,
    ) -> Optional[ProposalResult]:
        if not self._loaded:
            self.load()

        batch_size = len(sequences)
        all_tokens: List[mx.array] = []
        all_probs: List[mx.array] = []

        # 초기 hidden state: target model의 마지막 hidden state
        # ★ 기존 엔진에서 forward 시 hidden state를 반환하도록 수정 필요
        hidden = self._get_last_hidden_states(sequences)  # [batch, hidden_size]

        for step in range(k):
            logits = self.eagle_head(hidden)  # [batch, vocab]
            probs = mx.softmax(logits, axis=-1)
            tokens = mx.random.categorical(mx.log(probs + 1e-10))

            all_tokens.append(tokens)
            all_probs.append(probs)

            # 다음 step을 위한 hidden state 업데이트
            # 예측된 토큰을 target model에 넣어 hidden state 획득
            hidden = self._get_hidden_for_token(sequences, tokens)

        draft_tokens = mx.stack(all_tokens, axis=1)
        draft_probs = mx.stack(all_probs, axis=1)
        mx.eval(draft_tokens, draft_probs)

        return ProposalResult(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            proposal_lens=mx.full((batch_size,), k, dtype=mx.int32),
        )

    def _get_last_hidden_states(self, sequences) -> mx.array:
        """
        각 시퀀스에 대한 target model의 마지막 hidden state 추출.
        ★ 기존 엔진 수정 필요: forward 시 hidden state 캐싱.
        """
        raise NotImplementedError(
            "Target model forward에서 hidden state 반환 인터페이스 필요"
        )

    def _get_hidden_for_token(self, sequences, tokens) -> mx.array:
        """
        예측된 토큰을 target model embedding + 일부 layer에 통과시켜
        hidden state 획득. Full forward 대비 lightweight.
        """
        raise NotImplementedError(
            "Target model의 lightweight hidden state 계산"
        )
```

### 8.3 EAGLE Training (별도 프로세스)

```python
# EAGLE head 학습은 serving과 별도 프로세스
# target model의 (hidden_state, next_token) 쌍을 데이터로 학습

def train_eagle_head(
    target_model_path: str,
    train_data_path: str,
    output_path: str,
    num_layers: int = 1,
    epochs: int = 3,
    lr: float = 1e-4,
):
    """
    EAGLE head 학습 스크립트.

    데이터: target model을 실행하면서 수집한 (hidden_state, next_token) 쌍
    손실: cross-entropy loss

    사용:
    python -m hwquant.spec_decode.train_eagle \\
        --target-model mlx-community/Qwen3-32B-4bit \\
        --train-data /path/to/data \\
        --output /path/to/eagle_head.safetensors
    """
    pass  # TODO: 학습 스크립트 구현
```

---

## 9. Phase 4: Dynamic Controller (spec_decode/dynamic_controller.py)

### 9.1 설계 근거

- vLLM 벤치마크: spec decode는 low QPS에서 1.5–2.8x speedup, high QPS에서 1.4–1.8x slowdown
- `disable_by_batch_size`: batch size threshold 초과 시 자동 OFF
- Acceptance rate EMA: 낮으면 speculation 강도 줄이거나 OFF
- Per-request granularity: request-level override 지원

### 9.2 구현

```python
from dataclasses import dataclass, field
import time
from typing import Dict, Optional

from .config import SpecDecodeConfig


@dataclass
class SpecDecodeStats:
    """Step별 통계."""
    total_proposed: int = 0
    total_accepted: int = 0
    total_steps: int = 0
    total_bonus_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def avg_tokens_per_step(self) -> float:
        """Step당 평균 생성 토큰 수. 1.0 = spec decode 효과 없음."""
        if self.total_steps == 0:
            return 1.0
        return (self.total_accepted + self.total_bonus_tokens) / self.total_steps


class DynamicSpecController:
    """
    배치 크기와 acceptance rate에 따라 speculation 강도를 동적 조절.

    결정 흐름:
    1. batch_size >= disable_threshold → spec OFF
    2. acceptance_rate_ema < acceptance_rate_threshold → spec OFF
    3. acceptance_rate_ema에 따라 k 조절:
       - > 0.8 → k = max (공격적)
       - 0.5–0.8 → k = max - 2 (보수적)
       - 0.3–0.5 → k = 1 (최소)
       - < 0.3 → OFF
    """

    def __init__(self, config: SpecDecodeConfig):
        self.config = config
        self.acceptance_rate_ema: float = 0.7  # 초기값 (optimistic)
        self.stats = SpecDecodeStats()

        # 최근 N step의 상세 기록 (디버깅 및 모니터링)
        self._recent_rates: list[float] = []
        self._max_recent = 100

    def should_speculate(self, batch_size: int) -> bool:
        """현재 조건에서 spec decode를 활성화할지 결정."""
        if self.config.spec_decode_mode == "none":
            return False

        # Batch size threshold
        if batch_size >= self.config.disable_by_batch_size:
            return False

        # Dynamic control이 꺼져 있으면 항상 ON
        if not self.config.dynamic_spec_decode:
            return True

        # Acceptance rate threshold
        return self.acceptance_rate_ema >= self.config.acceptance_rate_threshold

    def get_num_spec_tokens(self, batch_size: int) -> int:
        """배치 크기와 acceptance rate에 따라 k 결정."""
        if not self.should_speculate(batch_size):
            return 0

        k = self.config.num_speculative_tokens

        if not self.config.adaptive_k:
            return k

        # Adaptive k: acceptance rate에 따라 조절
        if self.acceptance_rate_ema > 0.8:
            return k                          # 최대 speculation
        elif self.acceptance_rate_ema > 0.5:
            return max(1, k - 2)              # 보수적
        elif self.acceptance_rate_ema > 0.3:
            return 1                          # 최소
        else:
            return 0                          # OFF

    def update(self, num_proposed: int, num_accepted: int, num_bonus: int = 0):
        """
        매 engine step 후 호출.

        Args:
            num_proposed: 배치 전체에서 제안된 총 draft 토큰 수
            num_accepted: 배치 전체에서 accept된 총 토큰 수
            num_bonus: 배치 전체에서 생성된 bonus 토큰 수
        """
        self.stats.total_proposed += num_proposed
        self.stats.total_accepted += num_accepted
        self.stats.total_bonus_tokens += num_bonus
        self.stats.total_steps += 1

        if num_proposed > 0:
            step_rate = num_accepted / num_proposed
            alpha = self.config.acceptance_rate_ema_alpha
            self.acceptance_rate_ema = alpha * step_rate + (1 - alpha) * self.acceptance_rate_ema

            self._recent_rates.append(step_rate)
            if len(self._recent_rates) > self._max_recent:
                self._recent_rates.pop(0)

    def get_metrics(self) -> Dict:
        """모니터링용 메트릭 반환. /metrics endpoint 등에서 사용."""
        return {
            "spec_decode_enabled": self.config.spec_decode_mode != "none",
            "spec_decode_mode": self.config.spec_decode_mode,
            "acceptance_rate_ema": round(self.acceptance_rate_ema, 4),
            "acceptance_rate_overall": round(self.stats.acceptance_rate, 4),
            "avg_tokens_per_step": round(self.stats.avg_tokens_per_step, 2),
            "total_steps": self.stats.total_steps,
            "total_proposed": self.stats.total_proposed,
            "total_accepted": self.stats.total_accepted,
            "total_bonus_tokens": self.stats.total_bonus_tokens,
            "current_k": self.config.num_speculative_tokens,
        }
```

---

## 10. KV Cache Manager (spec_decode/kv_manager.py)

```python
from typing import Dict, List

import mlx.core as mx


class SpecDecodeKVManager:
    """
    Spec decode 전용 KV cache 관리.
    기존 continuous batching 엔진의 KV manager를 확장.

    Apple Silicon 핵심 이점:
    - Unified memory → zero-copy, KV 전송 불필요
    - Rollback = cache length 조정 (거의 무비용)
    - vLLM의 PagedAttention block deallocation 불필요
    """

    def __init__(self, base_kv_manager):
        """
        Args:
            base_kv_manager: 기존 continuous batching 엔진의 KV manager 참조
        """
        self.base = base_kv_manager

    def pre_allocate(self, request_id: str, num_tokens: int):
        """
        Verification을 위해 KV cache slots 사전 할당.
        MLX lazy allocation → 실제 forward 실행 전까지 메모리 점유 없음.
        """
        self.base.allocate_slots(request_id, num_tokens)

    def rollback(self, request_id: str, accepted_count: int, total_proposed: int):
        """
        Rejection 발생 시 KV cache rollback.

        CUDA (vLLM): block deallocation (PagedAttention)
        MLX: cache sequence length 조정만 → near zero-cost

        Args:
            request_id: 요청 ID
            accepted_count: accept된 토큰 수
            total_proposed: 전체 제안된 토큰 수
        """
        rejected_count = total_proposed - accepted_count
        if rejected_count > 0:
            # 기존 KV manager에 truncate 요청
            self.base.truncate(request_id, num_tokens_to_remove=rejected_count)

    def commit(self, request_id: str, num_accepted: int):
        """
        Accept된 토큰의 KV를 영구 확정.
        Target model verification에서 이미 계산된 KV를 유지.
        """
        self.base.confirm_extension(request_id, num_accepted)

    def reset_draft_cache(self, request_ids: List[str]):
        """
        Draft model KV cache 리셋 (Phase 2용).
        매 engine step 시작 시 호출.
        """
        # Draft cache는 일시적 → step마다 새로 생성
        for rid in request_ids:
            if hasattr(self.base, 'reset_draft'):
                self.base.reset_draft(rid)
```

---

## 11. Engine 통합 (engine/engine.py 수정)

기존 continuous batching 엔진의 `step()` 메서드에 spec decode를 삽입.

```python
# engine/engine.py 수정 사항

from spec_decode.config import SpecDecodeConfig
from spec_decode.proposer.base import create_proposer, ProposalResult
from spec_decode.verifier import BatchedVerifier
from spec_decode.rejection_sampler import (
    BatchedRejectionSampler, NGramVerifier, PLACEHOLDER_TOKEN_ID
)
from spec_decode.dynamic_controller import DynamicSpecController
from spec_decode.kv_manager import SpecDecodeKVManager


class ServingEngine:
    def __init__(self, model, tokenizer, config, spec_config: SpecDecodeConfig):
        # ... 기존 초기화 ...

        # ─── Spec decode 초기화 ───
        self.spec_config = spec_config
        self.proposer = create_proposer(spec_config, target_model=model)
        self.verifier = BatchedVerifier(model, tokenizer) if self.proposer else None
        self.rejection_sampler = BatchedRejectionSampler()
        self.ngram_verifier = NGramVerifier(mode="greedy")
        self.dynamic_controller = DynamicSpecController(spec_config)
        self.spec_kv_manager = SpecDecodeKVManager(self.kv_manager)

        # Draft model 로딩 (Phase 2)
        if self.proposer and hasattr(self.proposer, 'load'):
            self.proposer.load()

    async def step(self):
        """
        메인 엔진 루프 1 step.
        Spec decode가 활성화되어 있으면 propose → verify → accept/reject 흐름.
        비활성화면 기존 continuous batching 로직 그대로.
        """
        # ═══ Phase 0: Schedule ═══
        scheduled_requests = self.scheduler.schedule()
        if not scheduled_requests:
            return

        prefill_reqs = [r for r in scheduled_requests if r.state == "PREFILL"]
        decode_reqs = [r for r in scheduled_requests if r.state == "DECODE"]

        # Prefill 처리 (spec decode 적용 안 함)
        if prefill_reqs:
            await self._batched_prefill(prefill_reqs)
            # Prefill 완료된 시퀀스의 n-gram index 초기화 (Phase 1)
            if self.spec_config.spec_decode_mode == "ngram":
                for req in prefill_reqs:
                    self._init_ngram_index(req)

        if not decode_reqs:
            return

        # Spec decode 활성화 여부 결정
        batch_size = len(decode_reqs)
        use_spec = (
            self.proposer is not None
            and self.dynamic_controller.should_speculate(batch_size)
        )

        if not use_spec:
            # 기존 continuous batching 로직
            await self._normal_batched_decode(decode_reqs)
            return

        # ═══ Phase 1: Propose ═══
        k = self.dynamic_controller.get_num_spec_tokens(batch_size)
        if k == 0:
            await self._normal_batched_decode(decode_reqs)
            return

        proposal = self.proposer.propose(decode_reqs, k)
        if proposal is None:
            # 전체 배치 proposal 실패 → 일반 decode
            await self._normal_batched_decode(decode_reqs)
            return

        # KV cache pre-allocation
        for i, req in enumerate(decode_reqs):
            plen = int(proposal.proposal_lens[i])
            if plen > 0:
                self.spec_kv_manager.pre_allocate(req.id, plen + 1)

        # ═══ Phase 2: Verify ═══
        target_probs = self.verifier.verify(
            decode_reqs, proposal.draft_tokens, proposal.proposal_lens
        )

        # ═══ Phase 3: Accept / Reject ═══
        if self.proposer.needs_draft_probs:
            # Draft model / EAGLE → rejection sampling
            accepted_tokens = self.rejection_sampler(
                target_probs,
                proposal.draft_probs,
                proposal.draft_tokens,
                proposal.proposal_lens,
            )
        else:
            # N-gram → greedy verification
            accepted_tokens = self.ngram_verifier(
                target_probs,
                proposal.draft_tokens,
                proposal.proposal_lens,
            )

        # ═══ Phase 4: Postprocess ═══
        total_proposed = 0
        total_accepted = 0
        total_bonus = 0

        for i, req in enumerate(decode_reqs):
            tokens_row = accepted_tokens[i]  # [k+1]
            plen = int(proposal.proposal_lens[i])

            # -1 제거하여 유효 토큰만 추출
            valid_mask = tokens_row != PLACEHOLDER_TOKEN_ID
            valid_tokens = tokens_row[valid_mask]
            n_valid = int(valid_tokens.shape[0])

            # 시퀀스에 토큰 추가
            req.append_tokens(valid_tokens)

            # KV cache 정리
            if plen > 0:
                if n_valid <= plen:
                    # 일부 reject → rollback
                    self.spec_kv_manager.rollback(req.id, n_valid, plen + 1)
                else:
                    # 전부 accept + bonus
                    self.spec_kv_manager.commit(req.id, n_valid)
                    total_bonus += 1

            total_proposed += plen
            total_accepted += min(n_valid, plen)

            # N-gram index incremental update
            if self.spec_config.spec_decode_mode == "ngram" and n_valid > 0:
                self._update_ngram_index(req, n_valid)

            # 완료 체크
            if req.is_finished():
                self.scheduler.finish(req)
                await self._yield_output(req)

        # 통계 업데이트
        self.dynamic_controller.update(total_proposed, total_accepted, total_bonus)

        # Draft model cache 리셋 (Phase 2)
        if self.proposer and self.proposer.requires_gpu:
            self.spec_kv_manager.reset_draft_cache([r.id for r in decode_reqs])

    def _init_ngram_index(self, req):
        """Prefill 완료 후 n-gram suffix index 초기화."""
        from spec_decode.proposer.ngram import NGramProposer
        context = req.prompt_tokens + req.generated_tokens
        req._ngram_index = NGramProposer.build_suffix_index(
            context, self.spec_config.ngram_max
        )
        req._ngram_dirty = False

    def _update_ngram_index(self, req, new_token_count):
        """새 토큰 추가 시 n-gram index incremental 업데이트."""
        from spec_decode.proposer.ngram import NGramProposer
        context = req.prompt_tokens + req.generated_tokens
        NGramProposer.update_suffix_index(
            req._ngram_index, context, new_token_count, self.spec_config.ngram_max
        )
```

---

## 12. API Endpoint 수정 (serve.py)

```python
# /metrics endpoint에 spec decode 통계 추가
@app.get("/v1/spec_decode/metrics")
async def spec_decode_metrics():
    return engine.dynamic_controller.get_metrics()

# /v1/chat/completions에서 extra_body override 처리
async def handle_chat_completion(request: ChatCompletionRequest):
    # request-level spec decode override
    spec_override = getattr(request, 'extra_body', {})
    if 'spec_decode' in spec_override:
        # 이 request에 대해서만 spec decode 모드 변경
        # ★ per-request config는 별도 구현 필요
        pass
```

---

## 13. Source Code References

이식 대상 vLLM/mlx-lm 소스코드:

| Component | Source File | 이식 핵심 |
|-----------|-----------|----------|
| V0 Spec Worker | `vllm/spec_decode/spec_decode_worker.py` | `disable_by_batch_size`, proposer/scorer 패턴 |
| V0 Batch Expansion | `vllm/spec_decode/batch_expansion.py` | 시퀀스 복제 verification (Apple Silicon에서는 불필요) |
| V1 Rejection Sampler | `vllm/v1/sample/rejection_sampler.py` | -1 padding, bonus token, 벡터화 |
| V1 Scheduler | `vllm/v1/core/sched/scheduler.py` | `{req_id: num_tokens}` 통합 예산 |
| V1 Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `_calc_spec_decode_metadata()` |
| NGram Worker | `vllm/spec_decode/ngram_worker.py` | Draft-free n-gram speculation 알고리즘 |
| Top1 Proposer | `vllm/spec_decode/top1_proposer.py` | Non-spec sequence 처리, proposal 길이 0 핸들링 |
| mlx-lm Spec Gen | `mlx_lm/generate.py` → `speculative_generate_step` | 단일 시퀀스 spec decode 로직 |
| mlx-lm Batch Gen | `mlx_lm/generate.py` | 배치 생성 인프라 |
| vllm-mlx Paper | `arxiv.org/html/2601.19139v2` | Continuous batching 아키텍처 |

---

## 14. CUDA vs Apple Silicon 차이 정리

구현 시 vLLM 코드를 그대로 이식하면 안 되는 부분:

| vLLM (CUDA) | Apple Silicon (MLX) 대응 |
|-------------|-------------------------|
| PagedAttention block alloc/dealloc | 연속 메모리 + length 기반 관리 |
| GPU↔CPU explicit KV copy | Zero-copy (unified memory) |
| Batch expansion (tensor 복제) | Padding + mask (복제 불필요) |
| CUDA graphs (draft step fusion) | MLX lazy eval → 자동 fusion |
| Block table management | 불필요 → OS-level paging |
| `torch.where` / `torch.scatter` | `mx.where` / `mx.take_along_axis` |
| `torch.multinomial` | `mx.random.categorical` |

---

## 15. 구현 순서 요약

### 15.1 Phase 1 (N-gram) — 즉시 시작

1. `spec_decode/config.py` — SpecDecodeConfig + CLI args
2. `spec_decode/proposer/base.py` — BaseProposer + factory
3. `spec_decode/proposer/ngram.py` — NGramProposer
4. `spec_decode/rejection_sampler.py` — NGramVerifier (greedy)
5. `spec_decode/verifier.py` — BatchedVerifier (기존 forward 확장)
6. `spec_decode/dynamic_controller.py` — DynamicSpecController
7. `spec_decode/kv_manager.py` — SpecDecodeKVManager
8. `engine/engine.py` 수정 — step()에 spec decode 삽입
9. `serve.py` 수정 — CLI args + /metrics endpoint
10. 테스트

### 15.2 Phase 2 (Draft Model) — Phase 1 안정화 후

1. `spec_decode/proposer/draft_model.py` — DraftModelProposer
2. Draft model 로딩 (mlx-lm의 `load`)
3. Draft model KV cache 관리
4. `rejection_sampler.py`의 `BatchedRejectionSampler` 활성화
5. Engine에서 draft model forward 지원
6. 테스트

### 15.3 Phase 3 (EAGLE) — Phase 2 완료 후

1. `spec_decode/proposer/eagle.py` — EAGLEHead + EAGLEProposer
2. Target model에서 hidden state 반환 인터페이스 추가
3. EAGLE head 학습 스크립트
4. 테스트

### 15.4 Phase 4 (Dynamic Control 고도화) — 전 Phase와 병행

1. Adaptive k 구현 (이미 기본 구조 있음)
2. Per-request override 지원
3. /metrics endpoint 확장
4. 분산 추론 환경 spec decode 조율 (Mac Studio cluster)
