# GPT PRO Audit Fix Plan (v2 — Codex + GPT2 반영)

> GPT PRO 1차(10건) + GPT PRO 2차(8건) + Codex 코멘트(9건) + 8명 Researcher 코드 분석 결과를 종합한 최종 수정 플랜.

---

## 전체 이슈 맵 (19건 → 7 Phase)

| Phase | 이슈 수 | 성격 | 예상 변경량 |
|-------|---------|------|-----------|
| **A** | 3 | 정확도 직결 버그 (CRITICAL) | ~30줄 |
| **B** | 3 | 리소스 안전성 (HIGH) | ~25줄 |
| **C** | 3 | OpenAI API 호환성 (HIGH) | ~80줄 |
| **D** | 2 | 캐시 방어 로직 (MEDIUM) | ~40줄 |
| **E** | 3 | 구조 개선 (MEDIUM) | ~45줄 |
| **F** | 3 | 성능 최적화 (MEDIUM) | ~70줄 |
| **G** | 2 | SSD 기능 확장 (LOW-MEDIUM) | ~80줄 |

---

## Phase A: 정확도 직결 버그 (CRITICAL)

### A1: 채팅 토크나이즈 special token 중복

**문제:** `server.py:354`에서 `tok.encode(prompt_text)` 호출 시 `add_special_tokens` 미지정.
`_format_chat_messages()`가 `apply_chat_template(tokenize=False)`로 이미 special token이 포함된 문자열을 반환하므로, BOS/EOS가 이중 삽입됨.

**Codex 보완:** `SimpleTokenizer.encode()`는 `add_special_tokens` 인자를 받지 않아 TypeError 발생 → fallback chain 필요.

**Codex 2차 보완 (반영):** chat 경로만 적용. completions 경로는 plain prompt라 BOS가 정당하게 필요할 수 있으므로 별도 검증 후 적용.

**Codex 3차 보완 (반영):** `tokenize=True` 선택적 시도 + fallback. 결과가 `list[int]`이면 직접 사용, `str`이면 `_safe_encode` fallback. 인터페이스 변경 없이 의미적 정확도 향상. SimpleTokenizer는 str을 반환하므로 자연스럽게 fallback 경로 사용.

**수정 (server.py):**
```python
# 새 헬퍼 함수 추가
def _safe_encode(tokenizer, text: str) -> list[int]:
    """Encode text, suppressing special tokens if the tokenizer supports it."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)

# server.py:353-354 (chat 경로만 적용)
# 1순위: tokenize=True로 토큰 직접 반환 시도
result = _format_chat_messages(body.messages, tok, tokenize=True)
if isinstance(result, list):
    prompt_tokens = result  # 실제 토크나이저 — 토큰 직접 반환
else:
    prompt_tokens = _safe_encode(tok, result)  # SimpleTokenizer 등 — str fallback

# server.py:397 (completions 경로) — 변경하지 않음 (BOS 필요할 수 있음)
```

**`_format_chat_messages` 수정 필요:** `tokenize` 파라미터를 전달하도록 변경.
```python
def _format_chat_messages(messages, tokenizer, tokenize=False):
    # apply_chat_template(tokenize=tokenize, ...) 로 전달
```

**파일:** `mlx_lm_server/server.py` — 헬퍼 추가 + chat 경로 1곳만 변경
**테스트 영향:** 없음 — SimpleTokenizer는 TypeError fallback 경로 사용

---

### A2: 캐시 저장 시 KV 길이 불일치 (Sequence Cache)

**문제:** `_store_finished_caches()`에서 key=`prompt_tokens`, value=`prompt_cache`인데, 생성 완료 시점의 prompt_cache는 prompt+generated 토큰의 KV를 모두 포함. 키보다 캐시가 길어 재사용 시 KV 위치 불일치.

**Codex 보완:** 커스텀 `_trim_kv_cache` 대신, scheduler.py가 이미 import 중인 `can_trim_prompt_cache`/`trim_prompt_cache` 사용. 이 함수들은 KVCache, QuantizedKVCache, CacheList 등 모든 캐시 타입을 지원.

**Codex 3차 보완 (반영):** `offset` 추정 대신 `len(seq.output_tokens)` 기반 trim. 캐시 타입별 offset 유무에 의존하지 않아 더 견고.

**수정 (scheduler.py `_store_finished_caches` 내, ~line 550):**
```python
if self._sequence_cache is not None:
    # prompt_cache는 생성 토큰까지 포함 — 생성된 토큰 수만큼 trim
    num_generated = len(seq.output_tokens)
    if num_generated > 0 and prompt_cache is not None:
        if can_trim_prompt_cache(prompt_cache):
            trim_prompt_cache(prompt_cache, num_generated)
        else:
            # non-trimmable 캐시 → KV 길이 불일치 방지를 위해 저장 스킵
            logger.warning("Cannot trim prompt cache for %s — skipping sequence cache store", seq.request_id)
            prompt_cache = None  # store 건너뜀
    if prompt_cache is not None:
        self._sequence_cache.store(prompt_tokens, prompt_cache)
```

**파일:** `mlx_lm_server/scheduler.py`
**신규 import:** 없음 (`can_trim_prompt_cache`, `trim_prompt_cache` 이미 import됨)

---

### A3: Full cache hit 시 마지막 토큰 중복 (공통 경로)

**문제:** `_insert_new_requests_batch()`에서 block cache 또는 sequence cache가 full hit하여 `remaining_tokens = []`이 되면, `remaining_tokens = [last_token]`으로 설정하지만 cache에 이미 해당 토큰의 KV가 포함 → RoPE positional encoding이 다른 KV 중복.

**Codex 보완:** block cache뿐 아니라 sequence cache full hit에도 동일 적용. 공통 지점(line 727-728)에서 처리.

**수정 (scheduler.py `_insert_new_requests_batch`, line 727-728):**
```python
if not remaining_tokens:
    if cache is not None and can_trim_prompt_cache(cache):
        # 캐시를 1토큰 trim하여 마지막 토큰 중복 방지
        remaining_tokens = [seq.token_ids[-1]]
        trim_prompt_cache(cache, 1)
    else:
        # non-trimmable 캐시 → full-hit이지만 trim 불가 → uncached 경로로 fallback
        logger.debug("Full cache hit but non-trimmable — falling back to uncached path")
        if block_ids:
            self.kv_cache_manager.free_blocks(block_ids)
        seq.block_ids = []
        cache = None
        remaining_tokens = seq.token_ids
```

**파일:** `mlx_lm_server/scheduler.py`
**A2와 동일 함수(`trim_prompt_cache`) 재사용.**

---

## Phase B: 리소스 안전성 (HIGH)

### B1: get_result() timeout 시 부분 결과 반환 + 리소스 누수

**문제 (GPT2-1):** `get_result()`에서 `event.wait(timeout)` 반환값(True/False) 미확인. timeout 시에도 `_results`를 pop하여 부분 결과 반환. 요청은 스케줄러에서 계속 실행되나 결과 버퍼가 사라져 토큰이 유실됨.

**수정 (scheduler.py `get_result`, ~line 281-296):**
```python
def get_result(self, request_id, timeout=None):
    """Wait for and return generation results.

    API Contract:
    - Normal: blocks until finished, returns list[TokenEvent]
    - Timeout: raises TimeoutError — caller MUST call cancel_request() to free resources
    - After cancel: may raise KeyError (already cleaned up) or return [cancelled_event]
    - Recommended pattern: on TimeoutError → cancel_request() → do NOT call get_result() again
    """
    with self._results_lock:
        event = self._results_ready.get(request_id)
    if event is None:
        raise KeyError(f"Unknown request_id: {request_id}")
    if not event.wait(timeout=timeout):
        raise TimeoutError(f"Request {request_id} timed out after {timeout}s")
    with self._results_lock:
        tokens = self._results.pop(request_id, [])
        self._results_ready.pop(request_id, None)
    return tokens
```

**서버 측 (server.py `_do_inference`, ~line 309):**
```python
try:
    events = await loop.run_in_executor(
        None, lambda: sched.get_result(request_id, timeout=timeout)
    )
except TimeoutError:
    sched.cancel_request(request_id)
    raise HTTPException(status_code=504, detail="Request timed out")
```

기존 `if not events:` 504 처리를 이 try/except로 교체.

**Codex 3차 보완 (반영 — 기존 "완전" 판정 수정):** active request 취소 경로에서 `_cleanup_result_buffers()` 미호출 확인.

**코드 추적 결과:**
1. `cancel_request()` active 경로 (line 317-321): `_cancelled`에 추가 후 `True` 반환 — `_cleanup_result_buffers()` 미호출
2. `_process_cancellations_batch()` (line 589-618): `_signal_finish()`만 호출 — `_cleanup_result_buffers()` 미호출
3. `_signal_finish()` (line 1138-1157): `_results`에 finish event append + ready.set() — pop 없음
4. **결과:** `_results[rid]`와 `_results_ready[rid]`가 영구 잔류 → 메모리 누수

**수정:** `_process_cancellations_batch()`에서 `_signal_finish()` 호출 후 `_cleanup_result_buffers()` 추가.
```python
# scheduler.py _process_cancellations_batch, ~line 616 이후:
if seq is not None:
    seq.is_finished = True
    seq.finish_reason = "cancelled"
    self._signal_finish(rid, finish_reason="cancelled")
    self._cleanup_result_buffers(rid)  # 추가: 취소된 active 요청의 결과 버퍼 정리
```

**주의:** `_signal_finish()`가 먼저 실행되어 streaming 클라이언트에 finish event를 전달한 후, `_cleanup_result_buffers()`가 non-streaming 결과 버퍼를 정리. 순서 중요.

**파일:** `mlx_lm_server/scheduler.py`, `mlx_lm_server/server.py`

---

### B2: Stream 등록 누수 (submit 실패 시 _streams 미정리)

**문제 (GPT2-2):** `register_stream()` → `submit_request()` 실패 시 `_streams[request_id]`가 남음. `cancel_request()`는 요청을 못 찾은 경우 `_streams`를 정리하지 않음.

**수정 (scheduler.py `cancel_request`, ~line 324-325):**
```python
# 요청을 찾지 못한 경우 (기존 코드 끝부분)
self._cleanup_result_buffers(request_id)
# 추가: 고아 stream 정리
with self._streams_lock:
    self._streams.pop(request_id, None)
return False
```

**파일:** `mlx_lm_server/scheduler.py` — 2줄 추가

---

### B3: Scheduler.stop() 스레드 join 후 핸들 소실

**문제 (GPT2-8):** `stop()`에서 `join(timeout=5.0)` 후 스레드 생존 여부와 관계없이 `_inference_thread = None` 설정. 좀비 스레드 감지 불가.

**수정 (scheduler.py `stop`, ~line 346-357):**
```python
def stop(self):
    self._running = False
    self._new_request_event.set()
    if self._inference_thread is not None:
        self._inference_thread.join(timeout=5.0)
        if self._inference_thread.is_alive():
            logger.warning(
                "Inference thread did not stop within 5s — may be stuck in model inference"
            )
            # 핸들 유지하여 호출자가 감지 가능
        else:
            self._inference_thread = None
    # ... batch_generator cleanup ...
```

**파일:** `mlx_lm_server/scheduler.py`

---

## Phase C: OpenAI API 호환성 (HIGH)

### C1: 스트리밍 stop sequence 누출 + 버퍼링

**문제:** `_stream_response()`에서 토큰을 즉시 SSE로 전송. 여러 토큰에 걸친 stop sequence가 클라이언트에 누출됨. 비스트림은 정상 처리.

**수정 (server.py `_stream_response` 리팩터):**

핵심 알고리즘:
1. `stop_sequences`가 없으면 기존 동작 유지 (zero overhead)
2. 있으면 `max_stop_len - 1`만큼 텍스트 버퍼 유지
3. 새 토큰 도착: 버퍼에 추가 → stop 여부 확인 → 안전한 prefix만 flush
4. stop 감지: stop 앞 텍스트만 전송 + `cancel_request()` 호출
5. 자연 종료: 버퍼 최종 flush (stop 최종 확인 포함)

```python
max_stop_len = max((len(s) for s in (inf_req.stop_sequences or [])), default=0)
text_buffer = ""

# 루프 내:
text_buffer += token_text

# stop 확인
stop_found = False
for stop_seq in (inf_req.stop_sequences or []):
    idx = text_buffer.find(stop_seq)
    if idx != -1:
        safe_text = text_buffer[:idx]
        if safe_text:
            yield format_chunk(request_id, model_name, safe_text, None)
        yield format_chunk(request_id, model_name, "", "stop")
        stop_found = True
        scheduler.cancel_request(request_id)
        break

if stop_found:
    break

# 안전한 prefix flush
if max_stop_len > 0 and len(text_buffer) > max_stop_len - 1:
    safe_len = len(text_buffer) - (max_stop_len - 1)
    yield format_chunk(request_id, model_name, text_buffer[:safe_len], None)
    text_buffer = text_buffer[safe_len:]
elif max_stop_len == 0:
    # stop sequences 없으면 즉시 전송 (기존 동작)
    if text_buffer:
        yield format_chunk(request_id, model_name, text_buffer, event.finish_reason)
        text_buffer = ""
```

**필수 테스트 케이스 3건:**
1. stop 문자열이 여러 토큰 경계에 걸쳐 나타남 ("ST" + "OP" → "STOP")
2. 여러 stop 후보 중 가장 먼저 완성된 것 우선 (["END", "ENDING"])
3. 유니코드/멀티바이트 텍스트 경계 ("停" + "止" → "停止")

**파일:** `mlx_lm_server/server.py` (~50줄), 테스트 파일

---

### C2: EOS 필터링을 token_id 기반으로 전환

**문제 (GPT2-6):** server.py의 EOS 필터링이 `token_text == tokenizer.eos_token` (문자열 비교)에 의존. detokenizer가 EOS를 빈 문자열이나 다른 표현으로 만들면 EOS가 출력에 포함됨.

**수정 (server.py):**
```python
# 새 헬퍼 함수
def _get_eos_token_ids(tokenizer) -> set[int]:
    """토크나이저에서 EOS token ID set를 안전하게 추출."""
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids, (set, frozenset, list)):
        return set(eos_ids)
    if isinstance(eos_ids, int):
        return {eos_ids}
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return {eos_id}
    return set()

# _do_inference (line ~317-322): 문자열 비교 → token_id 비교
eos_ids = _get_eos_token_ids(tok) if tok else set()
if events and eos_ids and events[-1].token_id in eos_ids:
    filtered_events = events[:-1]

# _stream_response (line ~583-588): 동일하게 token_id 비교
# eos_ids를 루프 전에 한 번만 계산
if eos_ids and event.token_id in eos_ids:
    token_text = ""
```

**전제:** `TokenEvent`에 `token_id` 필드가 존재해야 함 (현재 존재 확인 필요, 없으면 추가).

**파일:** `mlx_lm_server/server.py`

---

### C3: 스트리밍 첫 토큰 timeout 분리

**문제 (GPT2-7):** `_stream_response()`에서 `token_queue.get(timeout=request_timeout_s)`가 모든 토큰에 동일 적용. 긴 프롬프트(16k+)에서 prefill 시간이 길면 첫 토큰 전에 timeout.

**수정:**
```python
# config.py — 새 필드
first_token_timeout_s: float = 300.0  # 첫 토큰 대기 (prefill 포함)

# server.py _stream_response 루프
first_token = True
while True:
    timeout = first_token_timeout_s if first_token else request_timeout_s
    event = await loop.run_in_executor(
        None, lambda t=timeout: token_queue.get(timeout=t)
    )
    first_token = False
    # ... rest of loop ...
```

**파일:** `mlx_lm_server/config.py`, `mlx_lm_server/server.py`
**CLI 인자 추가:** `--first-token-timeout-s`

---

## Phase D: 캐시 방어 로직 (MEDIUM)

### D1: Block data 부분 재구성 방지 (스케줄러 측 검증)

**문제 (GPT2-4 + Codex C1):** block_ids 중 일부의 `kv_data`가 None이면 `block_data`가 `block_ids`보다 짧아지는데, 그대로 `reconstruct_cache_from_blocks()` 호출 → KV 위치 불일치로 silent corruption.

**Codex 보완:** `find_cached_prefix()` 변경 대신 스케줄러 쪽에서 검증. 기존 테스트/의도를 보존.

**수정 (scheduler.py `_insert_new_requests_batch`, ~line 662-673):**
```python
block_data = []
all_blocks_valid = True
for i in range(len(block_ids)):
    block = self.kv_cache_manager.pool.blocks[block_ids[i]]
    if block.kv_data is not None:
        block_data.append(block.kv_data)
    else:
        all_blocks_valid = False
        break  # 중간 갭 → 부분 prefix 사용 불가

if all_blocks_valid and block_data:
    cache = reconstruct_cache_from_blocks(...)
    remaining_tokens = seq.token_ids[num_cached:]
else:
    # 부분/미완성 → uncached 경로로 폴백
    if block_ids:
        self.kv_cache_manager.free_blocks(block_ids)
    seq.block_ids = []
    num_cached = 0
    cache = None
    remaining_tokens = seq.token_ids
    logger.warning("Partial block data for %s — falling back to uncached", seq.request_id)
```

**파일:** `mlx_lm_server/scheduler.py`

---

### D2: deepcopy → 단계적 slice clone (sequence cache)

**문제:** `copy.deepcopy(prompt_cache)`가 MLX array에서 안전한 독립 사본을 보장하지 않을 수 있음.

**Codex 보완:** known type만 fast path, 나머지는 deepcopy fallback 유지.

**GPT2-3 추가 발견:**
- `_merge_caches()`는 원본을 mutate하지 않음 (confirmed)
- `store()`는 이미 `list(prompt_cache)`으로 최적화됨 (line 140-141)
- **주 최적화 대상: `find_longest_prefix()` line 108의 deepcopy 1곳**

**Codex 2차 보완 (반영):** `QuantizedKVCache`는 tuple 상태를 사용하여 `keys[:,:,:offset,:]` 방식이 작동하지 않음. plain `KVCache`만 fast path, 나머지(QuantizedKVCache, CacheList 등)는 deepcopy fallback 유지.

**수정 (sequence_cache.py):**
```python
def _clone_cache_list(cache_list: list) -> list:
    """KV 캐시 리스트의 독립 사본 생성 (plain KVCache만 fast path, 나머지 deepcopy)."""
    cloned = []
    for obj in cache_list:
        # plain KVCache만 fast path — keys/values가 mx.array이고 group_size 없는 경우
        if (hasattr(obj, 'keys') and hasattr(obj, 'values') and hasattr(obj, 'offset')
                and not hasattr(obj, 'group_size')):  # QuantizedKVCache 제외
            new_obj = type(obj).__new__(type(obj))
            new_obj.keys = obj.keys[:, :, :obj.offset, :]
            new_obj.values = obj.values[:, :, :obj.offset, :]
            new_obj.offset = obj.offset
            # step 등 기본 스칼라 속성 복사
            if hasattr(obj, 'step'):
                new_obj.step = obj.step
            cloned.append(new_obj)
        else:
            # QuantizedKVCache, CacheList, dict, 기타 → deepcopy fallback
            import copy
            cloned.append(copy.deepcopy(obj))
    return cloned
```

**적용:**
- `find_longest_prefix():108` — `copy.deepcopy(cache_ref)` → `_clone_cache_list(cache_ref)` (주 최적화 대상)
- `store():144,146` — 기존 `list()` 최적화 유지, deepcopy fallback만 `_clone_cache_list`로 교체
- QuantizedKVCache/CacheList fast path는 별도 검증 + 벤치 후 확대

**파일:** `mlx_lm_server/sequence_cache.py`

---

## Phase E: 구조 개선 (MEDIUM)

### E1: _tiered_cache 생성자 주입

**문제:** `__main__.py:45`에서 `scheduler._tiered_cache = tiered_cache` — private attr 주입.

**수정:**
```python
# scheduler.py __init__ 파라미터에 추가:
def __init__(self, config, model=None, tokenizer=None,
             kv_cache_manager=None, tiered_cache=None):
    ...
    self._tiered_cache = tiered_cache

# __main__.py — 생성자 인자로 전달, 직접 주입 코드 제거:
scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer,
                      kv_cache_manager=kv_cache_manager, tiered_cache=tiered_cache)
```

**파일:** `mlx_lm_server/scheduler.py`, `mlx_lm_server/__main__.py`

---

### E2: SSD 모델 fingerprint 분리 (확장)

**문제:** 모델이 바뀌어도 같은 SSD 디렉토리 → KV 충돌.

**Codex 보완:** model_name + shape만으로 충돌 가능 → kv_bits, kv_group_size 포함.

**수정 (kv_cache_manager.py 또는 config.py):**
```python
def compute_model_fingerprint(model_name: str, model, kv_bits: int, kv_group_size: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(model_name.encode("utf-8"))
    h.update(struct.pack("<ii", kv_bits, kv_group_size))
    if hasattr(model, "config"):
        cfg = model.config
        h.update(struct.pack("<iii",
            getattr(cfg, "num_hidden_layers", 0),
            getattr(cfg, "num_key_value_heads", 0),
            getattr(cfg, "hidden_size", 0),
        ))
    return h.hexdigest()
```

**통합 (__main__.py):**
```python
fingerprint = compute_model_fingerprint(config.model, model, config.kv_bits, config.kv_group_size)
ssd_dir = config.ssd_cache_dir / fingerprint
```

**파일:** `mlx_lm_server/kv_cache_manager.py`, `mlx_lm_server/__main__.py`

---

### E3: SSD index 메타데이터 + hash_version (D1 전제)

**문제 (Codex D1):** 해시 알고리즘 변경 시 SSD 인덱스가 광범위하게 깨짐. 버전 관리 필요.

**수정 (ssd_cache.py):**
```python
CURRENT_HASH_VERSION = 1  # 체인 해시 전환 시 2로 변경

# index.json 새 구조:
{
    "__metadata__": {
        "index_version": 1,
        "hash_version": 1,
        "model_fingerprint": "<hex>",
        "created_at": "..."
    },
    "blocks": { ... }
}

# load_index(): metadata 없거나 hash_version 불일치 → 인덱스 무효화, 빈 dict 반환
# save_index(): __metadata__ 포함하여 저장
```

**파일:** `mlx_lm_server/ssd_cache.py`

---

## Phase F: 성능 최적화 (MEDIUM)

### F1: 해시 계산 O(n²) → 체인 해시 O(n)

**문제:** `compute_block_hash(prefix_tokens, block_tokens)`가 매 블록마다 전체 prefix 재해시.

**Codex 2차 보완 (반영):** 시그니처 즉시 교체 대신 API 호환 레이어 유지. 기존 함수를 wrapper로 보존, 내부에 새 체인 해시 추가 후 단계적 전환.

**수정:**
```python
# 새 내부 함수 (체인 해시)
def _compute_chain_hash(block_tokens: list[int], prev_hash: str | None = None) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update((prev_hash or "").encode('ascii'))
    for tok in block_tokens:
        h.update(struct.pack("<i", tok))
    return h.hexdigest()

# 기존 함수 — 호환 wrapper로 유지 (단계적 전환)
def compute_block_hash(prefix_tokens: list[int], block_tokens: list[int]) -> str:
    """Legacy wrapper — 기존 호출부와 테스트 호환 유지."""
    if not block_tokens:
        # Codex 3~4차: 빈 block_tokens 방어 (range step 0 방지)
        # prefix 무관 deterministic sentinel 반환 — 정상 운영에서 미발생
        # (legacy hash(prefix+[])는 prefix마다 다른 빈 해시 → 해시 테이블 낭비)
        logger.warning("compute_block_hash called with empty block_tokens — returning sentinel hash")
        return _compute_chain_hash([], None)
    prev_hash = None
    block_size = len(block_tokens)
    for i in range(0, len(prefix_tokens), block_size):
        chunk = prefix_tokens[i:i+block_size]
        if len(chunk) == block_size:
            prev_hash = _compute_chain_hash(chunk, prev_hash)
    # 참고: prefix가 block_size 배수가 아닌 경우, trailing 부분은 무시됨
    # (기존 동작과 동일 — prefix는 항상 full block 단위로 전달되는 것이 전제)
    return _compute_chain_hash(block_tokens, prev_hash)
```

**호출부 전환:** find_cached_prefix, allocate_blocks 등 내부 루프는 `_compute_chain_hash` 직접 사용 (prev_hash 전달). 외부 API(`cache_block` 등)는 기존 시그니처 유지.

**SSD 캐시 무효화:** E3의 `hash_version`을 2로 올려 자동 invalidation.

**파일:** `mlx_lm_server/kv_cache_manager.py` + 관련 테스트 (기존 테스트는 wrapper 덕분에 변경 최소화)
**별도 커밋 권장.**

---

### F2: SSD index batch flush

**문제:** `save_block()`, `load_block()` 호출마다 `save_index()` → I/O 병목.

**수정 (ssd_cache.py):**
```python
self._index_dirty = False
self._mutation_count = 0
self._flush_interval = 10

def _mark_dirty(self):
    self._mutation_count += 1
    if self._mutation_count >= self._flush_interval:
        self.save_index()
        self._mutation_count = 0
        self._index_dirty = False
    else:
        self._index_dirty = True

def flush(self):
    if self._index_dirty:
        self.save_index()
        self._mutation_count = 0
        self._index_dirty = False
```

`save_block()`과 `load_block()`에서 `save_index()` → `_mark_dirty()`로 교체.
`prune_expired()`와 에러 경로는 즉시 `save_index()` 유지.
`Scheduler.stop()`에서 `ssd.flush()` 호출 추가.

**파일:** `mlx_lm_server/ssd_cache.py`, `mlx_lm_server/scheduler.py`

---

### F3: Sequence cache `best_key_len > len(tokens)` 분기 — Dead Code 확정

**1차 리서치(cache-trimmer):** "도달 가능"으로 판단했으나, **2차 리서치에서 반박 확인.**

**Codex 2차 + 검증 결과:** trie의 `store()`는 `cache_value`를 **leaf 노드에만** 설정 (`node.cache_value = cache_copy`, line 161). 중간 노드에는 `cache_value`가 없음. `find_longest_prefix()`는 `node.cache_value is not None`일 때만 `best_depth`를 갱신하므로, `best_key_len`은 `len(tokens)` 이하만 가능. cached seq [A,B,C,D]에 대해 query [A,B]로 탐색 시, 노드 B에는 cache_value가 없고(leaf가 아님), D에만 있으므로 hit 불가.

**결론: `best_key_len > len(tokens)` 분기는 dead code.**

**조치:**
- 해당 분기에 `# UNREACHABLE: trie stores cache_value only on leaf nodes` 주석 추가
- 또는 방어적 코드로 유지하되 dead code임을 명시
- `trim_prompt_cache` 호출은 실행되지 않으므로 성능 영향 없음

---

## Phase G: SSD 기능 확장 (LOW-MEDIUM)

### G1: SSD → RAM promote 읽기 경로 추가

**문제 (GPT2-5):** SSD는 eviction 시 저장만 하고, 이후 lookup에서 읽어오는 경로가 없음. `TieredKVCache.lookup()`은 구현되어 있으나 어디에서도 호출되지 않음. 사실상 write-only.

**Codex 2차 보완 (반영):** `ssd.load_block()`은 `list[dict]` 또는 legacy `dict`를 반환할 수 있음. RAM block에 대입하거나 `reconstruct_cache_from_blocks()`에 넘기기 전에 형식 정규화 필수. 또한 `self._tiered_cache.ssd` 존재 체크 필요.

**수정 방향 (scheduler.py block reconstruction 루프 내):**
```python
# block_data 수집 시 kv_data가 None이면 SSD promote 시도
if block.kv_data is None and block.block_hash and self._tiered_cache:
    if not hasattr(self._tiered_cache, 'ssd') or self._tiered_cache.ssd is None:
        all_blocks_valid = False
        break
    raw_data = self._tiered_cache.ssd.load_block(block.block_hash)
    # 형식 정규화: list[dict] 또는 dict → 통일된 kv_data 형태
    kv_data = _normalize_ssd_block_data(raw_data) if raw_data is not None else None
    if kv_data is not None:
        block.kv_data = kv_data
        block.last_accessed = time.time()
        block_data.append(kv_data)
    else:
        all_blocks_valid = False
        break
```

D1의 block validation 루프에 자연스럽게 통합.

**더 완전한 SSD 활용:** `find_cached_prefix()`에서 hash_table miss 시 SSD index도 확인하려면 `allocate_blocks()`에서 SSD 블록을 RAM으로 promote하는 로직이 필요. 이는 구현 복잡도가 높아 **별도 PR**로 분리 권장.

**파일:** `mlx_lm_server/scheduler.py`

---

### G2: Prefill 시점 캐시 조기 저장

**문제 (GPT1-10):** KV 블록을 요청 완료 후에만 저장 → 긴 생성 중 동일 prefix 요청이 캐시 혜택 못 받음.

**현실적 제약:** BatchGenerator 내부 `_caches` 접근이 명시 API로 제공되지 않아 private 구조 의존 시 유지보수 리스크. GPT2, Codex 모두 마지막 Phase로 동의.

**수정 방향:**
- `_pending_cache_saves: set[int]` — cache miss로 삽입된 UID 추적
- 첫 decode step 후 `_save_prefill_caches()` 호출
- BatchGenerator._caches 접근 불가 시: 시퀀스의 allocated block_ids 중 kv_data가 채워진 블록만 cache_block()으로 저장

**파일:** `mlx_lm_server/scheduler.py`
**구현 시 BatchGenerator 내부 구조 재확인 필요.**

---

## 최종 구현 순서

```
Phase A (CRITICAL — 정확도)
  A1 (encode fallback)
    ↓
  A2 (cache trim on store) + A3 (full hit trim by 1)
    ↓
Phase B (HIGH — 리소스)
  B1 (get_result TimeoutError) + B2 (stream leak) + B3 (stop() join)
    ↓
Phase D (MEDIUM — 방어)
  D1 (block data validation)
    ↓
Phase E (MEDIUM — 구조)
  E1 (tiered_cache constructor)
    ↓
Phase C (HIGH — API 호환)
  C1 (streaming stop buffer) + C2 (EOS token_id) + C3 (first_token_timeout)
    ↓
Phase D cont.
  D2 (deepcopy gradual clone)
    ↓
Phase E cont.
  E2 (model fingerprint) + E3 (SSD index metadata)
    ↓
Phase F (MEDIUM — 성능)
  F1 (chain hash) + F2 (SSD batch flush)
    ↓
Phase G (LOW-MEDIUM — SSD 확장)
  G1 (SSD promote) → G2 (prefill save)
```

**원칙:** 정확도 버그를 먼저, 리소스 안전성 다음, API 호환 + 방어, 최적화 마지막.
**각 Phase 완료 시:** `pytest tests/ -v --tb=short` 전체 통과 확인 후 다음 Phase 진행.

---

## 참고: 원본 리뷰 소스

| 소스 | 항목 수 | 반영 |
|------|---------|------|
| GPT PRO 1차 감사 | 10건 | A1~A3, C1, D2, E1~E3, F1~F2, G2 |
| GPT PRO 2차 제안 | 8건 | B1~B3, C2, C3, D1, G1, F3(확인만) |
| Codex 코멘트 | 9건 | A1~A3 보완, C1 위치 변경, D2 단계적, E2 확장, E3 메타데이터, F1 버전관리 |
| Researcher 코드 분석 | 8명 | 전 항목 코드 레벨 검증 완료 |

---

## Codex 2차 검증 코멘트 — 판정 결과

| # | Codex 제안 | 판정 | 근거 |
|---|-----------|------|------|
| 1 | A1 chat 경로만 적용 | **동의 → 반영** | completions 경로는 BOS가 정당할 수 있음 |
| 2 | A1 tokenize=True 우선 | **반대 → 기각** | SimpleTokenizer가 tokenize=True 무시 (항상 str 반환). Union[str,list[int]] 분기 + 인터페이스 변경 필요 → _safe_encode가 더 실용적 |
| 3 | B1 buffer 정리 경로 | **조건부 동의 → 반영** | 리서치 결과 cleanup 경로 완전 확인. TimeoutError 시 get_result()는 pop 안 함 → cancel_request()의 _cleanup_result_buffers()가 idempotent하게 정리. 명시적 문서화 추가 |
| 4 | D2 QuantizedKVCache 제외 | **동의 → 반영** | tuple 상태라 keys/values slice 불가. plain KVCache만 fast path |
| 5 | F3 도달 불가 | **동의 → 반영** | trie의 cache_value는 leaf 노드에만 설정. best_key_len > len(tokens) 불가. Dead code 확정 |
| 6 | G1 데이터 정규화 | **동의 → 반영** | load_block() 반환 형태 다양. 정규화 + ssd 존재 체크 추가 |
| 7 | F1 호환 wrapper | **동의 → 반영** | 기존 compute_block_hash를 wrapper로 유지, 내부 _compute_chain_hash 추가 |

---

## Codex 3차 코멘트 — 판정 결과

| # | Codex 제안 | 판정 | 반영 내용 |
|---|-----------|------|----------|
| 1 | B1 "cleanup 완전" 과대판정 | **동의 → 수정** | active request 경로에서 `_cleanup_result_buffers()` 미호출 확인. `_process_cancellations_batch()`에 cleanup 추가 |
| 2 | A1 tokenize=True 선택적 시도 | **동의 → 수정** | `tokenize=True` 우선 시도 → `list[int]` 반환 시 채택, `str` 반환 시 `_safe_encode` fallback |
| 3 | F1 wrapper edge case 가드 | **동의 → 추가** | `block_tokens=[]` 방어 + trailing prefix 처리 주석 추가 |
| 4 | A2 offset → output_tokens 길이 | **동의 → 수정** | `num_to_trim = len(seq.output_tokens)` 로 캐시 타입 독립적 trim |

---

## Codex 최종 추천 사항 — 판정 결과

| # | Codex 제안 | 판정 | 반영 내용 |
|---|-----------|------|----------|
| 1 | A2/A3 non-trimmable 캐시 정책 고정 | **동의 → 반영** | A2: trim 불가 시 저장 스킵 (warning 로깅). A3: trim 불가 시 uncached 경로 fallback (블록 해제 + 전체 재계산) |
| 2 | F1 empty block legacy 의미 규칙 | **동의 → 반영** | prefix 무관 deterministic sentinel 유지 (legacy hash(prefix+[])보다 낫음). `logger.warning` 추가. 정상 운영에서 미발생 문서화 |
| 3 | B1 cancel 후 get_result() 계약 문서화 | **동의 → 반영** | docstring에 API contract 추가: timeout→TimeoutError→cancel 필수, cancel 후 재호출 금지 |

**합의 완료 — 구현 단계 진행 가능.**

---

## Codex 구현 포인트 (Claude 실구현 보조)

아래는 합의된 플랜을 코드로 옮길 때의 핵심 포인트와 최소 스니펫입니다.

### Phase A

#### A1 (`mlx_lm_server/server.py`)
- chat 경로에만 적용
- `apply_chat_template(tokenize=True)`가 `list[int]`를 줄 때만 채택
- 아니면 `_safe_encode()` fallback

```python
def _safe_encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)

templ = _format_chat_messages(body.messages, tok, tokenize=True)
prompt_tokens = templ if isinstance(templ, list) else _safe_encode(tok, templ)
```

#### A2 (`mlx_lm_server/scheduler.py:_store_finished_caches`)
- trim 길이: `len(seq.output_tokens)`
- 합의안: non-trimmable이면 sequence cache 저장 스킵

```python
can_store = True
num_generated = len(seq.output_tokens)
if num_generated > 0 and prompt_cache is not None:
    if can_trim_prompt_cache(prompt_cache):
        trim_prompt_cache(prompt_cache, num_generated)
    else:
        logger.warning("Skip sequence-cache store: non-trimmable cache")
        can_store = False
if can_store:
    self._sequence_cache.store(prompt_tokens, prompt_cache)
```

#### A3 (`mlx_lm_server/scheduler.py:_insert_new_requests_batch`)
- full hit에서 `remaining_tokens == []` 처리
- 합의안: trim 불가면 uncached fallback

```python
if not remaining_tokens:
    remaining_tokens = [seq.token_ids[-1]]
    if cache is not None:
        if can_trim_prompt_cache(cache):
            trim_prompt_cache(cache, 1)
        else:
            if self.kv_cache_manager is not None and seq.block_ids:
                self.kv_cache_manager.free_blocks(seq.block_ids)
            seq.block_ids = []
            cache = None
            remaining_tokens = seq.token_ids
```

### Phase B

#### B1
- `get_result()` timeout은 `TimeoutError`
- 서버는 catch 후 `cancel_request()`
- active cancel 누수 방지: `_process_cancellations_batch()`에서 `_cleanup_result_buffers(rid)` 호출

```python
if not event.wait(timeout=timeout):
    raise TimeoutError(...)

# cancellation batch
self._signal_finish(rid, finish_reason="cancelled")
self._cleanup_result_buffers(rid)
```

#### B2
- `cancel_request()` not-found 경로에서 orphan stream 정리

```python
with self._streams_lock:
    self._streams.pop(request_id, None)
```

#### B3
- `stop()` join 후 `is_alive()`면 warning + thread handle 유지

### Phase C

#### C1 (`_stream_response`)
- stop 없으면 즉시 전송(기존 동작 유지)
- stop 있으면 `max_stop_len - 1` 버퍼링
- stop 감지 시 stop 앞 텍스트만 전송

#### C2
- EOS는 `token_text`가 아니라 `token_id` 기준 필터

```python
if eos_ids and event.token_id in eos_ids:
    token_text = ""
```

#### C3
- `first_token_timeout_s` 추가
- 첫 토큰만 긴 timeout, 이후는 기존 timeout

### Phase D

#### D1
- block reconstruct 시 `len(block_data) != len(block_ids)`면 partial miss
- 즉시 rollback (`free_blocks`) 후 uncached 경로

#### D2
- fast clone은 plain `KVCache`만
- `QuantizedKVCache`, `CacheList`는 deepcopy fallback

### Phase E

#### E1
- `Scheduler.__init__(..., tiered_cache=None)` 정식 주입
- `__main__.py`의 private attr 주입 제거

#### E2
- fingerprint: model + kv_bits + kv_group_size + model shape
- SSD 경로 분리: `ssd_cache_dir / fingerprint`

#### E3
- SSD index metadata: `index_version`, `hash_version`, `model_fingerprint`
- mismatch 시 index invalidate

### Phase F

#### F1 (`mlx_lm_server/kv_cache_manager.py`)
- 기존 `compute_block_hash(prefix, block)`는 wrapper 유지
- 내부 루프는 체인 해시 사용
- empty block 규칙은 합의안대로 고정

#### F2 (`mlx_lm_server/ssd_cache.py`)
- `_mark_dirty()` / `flush()` 도입
- shutdown 시 flush 보장

#### F3
- dead branch는 주석으로 의도 명시

### Phase G

#### G1
- `ssd.load_block()` 결과 정규화 helper 필수 (`list[dict]` 통일)
- promote 성공 시 `block.kv_data`와 `last_accessed` 갱신

#### G2
- 마지막 단계에서 진행
- private 내부 구조 의존 최소화

---

## 구현 체크 순서 (권장)

1. A1 → A2/A3
2. B1/B2/B3
3. D1
4. C1/C2/C3
5. D2
6. E1/E2/E3
7. F1/F2/F3
8. G1/G2

각 단계 끝에서 관련 테스트 subset 먼저 실행 후 전체 `pytest` 실행.
