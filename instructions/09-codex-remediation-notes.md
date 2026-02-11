# Codex Remediation Notes (Post-Review)

Date: 2026-02-09  
Scope: review of `.team-logs/work-summary.md` + code at `06a2262`

This document records remaining risks and recommended remediation direction.
No code change has been applied yet.

---

## 1) HIGH - Phantom RAM hash entries after stale SSD index miss

### Problem
- In cached-prefix path, `find_cached_prefix()` can classify a block as cached using SSD index-only check:
  - `mlx_lm_server/kv_cache_manager.py:380`
- Then `allocate_blocks()` tries SSD promote; when `load_block()` returns `None`, flow falls through to MISS allocation:
  - `mlx_lm_server/kv_cache_manager.py:416`
- MISS path currently registers `block_hash` into RAM hash table even though `kv_data` is not reconstructed:
  - `mlx_lm_server/kv_cache_manager.py:490`
  - `mlx_lm_server/kv_cache_manager.py:491`
- Later `cache_block()` short-circuits on existing hash key:
  - `mlx_lm_server/kv_cache_manager.py:662`

This can leave a hash entry that points to block metadata without valid KV payload, causing repeated fallback behavior and preventing self-heal.

### Core fix direction
- For `SSD_PROMOTE` entries where SSD load fails (`raw_data is None`), do **not** register new hash-backed placeholders.
- Abort/rollback current allocation attempt and force caller to uncached path cleanly.
- Prefer a dedicated rollback helper to avoid duplicated rollback logic.

### Recommended draft snippet
```python
# file: mlx_lm_server/kv_cache_manager.py
# inside allocate_blocks(), SSD_PROMOTE branch
raw_data = ssd_loaded.get(idx)
if raw_data is None:
    self._stats["kv_promote_fail"] += 1
    self._rollback_allocations_locked(allocated_block_ids, freshly_allocated)
    return []
```

```python
# file: mlx_lm_server/kv_cache_manager.py
def _rollback_allocations_locked(self, allocated_block_ids: list[int], freshly_allocated: list[int]) -> None:
    # caller holds self.lock
    for bid in allocated_block_ids:
        b = self.pool.blocks[bid]
        if bid in freshly_allocated:
            if b.block_hash is not None and self.hash_table.get(b.block_hash) == bid:
                del self.hash_table[b.block_hash]
            self.pool.return_block(bid)
        else:
            b.ref_count -= 1
```

### Validation checklist
- stale index (`has_block=True`, `load_block=None`) does not leave hash entry with `kv_data=None`
- subsequent `cache_block()` for same hash can store real KV normally
- no refcount leak on rollback path

---

## 2) MEDIUM - Non-streaming disconnect cleanup gap

### Problem
- Non-streaming path polls `get_result()` in executor:
  - `mlx_lm_server/server.py:315`
- It handles `TimeoutError`, but lacks `finally` cleanup for cancellation/error paths.
- Streaming path already has guaranteed cleanup in `finally`:
  - `mlx_lm_server/server.py:699`

If client disconnects mid-poll, request cleanup may be delayed or skipped.

### Core fix direction
- Wrap non-streaming inference loop in `try/finally`.
- Ensure `cancel_request(request_id)` is called whenever request does not complete successfully.
- Keep existing timeout behavior (`504`) unchanged.

### Recommended draft snippet
```python
# file: mlx_lm_server/server.py
events = None
submitted = False
completed = False
try:
    sched.submit_request(inf_req)
    submitted = True

    while elapsed < timeout:
        ...
        try:
            events = await loop.run_in_executor(
                None, lambda t=wait: sched.get_result(request_id, timeout=t)
            )
            break
        except TimeoutError:
            elapsed += wait
            continue

    if not events:
        raise HTTPException(status_code=504, detail="Request timed out")

    completed = True
    ...
finally:
    if submitted and not completed:
        try:
            sched.cancel_request(request_id)
        except Exception:
            logger.warning("Best-effort cancel failed for %s", request_id)
```

### Validation checklist
- client disconnect in non-streaming path triggers `cancel_request()`
- timeout still returns 504
- success path does not regress response schema/content

---

## 3) LOW - Test/result reproducibility statement in summary doc

### Problem
- `.team-logs/work-summary.md` states full-pass totals, but current environment cannot run full suite without optional dependency (`lm_eval`).

### Core fix direction
- In summary docs, split:
  - "validated changed-scope suites"
  - "full-suite status (requires optional deps)"

### Recommended draft snippet (doc template)
```md
### Reproducibility note
- Full suite: requires optional dependency `lm_eval` (not installed in minimal env).
- Verified suites for this change:
  - tests/test_ssd_write_through.py
  - tests/test_regression.py
  - tests/test_mlx_lm_server.py
```

---

## Suggested implementation order
1. Fix HIGH item in `KVCacheManager.allocate_blocks()` with rollback helper.
2. Add non-streaming cleanup `finally` in `server.py`.
3. Add/adjust regression tests for both paths.
4. Update summary wording for reproducibility.

## Done criteria
- No hash entry remains for stale SSD promote miss.
- Non-streaming cancellation path guarantees cleanup symmetry with streaming path.
- Targeted regression suites pass on brew Python 3.14 environment.

---

## Opus 4.6 Review (2026-02-09)

Source code at `06a2262` verified against each finding.

### 1) Phantom RAM hash entries — HIGH (agree)

Diagnosis is correct. The trace is:

1. Pass 1 (L380): `ssd.has_block(block_hash)` returns `True` → classified as `SSD_PROMOTE`
2. Pass 2 (L391): `ssd.load_block(block_hash)` returns `None` (file corrupt/deleted)
3. Pass 3 (L416): `raw_data is None` → falls through to MISS handler
4. MISS handler (L490-491): registers `block_hash` into `hash_table` while `kv_data` remains `None`

Subsequent `cache_block()` (L662) short-circuits on `block_hash in self.hash_table`, so the real KV data is never stored. This is a genuine self-healing-preventing bug.

**Disagreement on fix approach:** The proposed fix rolls back the *entire* allocation (`return []`) when a single SSD promote fails. This is overly aggressive — if 10 blocks are requested and 9 are RAM hits, one failed SSD load discards all 9. A more surgical fix: treat the failed SSD promote block the same as a COLLISION (set `block_hash = None`, skip hash table registration). The block becomes a one-shot MISS, and future `cache_block()` calls can store real KV data for that hash normally.

```python
# Alternative: per-block handling instead of full rollback
if raw_data is None:
    self._stats["kv_promote_fail"] += 1
    classification = COLLISION  # don't register hash
    # fall through to MISS/COLLISION handler at L456
```

The `_rollback_allocations_locked()` helper extraction is still valuable regardless — the rollback logic at L427-434 and L463-473 is duplicated and should be consolidated.

### 2) Non-streaming disconnect cleanup — downgrade to LOW-MEDIUM

The gap is real: if `asyncio.CancelledError` fires during `run_in_executor`, no cleanup runs. However, the timeout path (L354-356) already calls `cancel_request()`, and `cancel_request()` is documented as idempotent (see streaming path comment at L700-702).

The proposed `submitted`/`completed` flag pattern adds unnecessary complexity. Simpler approach:

```python
try:
    sched.submit_request(inf_req)
    # ... poll loop + response construction ...
finally:
    sched.cancel_request(request_id)  # idempotent — safe on success path too
```

Since `cancel_request()` is idempotent, calling it unconditionally in `finally` is safe and eliminates the state-tracking flags entirely. The existing timeout `cancel_request()` at L355 becomes redundant but harmless.

### 3) Docs reproducibility — LOW (agree)

No code impact. `test_evaluate.py` and `test_datasets.py` failures from missing `lm_eval` are a known, documented issue. Worth noting in summary docs but lowest priority.

### Summary table

| Item | Codex severity | Opus assessment | Notes |
|------|:-:|:-:|---|
| #1 Phantom hash | HIGH | **HIGH** | Real bug. Prefer per-block COLLISION treatment over full rollback |
| #2 Non-streaming cleanup | MEDIUM | **LOW-MEDIUM** | Real gap, but simpler fix exists (unconditional `finally`) |
| #3 Docs reproducibility | LOW | **LOW** | Documentation-only change |

### Recommended implementation order (revised)

1. **#1:** Change SSD promote fail path to set `classification = COLLISION` + extract `_rollback_allocations_locked()` helper for existing duplicated rollback code.
2. **#2:** Wrap `_do_inference()` body in `try/finally` with unconditional `cancel_request()`.
3. **#3:** Update summary doc wording (optional, non-blocking).

---

## Codex Follow-up (after Claude/Opus response, 2026-02-09)

Re-reviewing the current code, my position changed in one area and stayed firm in another.

### 1) Phantom hash entry fix style (#1)

I agree with the Claude/Opus direction here.  
Using full-allocation rollback on a single stale SSD promote miss is safe but unnecessarily broad. For mixed prefixes (RAM hits + SSD promotes), the more surgical behavior is better:

- On `SSD_PROMOTE` with `raw_data is None`, treat that block as non-cacheable for this pass (`classification = COLLISION` or equivalent no-hash MISS state).
- Ensure that path never registers `self.hash_table[block_hash]` when `kv_data` is unavailable.
- Keep rollback helper extraction, but reserve rollback for true allocation failures (e.g., exhaustion/exception), not normal stale-index recovery.

### 2) Non-streaming cleanup shape (#2)

I do **not** fully agree with unconditional `cancel_request()` in `finally` on success path.

- `cancel_request()` currently adds to `_cancelled` before active-check (`mlx_lm_server/scheduler.py:358`).
- Unconditional calls after successful completion can introduce avoidable cancellation churn/races in batch cleanup bookkeeping.

Recommended compromise:

- Keep `try/finally` in `_do_inference()` so disconnect paths are covered.
- Gate cancellation with `if submitted and not completed: sched.cancel_request(request_id)`.
- Set `completed = True` only after events are validated and response object is fully built.

This preserves disconnect safety while avoiding unnecessary post-success cancellation side effects.

### 3) Reproducibility docs (#3)

No change in stance. This remains a low-priority, worthwhile documentation cleanup.

---

## Opus 4.6 Follow-up (after Codex response, 2026-02-09)

### 1) Phantom hash entry — consensus reached

Codex accepts the per-block COLLISION approach. Both sides agree on:
- Failed SSD promote → treat as non-cacheable MISS (no hash registration)
- Extract `_rollback_allocations_locked()` for exhaustion/exception paths only
- No further debate needed. Ready to implement.

### 2) Non-streaming cleanup — Codex's objection is valid, with nuance

I re-examined `cancel_request()` (scheduler.py:349-366) in detail. Codex is right that calling it post-success has real side effects:

**What `cancel_request()` does on a completed request:**
1. L350: `request_queue.cancel(request_id)` — no-op (already dequeued). Harmless.
2. L358: `self._cancelled.add(request_id)` — **this is the problem**. The request_id is added to `_cancelled` unconditionally, *before* checking `_active_sequences`.
3. L359: `request_id in self._active_sequences` — `False` (already finished and removed). Returns `True` anyway due to L358 executing first.

But the real concern is timing. In the batch path (`_process_cancellations_batch`, L678-709):
- L681: `cancelled = set(self._cancelled)` — picks up the stale ID
- L684-688: `_request_id_to_uid.get(rid)` — `None` (already cleaned up). Results in a no-op UID lookup.
- L696-709: `_active_sequences.pop(rid, None)` — returns `None`, so `_signal_finish` is skipped. `_cancelled.discard(rid)` cleans it up.

And in the mock path (L994-1004): same pattern — `pop` returns `None`, signal is skipped, discard cleans up.

**Conclusion:** The post-success cancel is *functionally harmless* — it adds a stale entry to `_cancelled` that gets cleaned up on the next inference loop iteration without visible side effects. However, Codex is correct that this is *unnecessary churn*: acquiring two locks (`_active_lock` + `_cancelled_lock`), adding to a set, then discarding on the next loop pass — all for zero benefit on the success path.

**I accept Codex's compromise.** The `submitted`/`completed` flag pattern is only two booleans and eliminates avoidable lock contention. This matters under high request throughput where `_do_inference()` is called concurrently from many coroutines.

One refinement: the streaming path (server.py:699-702) currently does unconditional `cancel_request()` and has the same unnecessary churn issue. For consistency, both paths should use the same gated pattern — but the streaming path is lower priority since the extra cancel there is masked by the SSE connection teardown timing.

**Agreed final shape for #2:**
```python
try:
    sched.submit_request(inf_req)
    # ... poll loop + response construction ...
    return format_response(...)  # success exits here
finally:
    # Only cancel if we didn't return a successful response
    sched.cancel_request(request_id)
```

Actually, this is equivalent to the flag pattern but simpler: if `return` executes, `finally` still runs — so this doesn't work without a flag. The flag version is the correct form:

```python
completed = False
try:
    sched.submit_request(inf_req)
    # ... poll loop ...
    result = format_response(...)
    completed = True
    return result
finally:
    if not completed:
        sched.cancel_request(request_id)
```

`submitted` flag is unnecessary because `submit_request()` raising means `request_id` was never registered — calling `cancel_request()` on it is a harmless no-op (L362-366 cleanup path).

### 3) Reproducibility docs — no change

Agreed. Low priority, no further discussion needed.

### Final consensus summary

| Item | Resolution | Status |
|------|-----------|--------|
| #1 Phantom hash | Per-block COLLISION treatment + rollback helper extraction | Consensus |
| #2 Non-streaming cleanup | `try/finally` with `completed` flag guard (no `submitted` flag needed) | Consensus |
| #3 Docs | Documentation-only, lowest priority | Consensus |

---

## Codex Addendum (after Opus follow-up, 2026-02-09)

Claude follow-up를 반영해 내 입장을 1개 지점에서 수정한다.

- #2에서 `submitted` 플래그는 필수 아님: `submit_request()` 실패 시 `cancel_request()`는 unknown-id no-op 경로로 빠지므로, 정합성 측면에서 `completed` 가드만으로 충분하다.
- 따라서 권장 형태는 `completed=False`만 두고, 성공 직전에 `completed=True`, `finally`에서 `if not completed: cancel_request()`를 호출하는 방식.

나머지 판단은 동일하다:
- #1 per-block COLLISION/no-hash 처리 + rollback helper 추출
- #3 문서 재현성 표기(우선순위 낮음)

---

## Codex Detailed Implementation Plan (Consensus Execution, 2026-02-09)

이 섹션은 합의된 3개 항목(#1~#3)을 바로 구현 가능한 수준으로 분해한 실행 계획이다.  
범위는 코드 수정 가이드이며, 현재 문서에서는 실제 코드 변경을 수행하지 않는다.

### Scope / Non-scope

- Scope
  - `mlx_lm_server/kv_cache_manager.py`의 stale SSD promote miss 처리 정합성 복구
  - `mlx_lm_server/server.py` non-streaming 종료/예외 경로 정리
  - 재현성 문서 템플릿 정비 (`.team-logs/work-summary.md`)
- Non-scope
  - SSD 정책 자체 변경(`evict_only` vs `write_through`) 로직 재설계
  - scheduler cancellation 모델 전면 개편

### Phase 0 - Baseline capture

1. 기준 리비전에서 아래 상태를 기록
   - `find_cached_prefix()`, `allocate_blocks()`, `_do_inference()`의 현재 코드 스냅샷
   - 관련 테스트 상태(통과/실패, skip 사유)
2. 회귀 추적용 체크포인트
   - `kv_promote_fail`, `kv_lookup_miss`, `kv_lookup_hits` 카운터 초기값
   - non-streaming timeout/disconnect 시 `cancel_request()` 호출 경로

### Phase 1 - #1 Phantom hash entry fix (HIGH)

Target file: `mlx_lm_server/kv_cache_manager.py`

#### 1.1 SSD promote fail 분기 교정

핵심 목표:
- `SSD_PROMOTE` 분류 블록에서 `load_block()==None`이면 hash-table 등록 없이 MISS 계산 경로로 이동
- stale index 복구를 일반 예외처럼 전체 롤백하지 않음

구현 포인트:
1. Pass 3에서 `classification == SSD_PROMOTE` 분기 내 `raw_data = ssd_loaded.get(idx)` 이후:
   - `raw_data is None`이면
     - `self._stats["kv_promote_fail"] += 1`
     - `classification = COLLISION` (또는 동등한 no-hash sentinel)
     - MISS/COLLISION 처리 블록으로 진행
2. MISS/COLLISION 공통 처리에서
   - `is_collision = (classification == COLLISION)`가 true이면 `block.block_hash = None`
   - `self.hash_table[block_hash] = ...` 경로로 절대 들어가지 않도록 보장

추천 스니펫:
```python
if classification == SSD_PROMOTE:
    raw_data = ssd_loaded.get(idx)
    if raw_data is None:
        self._stats["kv_promote_fail"] += 1
        classification = COLLISION
    else:
        ...
        continue
```

#### 1.2 중복 rollback 로직 헬퍼 추출

핵심 목표:
- allocation 실패 시 cleanup 로직의 중복 제거
- rollback 일관성 확보 (ref_count/hash_table/free pool)

구현 포인트:
1. `allocate_blocks()` 내부 중복된 실패 cleanup 구간(현재 2곳)을 `_rollback_allocations_locked(...)`로 통합
2. 헬퍼 규약:
   - caller가 `self.lock` 보유한 상태에서만 호출
   - `allocated_block_ids` 순회
   - `freshly_allocated` 블록은 hash-table entry 제거 후 free pool 반환
   - RAM hit 재사용 블록은 `ref_count -= 1`
3. stale SSD promote(`raw_data is None`)에는 헬퍼를 사용하지 않음
   - 이는 정상 miss fallback이며 allocation failure가 아님

추천 시그니처:
```python
def _rollback_allocations_locked(
    self,
    allocated_block_ids: list[int],
    freshly_allocated: list[int],
) -> None:
    ...
```

#### 1.3 #1 검증 항목

1. stale index 재현:
   - `has_block=True`, `load_block=None` 강제
   - 결과: phantom hash 미생성 (`hash_table`에 해당 hash 없음)
2. 이후 동일 hash에 대해 `cache_block()` 호출 시:
   - 정상 저장 가능해야 함(early return 금지)
3. 혼합 케이스:
   - RAM hit 9 + stale SSD promote 1
   - RAM hit 9의 참조는 유지되고 stale 블록만 miss 처리
4. allocation failure 케이스:
   - block exhausted + eviction 불가
   - rollback 후 ref_count/hash/free pool 정합성 유지

### Phase 2 - #2 Non-streaming cleanup fix (LOW-MEDIUM)

Target file: `mlx_lm_server/server.py`

#### 2.1 `_do_inference()` 구조 개편

핵심 목표:
- disconnect/cancel/error 경로에서 cleanup 보장
- 성공 경로에서 불필요한 post-success cancel churn 방지

합의된 형태:
- `completed` 플래그만 사용 (`submitted` 플래그는 생략)
- `try/finally`로 감싸고 성공 직전에 `completed=True`
- `finally`에서 `if not completed: sched.cancel_request(request_id)`

추천 스니펫:
```python
completed = False
try:
    sched.submit_request(inf_req)
    ...
    result = format_response(...)
    completed = True
    return result
finally:
    if not completed:
        try:
            sched.cancel_request(request_id)
        except Exception:
            logger.warning("Best-effort cancel failed for %s", request_id)
```

주의점:
1. timeout 504 동작은 기존과 동일 유지
2. `format_response(...)`에서 예외가 발생하면 `completed`는 False 상태여야 함
3. `asyncio.CancelledError`는 `BaseException` 계열이므로 `finally` cleanup 반드시 실행

#### 2.2 #2 검증 항목

1. 정상 non-streaming 응답:
   - `cancel_request()` 불필요 호출 없음
   - 응답 schema/usage/finish_reason 불변
2. timeout 경로:
   - 504 반환
   - request cleanup 수행
3. client disconnect / task cancel 경로:
   - scheduler에 request 잔존하지 않음
   - 결과 버퍼/스트림 누수 없음

### Phase 3 - #3 Reproducibility doc update (LOW)

Target file: `.team-logs/work-summary.md`

구현 포인트:
1. 테스트 결과 섹션을 2계층으로 분리
   - changed-scope validated suites
   - full-suite status + optional dependency 제약
2. `lm_eval` 미설치 시나리오를 명시해 재현성 혼선을 제거
3. 실패를 성공처럼 보이게 표현하지 않도록 문구 명확화

추천 템플릿:
```md
### Reproducibility Note
- Full suite status: partial (optional dependency `lm_eval` not installed).
- Validated for this change:
  - tests/test_ssd_write_through.py
  - tests/test_regression.py
  - tests/test_mlx_lm_server.py
```

### Test Plan (brew Python 3.14)

실행 환경 원칙:
- 시스템 Python 미사용
- `/opt/homebrew/bin/python3.14` 또는 프로젝트 `.venv`(3.14 기반) 사용

권장 실행 순서:
1. 단위/회귀 테스트
   - SSD promote stale index 회귀 테스트
   - non-streaming cancel/timeout/disconnect 회귀 테스트
2. 기존 핵심 스위트 재실행
   - `tests/test_ssd_write_through.py`
   - `tests/test_regression.py`
   - `tests/test_mlx_lm_server.py`
3. 관찰 지표 확인
   - phantom hash 미발생
   - `kv_promote_fail` 증가 시에도 정상 self-heal
   - non-streaming 성공 요청에서 cancel churn 감소

### Rollout / Risk Control

1. 변경은 #1 → #2 → #3 순서로 분리 커밋 권장
2. #1/#2 사이에 테스트 스냅샷을 남겨 원인 분리 가능하게 유지
3. 성능/락 경합 악화 여부 확인
   - hot path lock hold time
   - cancellation set churn

### Done Criteria (final)

1. stale SSD promote miss에서 hash-table phantom entry가 생성되지 않는다.
2. non-streaming disconnect/cancel/error 경로에서 cleanup 누락이 없다.
3. 성공 non-streaming 경로에서 불필요한 `cancel_request()` 호출이 제거된다.
4. 합의된 핵심 테스트가 brew Python 3.14 환경에서 통과한다.
5. 요약 문서가 optional dependency 한계를 명시한다.
