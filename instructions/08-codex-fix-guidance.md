# SSD Write-Through 통합 수정 지침 (최종 재정리)

기준 코드: `d263fd8`  
목표: 이번 턴에서 SSD write-through 경로의 남은 리스크를 한 번에 닫을 수 있도록, 문제점과 구현안을 파일/함수 단위로 정리.

---

## 1) 점검 범위

아래 경로 전체를 재점검함.

1. 설정/CLI: `mlx_lm_server/config.py`, `mlx_lm_server/server.py`
2. 런타임 배선: `mlx_lm_server/__main__.py`
3. 요청 처리: `mlx_lm_server/scheduler.py`
4. RAM/SSD tier: `mlx_lm_server/kv_cache_manager.py`
5. SSD 저장소/쓰기 스레드: `mlx_lm_server/ssd_cache.py`, `mlx_lm_server/ssd_writer.py`
6. 테스트 커버리지: `tests/test_ssd_write_through.py`, `tests/test_server_parse_args.py`, `tests/test_adversarial.py`, 회귀 일부

실행 확인:
```bash
.venv/bin/python -m pytest -q tests/test_ssd_write_through.py tests/test_server_parse_args.py tests/test_regression.py::TestF08_DeadlockTwoPhaseEviction
```
결과: `64 passed`.

---

## 2) 이미 해결된 항목 (재수정 불필요)

1. sync 경로 durability 미적용:
   - `TieredKVCache._save_to_ssd_with_durability()` 도입으로 해결됨.
2. `ssd_flush_interval_s` dead config:
   - `SSDCache.__init__(flush_interval_s=...)` 배선됨.
3. Python 3.8 argparse 호환:
   - `BooleanOptionalAction` 존재 여부 분기 처리됨.
4. `Scheduler` writer private 주입:
   - 생성자 인자로 주입하도록 정리됨.

---

## 3) 남은 이슈와 구현안

## P0-1. writer 종료 실패 시 non-daemon thread가 프로세스 종료를 막을 수 있음

문제:
1. `SSDWriterThread.stop()`가 sentinel enqueue 실패 또는 timeout으로 `False` 반환 가능.
2. 이때 worker thread는 non-daemon 상태로 살아남을 수 있음.
3. sentinel 없이 `queue.get()` 대기 상태로 남으면 프로세스 종료가 걸릴 수 있음.

근거:
- `mlx_lm_server/ssd_writer.py:170`~`mlx_lm_server/ssd_writer.py:203`
- `mlx_lm_server/ssd_writer.py:70` (`daemon=False`)

구현안:
1. sentinel 의존 종료를 제거하고 `closing + queue drain` 기반 종료로 변경.
2. worker loop를 `queue.get(timeout=...)`로 polling.
3. 종료 조건: `_closing and _inflight_enqueues==0 and queue empty`.

권장 스케치:
```python
def _run(self) -> None:
    while True:
        try:
            item = self._queue.get(timeout=0.1)
        except queue.Empty:
            with self._life_lock:
                if self._closing and self._inflight_enqueues == 0 and self._queue.empty():
                    break
            continue
        block_hash, kv_data, num_tokens = item
        ...
```

`stop()`:
```python
def stop(self, drain_timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + drain_timeout
    with self._life_lock:
        self._closing = True
        while self._inflight_enqueues > 0:
            ...
    self._thread.join(timeout=max(0, deadline - time.monotonic()))
    return not self._thread.is_alive()
```

테스트 추가:
1. `stop()` timeout 경계에서 worker가 결국 종료 가능한지 검증.
2. queue full + slow save 상황에서 `stop()` 이후 thread가 zombie로 남지 않는지 검증.

---

## P0-2. eviction 경로에서 `has_block()` index-only 판단으로 저장 없이 RAM eviction 가능

문제:
1. `evict_to_ssd()`가 `ssd.has_block()`이면 저장을 스킵.
2. `has_block()`은 파일 존재 확인 없이 index membership만 확인.
3. stale index(파일 유실) 상태에서 저장 없이 RAM을 비우면 SSD 복구가 불가능.

근거:
- `mlx_lm_server/kv_cache_manager.py:936`~`mlx_lm_server/kv_cache_manager.py:940`
- `mlx_lm_server/ssd_cache.py:390` (`has_block` index-only)

구현안:
1. eviction 경로에서 `has_block()` pre-check 제거.
2. 항상 `_save_to_ssd_with_durability()` 호출.
3. 반환값 판정:
   - `saved`, `dedup`만 eviction 허용.
   - `error`, `collision`은 eviction 금지(keep in RAM).

권장 스케치:
```python
result = self._save_to_ssd_with_durability(block.block_hash, block.kv_data)
if result not in {"saved", "dedup"}:
    logger.warning("SSD persist not confirmed (%s) for block %d; skip eviction", result, block.block_id)
    continue
```

보강:
1. `collision` 반복 시 hot-loop 방지를 위해 `block.last_accessed = time.time()` 갱신해 즉시 재대상화 방지.

테스트 추가:
1. stale index(엔트리만 있고 파일 없음) 상태에서 eviction 시 재저장 후 eviction 되는지.
2. `save_block -> "collision"` 반환 시 eviction이 실제로 skip되는지.

---

## P1-1. `writer.enqueue()` bool 반환으로 dedup/closing 구분 불가

문제:
1. 현재 `False`는 `closing` 또는 `dedup`를 동시에 의미.
2. `write_through()`는 둘을 구분하지 못하고 sync fallback 호출.
3. dedup 상황에서도 불필요한 sync path를 타서 latency noise 유발 가능.

근거:
- `mlx_lm_server/ssd_writer.py:123`~`mlx_lm_server/ssd_writer.py:139`
- `mlx_lm_server/kv_cache_manager.py:890`~`mlx_lm_server/kv_cache_manager.py:894`

구현안:
1. 반환형을 enum-like 문자열로 확장: `"queued" | "dedup" | "closing"`.
2. `write_through()` 분기:
   - `"queued"`: 즉시 return
   - `"dedup"`: no-op return
   - `"closing"`: sync durability fallback 시도

권장 스케치:
```python
status = self._writer.enqueue(...)
if status == "queued":
    return
if status == "dedup":
    return
# closing
self._save_to_ssd_with_durability(...)
```

테스트 추가:
1. dedup 반환 시 sync fallback이 호출되지 않는지.
2. closing 반환 시 sync fallback이 호출되는지.

---

## P1-2. SSDCache thread-safe 선언과 public API lock 일관성 불일치

문제:
1. 클래스 주석은 public method thread-safe를 주장.
2. `num_blocks` property, `save_index`, `load_index`는 lock-less 경로가 존재.
3. 외부 호출 시 race 가능성/계약 불일치.

근거:
- `mlx_lm_server/ssd_cache.py` 전체 선언 vs 메서드 구현

구현안:
1. `_lock`을 `threading.RLock`으로 변경.
2. `save_index`, `load_index`, `num_blocks`에 lock 적용.
3. 내부 중복 lock을 피하려면 `_save_index_unlocked`, `_load_index_unlocked` 분리.

권장 스케치:
```python
self._lock = threading.RLock()

def save_index(self):
    with self._lock:
        self._save_index_unlocked()
```

테스트 추가:
1. concurrent `save_block` + `num_blocks` 읽기 race-free 확인.
2. `validate_index`/`flush`/`save_index` 혼합 호출에서 deadlock 없는지.

---

## P1-3. flush timestamp 관리 일관성 보강 필요

문제:
1. `_last_flush_time`은 `_mark_dirty`/`flush`에서만 갱신됨.
2. `save_block()`의 즉시 `save_index()` 이후 timestamp 미갱신으로 over-flush 가능.

구현안:
1. `save_index()` 내부에서 성공 시 `_last_flush_time = time.monotonic()` 업데이트.
2. 호출자별 중복 업데이트 제거.

권장 스케치:
```python
def save_index(...):
    ...
    os.replace(...)
    self._last_flush_time = time.monotonic()
```

테스트 추가:
1. 즉시 flush 직후 `_mark_dirty()`가 불필요하게 연속 flush하지 않는지.

---

## P2-1. sync durability 경로 observability 부족

문제:
1. async writer는 retry/fail counters가 풍부함.
2. sync path(`TieredKVCache`)의 retry/fail은 별도 계측이 없음.
3. `/health`에서 정책별 실패 원인 분석이 어려움.

구현안:
1. `TieredKVCache`에 sync stats 추가:
   - `tiered_sync_save_attempts`
   - `tiered_sync_retry_attempts`
   - `tiered_sync_save_fail`
   - `tiered_sync_save_collision`
2. `get_tiered_stats()` 추가 후 `Scheduler.get_cache_stats()`에서 merge.

---

## 4) 일괄 적용 순서 (한 번에 끝내기)

1. `ssd_writer.py` 종료 프로토콜 개선(P0-1)
2. `kv_cache_manager.py` eviction 저장 판정 재정의(P0-2)
3. `ssd_writer.py` enqueue 상태값 분리 + `kv_cache_manager.py` 분기 보정(P1-1)
4. `ssd_cache.py` lock 계약 정리 + flush timestamp 보강(P1-2/P1-3)
5. sync observability 카운터 추가(P2-1)
6. 테스트 추가 후 회귀 실행

---

## 5) 테스트 체크리스트 (최소)

```bash
.venv/bin/python -m pytest -q tests/test_ssd_write_through.py
.venv/bin/python -m pytest -q tests/test_server_parse_args.py
.venv/bin/python -m pytest -q tests/test_regression.py::TestF08_DeadlockTwoPhaseEviction
.venv/bin/python -m pytest -q tests/test_adversarial.py::TestDA_F2_SchedulerFreesBlocksDuringSSDSave
```

신규 테스트(추가 필요):
1. writer stop timeout 후 thread liveness 보장 테스트
2. stale index + eviction 재저장 테스트
3. collision 반환 시 eviction skip 테스트
4. enqueue dedup/closing 상태 분기 테스트
5. SSDCache public API lock 일관성 테스트

---

## 6) 완료 기준 (DoD)

1. writer stop 실패 경계에서도 프로세스 종료가 hang되지 않음
2. eviction 전 SSD persist 확인이 `saved|dedup` 기준으로 엄격화됨
3. dedup과 closing이 구분되어 write_through fallback이 의도대로 동작
4. SSDCache thread-safe 계약과 구현이 일치
5. flush timestamp가 일관되어 불필요한 연속 flush가 줄어듦
6. sync/async durability 실패가 health 지표로 추적 가능
7. 신규 테스트 + 기존 회귀 테스트 통과

