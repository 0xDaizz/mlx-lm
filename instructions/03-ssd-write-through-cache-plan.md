# SSD Write-Through Cache Plan (확정)

> Codex 6회 검토 완료. 모든 보류 사안 해소. 본 문서가 구현 기준.

---

## 목적

- SSD를 eviction 보조 저장소가 아니라 **영속 캐시**로 사용.
- RAM은 SSD의 **hot mirror (가속 계층)** 로 동작.
- 새로 생성된 KV cache는 자동으로 RAM+SSD에 누적.
- 프로세스 재시작 후에도 SSD 기반 prefix hit가 즉시 가능.
- 장기 미사용 블록(TTL 7일 기본)은 자동 정리.

---

## 용어 정의

| 용어 | 정의 |
|------|------|
| `write_through` | 새 블록 생성 시 RAM + SSD에 동시 저장하는 정책 |
| `evict_only` | 기존 방식. RAM 압박 시에만 SSD로 내보냄 |
| `best_effort` | SSD 저장 실패 시 1회 시도 후 포기, RAM 유지 |
| `persistent` | SSD 저장 실패 시 최대 N회 재시도. **fsync/ack durable 보장 아님** |
| sentinel | writer thread 종료 신호용 특수 객체 |
| inflight counter | enqueue()가 queue.put()을 완료하기 전인 호출 수를 추적하는 카운터 |

---

## 설계 원칙

- **SSD는 cold data의 source of truth, RAM hash_table은 hot path의 authority.**
  추론 중에는 RAM이 모든 lookup의 1차 소스(성능). 재시작 후에는 SSD index가 복원 기반(내구성).
- 성능 보호: SSD I/O가 decode path를 과도하게 막지 않도록 비동기 writer 제공.
- 운영 안전성: crash/restart 후에도 index/data 불일치 최소화 (validate_index).
- 점진 도입: `ssd_policy` 플래그로 기존(evict_only)과 신규(write_through) 공존.

---

## 갭 분석 — 현재 구현 vs 계획

### 이미 존재하는 것

| 기능 | 비고 |
|------|------|
| SSD save/load (safetensors) | atomic write (tmpfile + os.replace) |
| SSD index 관리 | batch flush (10건당) + 즉시 flush |
| SSD promote (load → RAM) | `allocate_blocks()` 내 inline |
| Evict-to-SSD | `TieredKVCache.evict_to_ssd()` |
| TTL pruning | 1000 step마다 실행 |
| SSD prefix lookup | `find_cached_prefix()` → `ssd.has_block()` |
| Graceful shutdown flush | `scheduler.stop()` → `ssd.flush()` |

### 새로 구현해야 하는 것

| 기능 | 핵심 변경점 |
|------|------------|
| `ssd_policy` config | config.py에 정책 플래그 추가 |
| Write-through on creation | `cache_block()` 경로에서 SSD 저장 |
| Async writer thread | inflight counter + sentinel + non-daemon |
| SSD 통계 카운터 | 모듈별 로컬 카운터 + snapshot merge |
| Crash recovery (startup) | index ↔ 파일 정합성 검증 |
| Save dedup guard | index + 파일 존재 + num_tokens 이중 검증 |
| Collision 격리 | non-cacheable set (TTL 1시간) |
| SSDCache thread-safety | threading.Lock 추가 |

---

## Config 확정

```python
ssd_enabled: bool = True
ssd_cache_dir: Path = Path.home() / ".cache" / "mlx-lm-server" / "kv-cache"
ssd_ttl_days: int = 7
ssd_policy: str = "evict_only"            # "evict_only" | "write_through"
ssd_durability: str = "best_effort"       # "best_effort" | "persistent"
ssd_async_writes: bool = True
ssd_writer_queue_size: int = 512
ssd_flush_interval_s: float = 1.0
ssd_persistent_max_retries: int = 3
```

---

## API 확정

```python
# ssd_cache.py
SSDCache.save_block(block_hash: str, kv_data: list[dict]|dict, num_tokens: int|None = None) -> str
    # Returns: "saved" | "dedup" | "collision" | "error"
SSDCache.validate_index() -> dict   # {"orphans_cleaned": N, "missing_cleaned": N}
SSDCache.get_stats() -> dict

# ssd_writer.py (Phase 4)
SSDWriterThread.__init__(ssd: SSDCache, queue_size: int, durability: str)
SSDWriterThread.enqueue(block_hash: str, kv_data, num_tokens: int|None = None) -> bool
SSDWriterThread.stop(drain_timeout: float = 5.0) -> bool
SSDWriterThread.get_stats() -> dict

# kv_cache_manager.py
KVCacheManager.cache_block(..., ssd_policy: str = "evict_only") -> int|None
KVCacheManager.get_stats() -> dict

# scheduler.py
Scheduler.get_cache_stats() -> dict  # snapshot merge from all modules
```

---

## 요청 처리 플로우

### A. Cache miss → prefill 발생
1. prefill 완료 후 block 분해
2. RAM cache 등록 (`cache_block()`)
3. SSD write-through 등록 (sync 또는 async queue)
4. 이후 동일 prefix 요청은 RAM hit, 또는 재시작 후 SSD hit 가능

### B. Cache hit (RAM)
1. RAM block 사용
2. `last_accessed` 갱신

### C. Cache hit (SSD)
1. prefix lookup 단계에서 SSD 존재 확인
2. allocate 단계에서 SSD에서 block 로드
3. RAM promote + hash_table 재등록
4. decode 전 prefill skip

---

## 핵심 구현 사양

### 1. SSDCache — thread-safe 승격

writer thread 도입으로 `save_block()`(worker)과 `load_block()`(promote, inference thread)이
동시에 실행될 수 있으므로 `threading.Lock`으로 보호.

```python
class SSDCache:
    def __init__(self, ...):
        self._lock = threading.Lock()  # RLock 아님 (재진입 경로 없음)
        self._noncacheable: dict[str, float] = {}  # collision 격리
        self._stats = {
            "ssd_save_success": 0,
            "ssd_save_fail": 0,
            "ssd_save_dedup_skip": 0,
            "ssd_stale_index_cleaned": 0,
            "ssd_hash_collision_detected": 0,
        }
        ...

    def save_block(self, block_hash: str, kv_data, num_tokens: int | None = None) -> str:
        with self._lock:
            # non-cacheable 체크 (collision 격리)
            if block_hash in self._noncacheable:
                if time.time() < self._noncacheable[block_hash]:
                    return "collision"
                else:
                    del self._noncacheable[block_hash]

            if block_hash in self._index:
                entry = self._index[block_hash]
                filepath = Path(entry["filepath"])

                if not filepath.exists():
                    # stale index → 정리 후 재저장으로 fall through
                    del self._index[block_hash]
                    self._stats["ssd_stale_index_cleaned"] += 1
                    logger.warning("Stale index entry cleaned: %s", block_hash)
                else:
                    # collision 검증
                    stored_num = entry.get("num_tokens")
                    if stored_num is not None and num_tokens is not None and stored_num != num_tokens:
                        logger.warning(
                            "HASH COLLISION: hash=%s stored=%d new=%d — non-cacheable",
                            block_hash, stored_num, num_tokens,
                        )
                        self._noncacheable[block_hash] = time.time() + 3600  # 1시간 격리
                        self._stats["ssd_hash_collision_detected"] += 1
                        return "collision"

                    # 정상 dedup → touch
                    entry["last_accessed"] = datetime.now().isoformat()
                    self._mark_dirty()
                    self._stats["ssd_save_dedup_skip"] += 1
                    return "dedup"

            # ... 실제 저장 (atomic tmpfile + os.replace)
            # index에 num_tokens 포함
            return "saved"

    def load_block(self, block_hash):
        with self._lock:
            ...

    def flush(self):
        with self._lock:
            ...

    def validate_index(self) -> dict:
        """Startup: orphan 파일 정리 + missing 파일 엔트리 제거."""
        with self._lock:
            ...

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
```

### 2. SSDWriterThread — inflight counter + sentinel 결정적 종료

**종료 프로토콜:**
1. `stop()` → `_life_lock` 안에서 `_closing=True` + in-flight enqueue 완료 대기
2. `_inflight_enqueues==0` 확인 후 sentinel 삽입 → sentinel은 반드시 큐의 마지막
3. worker가 sentinel 수신 시 break → `thread.join()`

**동시성 보호:**
- `_life_lock`: enqueue/stop 간 lifecycle 동기화
- `_pending_lock`: producer↔worker 간 `_pending` set 동기화
- `_stats_lock`: producer↔worker 간 `_stats` dict 동기화

```python
class SSDWriterThread:
    def __init__(self, ssd: SSDCache, queue_size: int = 512, durability: str = "best_effort"):
        self._ssd = ssd
        self._queue = queue.Queue(maxsize=queue_size)
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._durability = durability

        # lifecycle
        self._closing = False
        self._inflight_enqueues = 0
        self._life_lock = threading.Lock()
        self._no_inflight = threading.Condition(self._life_lock)
        self._sentinel = object()

        # stats
        self._stats_lock = threading.Lock()
        self._stats = {
            "writer_enqueue_total": 0,
            "writer_enqueue_dedup_skip": 0,
            "writer_save_success": 0,
            "writer_save_fail": 0,
            "writer_queue_full_sync_fallback": 0,
            "writer_retry_attempts": 0,
            "writer_retry_final_fail": 0,
        }

        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def _inc(self, key: str, n: int = 1):
        with self._stats_lock:
            self._stats[key] += n

    def _run(self):
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                break
            block_hash, kv_data, num_tokens = item
            try:
                self._ssd.save_block(block_hash, kv_data, num_tokens)
                self._inc("writer_save_success")
            except Exception:
                self._inc("writer_save_fail")
                logger.exception("SSD writer error: %s", block_hash)
            finally:
                with self._pending_lock:
                    self._pending.discard(block_hash)

    def enqueue(self, block_hash: str, kv_data, num_tokens: int | None = None) -> bool:
        """Returns False if writer is closing or block was deduped."""
        with self._life_lock:
            if self._closing:
                return False
            self._inflight_enqueues += 1
        try:
            with self._pending_lock:
                if block_hash in self._pending:
                    self._inc("writer_enqueue_dedup_skip")
                    return False
                self._pending.add(block_hash)
            self._inc("writer_enqueue_total")

            # Level 0: non-blocking
            try:
                self._queue.put_nowait((block_hash, kv_data, num_tokens))
                return True
            except queue.Full:
                pass
            # Level 1: short wait (50ms)
            try:
                self._queue.put((block_hash, kv_data, num_tokens), timeout=0.05)
                return True
            except queue.Full:
                pass
            # Level 2: sync fallback
            with self._pending_lock:
                self._pending.discard(block_hash)
            self._inc("writer_queue_full_sync_fallback")
            try:
                self._ssd.save_block(block_hash, kv_data, num_tokens)
                self._inc("writer_save_success")
            except Exception:
                self._inc("writer_save_fail")
            return True
        finally:
            with self._life_lock:
                self._inflight_enqueues -= 1
                if self._closing and self._inflight_enqueues == 0:
                    self._no_inflight.notify_all()

    def stop(self, drain_timeout: float = 5.0) -> bool:
        """Graceful shutdown. Race-free: waits for in-flight enqueues, then sentinel."""
        with self._life_lock:
            self._closing = True
            deadline = time.monotonic() + drain_timeout
            while self._inflight_enqueues > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error("Timed out waiting for in-flight enqueues")
                    return False
                self._no_inflight.wait(timeout=remaining)

        # 모든 in-flight enqueue 완료 → sentinel은 반드시 큐의 마지막
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            self._queue.put(self._sentinel, timeout=remaining)
        except queue.Full:
            logger.error("Cannot enqueue sentinel: queue full after drain wait")
            return False

        self._thread.join(timeout=max(0, deadline - time.monotonic()))
        ok = not self._thread.is_alive()
        if not ok:
            logger.error(
                "SSD writer thread did not exit within %ss, ~%d items may be lost",
                drain_timeout, self._queue.qsize()
            )
        return ok

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)
```

### 3. KV manager — 2-phase lock (RAM in lock, SSD I/O outside)

```python
def cache_block(self, block_hash, token_ids, kv_data,
                tiered_cache=None, exclude_ids=None,
                ssd_policy="evict_only") -> int | None:
    # Phase 1: lock 안 — RAM 할당/등록
    with self.lock:
        if block_hash in self.hash_table:
            return None
        block = self.pool.get_free_block()
        block.block_hash = block_hash
        block.token_ids = list(token_ids)
        block.kv_data = kv_data
        block.ref_count = 1
        block.last_accessed = time.time()
        self.hash_table[block_hash] = block.block_id
        result_id = block.block_id

    # Phase 2: lock 밖 — SSD write-through
    if ssd_policy == "write_through" and tiered_cache is not None:
        tiered_cache.write_through(block_hash, kv_data, len(token_ids))

    return result_id
```

`TieredKVCache.write_through()`:
```python
def write_through(self, block_hash, kv_data, num_tokens):
    if self._writer:  # async mode
        self._writer.enqueue(block_hash, kv_data, num_tokens)
    elif self.ssd:    # sync mode
        self.ssd.save_block(block_hash, kv_data, num_tokens)
```

### 4. Scheduler shutdown

```python
def stop(self):
    self._shutdown_clean = True
    self._shutdown_partial_flush = False

    if self._ssd_writer:
        writer_ok = self._ssd_writer.stop(drain_timeout=5.0)
        if not writer_ok:
            self._shutdown_clean = False
            logger.critical(
                "SSD writer did not stop cleanly. Partial durability state."
            )

    if self._tiered_cache and self._tiered_cache.ssd:
        self._tiered_cache.ssd.flush()
        if not self._shutdown_clean:
            self._shutdown_partial_flush = True
            logger.error(
                "Partial SSD flush after incomplete writer shutdown. "
                "validate_index() will reconcile on next startup."
            )

# os._exit() 호출 금지. 강제 종료는 프로세스 매니저가 담당:
# - systemd: TimeoutStopSec=10
# - Docker: --stop-timeout 10
# - Kubernetes: terminationGracePeriodSeconds: 10
```

### 5. Stats 소유권

각 모듈이 **prefix가 다른** 로컬 카운터를 소유하고, `/health`에서 `dict.update()` merge.
키 충돌 없음을 보장.

```
ssd_cache._stats (prefix: ssd_)
  ssd_save_success, ssd_save_fail, ssd_save_dedup_skip
  ssd_stale_index_cleaned, ssd_hash_collision_detected

ssd_writer._stats (prefix: writer_)
  writer_enqueue_total, writer_enqueue_dedup_skip
  writer_save_success, writer_save_fail
  writer_queue_full_sync_fallback
  writer_retry_attempts, writer_retry_final_fail

kv_cache_manager._stats (prefix: kv_)
  kv_promote_hits, kv_promote_fail
  kv_lookup_hits, kv_lookup_miss

scheduler.get_cache_stats() → dict.update() merge (키 충돌 없음)
```

**병합 규칙:** `dict.update()`로 단순 합산. 모듈 prefix가 다르므로 덮어쓰기 불가.
async 모드에서 `writer_save_success`와 `ssd_save_success`는 동일한 save_block() 호출에서
각각 writer 관점/storage 관점으로 독립 카운팅됨 (1:1 대응, 중복 아님).

Health 노출 필드:
- `shutdown_clean`: bool
- `shutdown_partial_flush`: bool

---

## 실패 시나리오 정책

| 상황 | `best_effort` | `persistent` |
|------|--------------|-------------|
| SSD write 실패 | 로그 + 카운터, RAM 유지 | retry queue에 적재, 최대 N회 재시도 |
| retry N회 초과 | — | 포기, RAM만 유지 + WARNING |
| RAM 롤백 | 안 함 | 안 함 |
| SSD load 실패 | stale index 정리 → uncached fallback | 동일 |
| Writer queue full | 3단계 backpressure (put_nowait → put(50ms) → sync fallback) | 동일 |
| Hash collision (blake2b-128) | non-cacheable 격리 (1시간 TTL) + WARNING + 카운터 | 동일 |
| Writer stop timeout | `shutdown_clean=False`, partial flush + health 노출 | 동일 |

---

## 태스크 분해 (구현 순서)

### Phase 1 — Config + Stats 인프라
- [ ] P1.1: `config.py`에 SSD 필드 추가 (위 Config 스키마)
- [ ] P1.2: `ssd_cache.py`에 `_stats` dict + `get_stats()` 추가
- [ ] P1.3: `kv_cache_manager.py`에 SSD 관련 카운터 추가
- [ ] P1.4: `scheduler.get_cache_stats()`에서 snapshot merge 방식으로 통합 노출
- [ ] P1.5: 테스트 — `evict_only` 기본값으로 회귀 없음 확인

### Phase 2 — Sync Write-Through + Dedup
- [ ] P2.1: `ssd_cache.save_block()`에 dedup guard (index + 파일 존재 + num_tokens 검증)
- [ ] P2.2: `ssd_cache` 메타데이터에 `num_tokens` 필드 추가
- [ ] P2.3: `ssd_cache`에 non-cacheable 격리 set 추가 (collision 방어)
- [ ] P2.4: `ssd_cache`에 `threading.Lock` 추가 (thread-safe 승격)
- [ ] P2.5: `kv_cache_manager.cache_block()`에 write-through 경로 추가 (2-phase lock)
- [ ] P2.6: `evict_to_ssd()`에서 이미 SSD에 있는 블록 skip
- [ ] P2.7: `persistent` 모드 sync retry (즉시 N회, async writer 도입 전 임시)
- [ ] P2.8: 테스트 — write-through → SSD 존재, dedup skip, stale 재저장
- [ ] P2.9: 테스트 — eviction 시 이중 저장 없음

### Phase 3 — Crash Recovery
- [ ] P3.1: `ssd_cache.validate_index()` — orphan 파일 정리 + missing 엔트리 제거
- [ ] P3.2: scheduler 초기화 시 `validate_index()` 호출 + 통계 리포트 로깅
- [ ] P3.3: 테스트 — 손상 index, orphan 파일, missing 파일

### Phase 4 — Async Writer + Retry
- [ ] P4.1: `SSDWriterThread` (inflight counter + sentinel + non-daemon + 3 locks)
- [ ] P4.2: 3단계 backpressure (put_nowait → put(50ms) → sync fallback)
- [ ] P4.3: `persistent` 모드 async retry queue (Phase 2의 sync retry 대체)
- [ ] P4.4: `TieredKVCache`에 writer thread 통합 + `write_through()` 메서드
- [ ] P4.5: `scheduler.stop()` → `writer.stop()` → `ssd.flush()` + shutdown 상태 노출
- [ ] P4.6: 테스트 — sentinel 종료, backpressure, drain 보장, 부하 손실 없음

### Phase 5 — TTL 개선
- [ ] P5.1: 시간 기반 prune 트리거 (step 기반과 병행)
- [ ] P5.2: async writer 지연 감안 `last_accessed` 검증

---

## 테스트 계획

### 단위 테스트
- write-through 모드에서 새 block 생성 시 SSD index 즉시 반영
- async writer dedup 동작 (동일 hash 다중 요청 → 1회 write)
- shutdown drain/flush 보장 (sentinel 종료, inflight 대기)
- SSD write 실패 정책 (best_effort / persistent) 분기
- dedup guard: stale index (파일 없음) 시 재저장
- collision 감지 시 non-cacheable 격리 + WARNING 반환
- stats lock: producer+worker 동시 갱신 시 카운터 유실 없음

### 통합 테스트
- miss → prefill → write-through → 재요청 hit (RAM)
- 프로세스 재시작 후 hit (SSD → promote)
- TTL 만료 prune 후 miss 복귀
- 고부하 상황 (queue pressure) 에서 손실/교착 없음

### 회귀 테스트
- 기존 `evict_only` 모드 동작 불변
- 기존 A~G 감사 이슈 동작 불변

---

## 성능/운영 트레이드오프

| 모드 | 장점 | 단점 |
|------|------|------|
| Sync write-through | 내구성/정합성 단순 | write latency 증가 |
| Async write-through | 추론 경로 latency 보호 | crash 직전 소량 미반영 가능 |
| evict_only (fallback) | 기존 동작 불변 | 재시작 시 SSD prefix hit 없음 |

권장 기본값:
- 운영: `write_through` + `ssd_async_writes=True`
- 개발/검증: `write_through` + `ssd_async_writes=False` (sync)
- fallback: `evict_only` (안정성 이슈 시 즉시 롤백)

---

## 최종 판단 기준 (Done Definition)

- 재시작 후 SSD prefix hit율이 유의미하게 확보됨
- prefill 절감률이 기존 대비 개선됨
- 장시간 운영에서 index/data 불일치율 허용치 이내
- shutdown 후 metadata 유실 경고 없음
- 기존 회귀 테스트 + 신규 write-through 테스트 전체 통과
- `shutdown_clean=True` 상태로 정상 종료됨

---

## 파일별 변경 사항 요약

| 파일 | 변경 |
|------|------|
| `config.py` | SSD 필드 6개 추가 |
| `ssd_cache.py` | Lock, dedup guard, collision 격리, validate_index(), stats, save_block 시그니처 변경 |
| `ssd_writer.py` (신규) | SSDWriterThread (Phase 4) |
| `kv_cache_manager.py` | cache_block() write-through 경로, 2-phase lock, stats |
| `scheduler.py` | stats merge, stop() shutdown 상태, startup validate_index() |
| `sequence_cache.py` | 변경 없음 |
