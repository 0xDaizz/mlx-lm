# ERR-07: Idle 후 Rank 동기화 불일치 Deadlock

## 발견 일시
2026-02-14 21:10 (KST), Phase 1 Baseline 벤치마크 중

## 재현 조건
- 2-node JACCL TP-2 (hwstudio1 rank 0, hwstudio2 rank 1)
- Kimi-K2.5 (1T MoE, 612GB, ~306GB per node)
- bench_single 3건 처리 후 약 15분 idle → bench_batch 시작 시 deadlock 발생
- ERR-06 fix (stream injection) 적용 상태

## 증상
- inference_loop_stale_s: 1090초 (18분)
- submitted_requests: 5, accepted_requests: 3 (새 요청 2건이 accept되지 않음)
- dist_outbox_size: 3 (분산 메시지 미전송)
- 서버 health: OK 반환하지만 inference 처리 불가

## Thread Dump 분석

### Thread 1: `scheduler-inference-loop` (BLOCKED)
```
mx.eval() -> array::wait() -> Event::wait() -> IOSurfaceSharedEvent waitUntilSignaledValue -> iokit_user_client_trap
```
inference loop이 mx.eval() 호출 후 GPU Event를 기다리는 중. 내부 all_reduce가 완료되지 않아 영원히 대기.

### Thread 2: `StreamThread` (BUSY-POLLING, 100% samples)
```
StreamThread::thread_fn() -> all_reduce_impl<1, int, SumOp<int>> -> tbt_poll_cq (Thunderbolt RDMA)
```
이전 inference step의 잔여 all_sum 연산이 hwstudio2의 응답을 영원히 기다리는 RDMA busy-poll 상태.

### Thread 3: `inference-poll_0/1` (WAITING)
```
SimpleQueue.get() (566/722 samples) + lock_PyThread_acquire_lock (156/722)
```

## 메모리 상태 (deadlock 발생 시)

| 항목 | hwstudio1 (rank 0) | hwstudio2 (rank 1) |
|------|--------------------|--------------------|
| Wired | **5.2 GiB** (unwired!) | **316 GiB** (정상) |
| Anonymous pageable | **316.7 GiB** | **5.4 GiB** |
| IOAccelerator footprint | 310 GB | 310 GB |
| GPU active | **0.4%** (idle) | **100%** (busy-wait) |
| GPU Power | 1-2 mW | 3160-3192 mW |

## 근본 원인 분석

### Deadlock 메커니즘
```
1. bench_single 3건 완료 → 서버 idle 진입
2. rank 0: inference loop idle → GPU idle → Metal이 wired 페이지를 unwire (316GB → 5GB)
3. rank 1: all_sum() busy-wait 중 (GPU 100%, wired 316GB 유지)
4. idle 기간 동안 DistributedControlBus가 heartbeat/sync all_sum을 호출
5. rank 0과 rank 1의 all_sum 호출 횟수(collective count)가 불일치
6. 15분 후 bench_batch가 새 요청 제출 → inference loop가 mx.eval() 호출
7. mx.eval() 내부의 all_reduce가 rank 1과 매칭되지 않는 collective → RDMA poll 무한 대기
8. inference loop BLOCKED → 새 요청 accept 불가 → 서버 실질적 hang
```

### ERR-06과의 관계
- ERR-06: bus와 model이 **서로 다른 stream**에서 all_sum → 동시 실행 시 cross-match
- ERR-07: 같은 stream이지만 **idle 기간 동안 collective count 불일치** → 재활성화 시 mismatch
- ERR-06 fix (stream injection)는 동시성 문제를 해결했지만, idle 중 divergence 문제는 해결하지 못함

### 왜 idle이 문제를 일으키는가
- rank 0 (HTTP 서버): 요청이 없으면 inference loop가 멈춤 → bus sync만 all_sum 호출
- rank 1 (worker): inference loop에서 all_sum 대기 → bus sync의 all_sum을 inference all_sum으로 오인
- 결과: rank 0은 bus all_sum N번, rank 1은 inference all_sum M번 → N ≠ M이면 deadlock

## 즉시 조치
- 서버 재시작 (stop_server.sh → launch_baseline.sh)
- 벤치마크 사이 idle 시간 최소화 (연속 실행)

## 장기 해결 방안

### 방안 1: TCP/IPC Control Plane (instructions/18 참조)
- all_sum을 model tensor 연산 전용으로 제한
- bus 통신은 TCP socket으로 분리
- collective count 불일치 원천 차단

### 방안 2: Idle Keepalive
- idle 기간에도 양쪽 rank가 동일한 dummy all_sum을 주기적으로 호출
- collective count 동기화 유지
- 단점: idle 시에도 GPU 전력 소비

### 방안 3: Subprocess Isolation (exo 패턴)
- MLX inference를 child subprocess에서 실행
- idle deadlock 발생 시 child를 restart
- parent는 영향받지 않음

### 방안 4: Collective Count Tracking
- 각 rank의 all_sum 호출 횟수를 로깅
- divergence 감지 시 자동 recovery (양쪽 flush 후 재동기화)

## 참조
- ERR-06: instructions/19-err06-heartbeat.md, commit 47b4832
- 장기 계획: instructions/18-distributed-bus-tcp-migration.md
- MLX-LM PR #741: stream fix
- inspector agent 분석 (2026-02-14 21:14)

## 관련 파일
- `mlx_lm_server/scheduler.py` — inference loop, _batch_inference_step
- `mlx_lm_server/distributed_bus.py` — bus all_sum (generation_stream)
- `mlx_lm_server/__main__.py` — 서버 시작, signal handlers

## 적용된 수정 (2026-02-14)

### 근본 원인 확정
- **RNG Seed 미동기화**: 분산 모드에서 `mx.random` 상태가 각 rank마다 다름 → `temperature > 0`일 때 토큰 샘플링 divergence → EOS 타이밍 불일치 → `batch.filter()` divergence → TP model `all_sum` 횟수 불일치 → RDMA deadlock
- **`uids_to_remove` 로컬 전용 처리**: rank 0의 custom stop 결과가 rank >0에 공유되지 않음 → batch size divergence

### Fix 1: RNG Seed 동기화 (`__main__.py`)
- Scheduler 생성 직전에 `mx.distributed.all_sum(mx.random.state[0])` 호출
- 모든 rank의 RNG seed를 동일하게 설정
- Upstream `server.py:706-707` 패턴 적용

### Fix 2: `share_object()` 메서드 (`distributed_bus.py`)
- `DistributedControlBus`에 `share_object(obj)` 추가
- Rank 0의 객체를 pickle + all_sum으로 전 rank에 broadcast
- Empty일 때도 1회 all_sum (size=0) 호출하여 collective count 일관성 유지
- `restricted_loads` 사용 (기존 보안 whitelist 적용)

### Fix 3: `uids_to_remove` 분산 동기화 (`scheduler.py`)
- `_batch_inference_step()`에서 `_emit_tokens()` 후, `remove()` 전에 sync point 삽입
- `self._control_bus.share_object(uids_to_remove)` 호출
- Non-distributed 모드에서는 guard로 skip

### Fix 4: 분산 Greedy Override (`scheduler.py`)
- `MLX_DIST_FORCE_GREEDY=1` 환경변수로 opt-in 활성화
- 분산 모드에서 temperature를 0.0으로 강제 (interim safety)
- `req.temperature`는 변경하지 않음 (사용자 응답에 영향 없음)

### 검증 계획
- [ ] 단위 테스트 통과 (`pytest`)
- [ ] `temperature=0.7`로 12+ 요청 후 deadlock 미발생
- [ ] 3건 → 15분 idle → 추가 요청 → 정상 응답 (ERR-07 재현 시나리오)
- [ ] 단일 노드 모드 regression 없음
