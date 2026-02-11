# Production Audit Report

> 목표: Spec Decode(10번) 구현 전, 기존 모든 구현의 프로덕션 레벨 검증
> 방법: 검증 → 수정 → 테스트 3사이클 반복
> 시작일: 2026-02-11
> 베이스라인: 968 passed, 3 skipped, 1 xfail

---

## Cycle 1: Deep Audit

### Phase 1 — 병렬 연구 (5개 에이전트) ✅ 완료

| Agent | 대상 영역 | CRIT | HIGH | MED | LOW | 테스트 갭 |
|-------|----------|:----:|:----:|:---:|:---:|:--------:|
| auditor-scheduler | Scheduler & Batch Inference | 0 | 3 | 6 | 3 | 12 |
| auditor-cache | KV Cache, SSD, Write-Through | 0 | 2 | 6 | 5 | 8 |
| auditor-server | API Server & Configuration | 0 | 0 | 4 | 10 | 8 |
| auditor-distributed | Distributed/TP Infrastructure | 0 | 1 | 6 | 4 | 8 |
| auditor-plans | Plan Conformance (86 items) | - | - | - | - | - |
| **합계** | | **0** | **6** | **22** | **22** | **36** |

**Plan Conformance:** 82/86 PASS (95.3%), 2 PARTIAL, 0 FAIL, 2 DEFERRED

### Phase 1 — HIGH Findings (수정 대상)

| ID | 영역 | 설명 | 실제 위험도 |
|----|------|------|-----------|
| CACHE-H1 | evict_to_ssd | Phase 1에서 캡처한 kv_data 참조가 Phase 2에서 stale될 수 있음 | 낮음 (Phase 3 가드) → 문서화 |
| CACHE-H2 | compute_block_hash | 부분 prefix chunk 무시 — API 계약 위반 가능 | 낮음 (외부 호출자 없음) → 문서화 |
| SCHED-3 | _process_cancellations_batch | _cancelled_lock 없이 _request_id_to_uid 접근 | 낮음 (단일 스레드) → 문서화 |
| SCHED-7 | stop() | inference loop과 _active_sequences 경합 | 낮음 (lock + 중복 신호 안전) → 문서화 |
| SCHED-13 | _insert_new_requests_batch | pool.blocks 직접 접근 | 낮음 (ref_count 보호) → 캡슐화 |
| DIST-2 | _broadcast_object | rank0 외 호출 시 데이터 손상 가능 | 중간 → assertion 추가 |

### Phase 1 — 수정 대상 MEDIUM Findings

| ID | 영역 | 설명 |
|----|------|------|
| DIST-1 | scheduler.py:804 | 팩토리 미사용 pickle.dumps — ControlEvent.batch() 사용으로 통일 |
| DIST-4 | _receive_object | 역직렬화 실패 시 영구 rank divergence — fatal 상태 전환 필요 |
| DIST-6 | _drain_bus_outbox | _signal_finish 후 _cleanup_result_buffers TOCTOU — cleanup 제거 |
| DIST-8 | distributed.py:83 | 죽은 pipeline_group 코드 제거 |
| CACHE-M4 | BlockPool.return_block | 이중 반환 방지 guard 추가 |
| SCHED-1 | MockResponse | prompt_cache 인터페이스 불일치 수정 |
| SCHED-10 | submit_request | 분산/비분산 stats 불일치 통일 |
| SRV-5 | _do_inference | `if not events` → `if events is None` 정밀화 |
| SRV-9 | config.py | deprecated use_distributed 필드 제거 |
| SRV-11 | completion_tokens | stop-sequence 트런케이션 후 토큰 수 부정확 |
| SRV-14 | /metrics | Prometheus content-type charset 추가 |

### Phase 1 — 핵심 테스트 갭 (추가 대상)

1. RestrictedUnpickler 화이트리스트 차단 테스트
2. concurrent cache_block + evict_to_ssd 테스트
3. SSD promote during block-level cache lookup 테스트
4. Double-stop 멱등성 테스트
5. Streaming disconnect cleanup 테스트
6. BlockPool double-return 감지 테스트
7. BatchGenerator error recovery 후 재생성 테스트
8. Bus payload size limit 테스트
9. MockResponse 인터페이스 정합성 수정
10. Non-streaming 완료 토큰 수 정확성 테스트

### Phase 2 — Fixes (진행 중)

#### 수정 그룹

| 그룹 | 에이전트 | 대상 findings | 상태 |
|------|---------|-------------|------|
| A: Distributed | fixer-distributed | DIST-1,2,4,6,8 + 테스트 | 대기 |
| B: Scheduler+Cache | fixer-scheduler | SCHED-1,3,7,10,13 + CACHE-M4 + 테스트 | 대기 |
| C: Server | fixer-server | SRV-5,9,11,14 + 테스트 | 대기 |

### Phase 3 — Test
*(Phase 2 완료 후)*

---

## Cycle 2: Re-Audit

*(Cycle 1 완료 후 시작)*

---

## Cycle 3: Final Verification

*(Cycle 2 완료 후 시작)*

---

## Summary

| Metric | Cycle 1 | Cycle 2 | Cycle 3 |
|--------|---------|---------|---------|
| Findings | 50 (6H/22M/22L) | - | - |
| Plan Conformance | 95.3% | - | - |
| Fixed | 진행 중 | - | - |
| Tests Added | 진행 중 | - | - |
| Tests Passing | 968 | - | - |
