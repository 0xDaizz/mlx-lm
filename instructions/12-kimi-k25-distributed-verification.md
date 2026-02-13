# 12. Kimi K2.5 분산 로딩 검증 (2-Node JACCL TP)

> Memory-Safe Distributed Loader를 612GB Kimi K2.5 모델로 검증하기 위한
> 2-node JACCL Tensor Parallelism 전체 절차 및 참조 문서.

---

## 목차

1. [개요](#1-개요)
2. [환경 정보](#2-환경-정보)
3. [핵심 구현](#3-핵심-구현)
4. [JACCL 설정](#4-jaccl-설정)
5. [연결 테스트](#5-연결-테스트)
6. [모델 로딩 테스트](#6-모델-로딩-테스트)
7. [트러블슈팅](#7-트러블슈팅)
8. [참고: exo 프로젝트의 JACCL 구현](#8-참고-exo-프로젝트의-jaccl-구현)
9. [진행 상태](#9-진행-상태)
10. [Git 커밋 이력](#10-git-커밋-이력)

---

## 1. 개요

### 목적

612GB Kimi K2.5 모델 (4-bit quantized, 1.03T params, 182 safetensors)을 2노드 JACCL Tensor Parallelism으로 로딩하여 memory-safe distributed loader를 검증한다.

### 문제

`mx.eval(model.parameters())` 호출 시 전체 가중치가 메모리에 올라가면서 512GB 노드의 물리 메모리를 초과하여 macOS 커널 패닉 (watchdog timeout)이 발생한다.

### 해결

`_chunked_eval_params()` -- 레이어별 `mx.eval()` + `gc.collect()` 방식으로 메모리 안전 로딩을 수행한다.

### 노드 구성

- hwStudio1 (512GB, macOS Tahoe 26.3) + hwStudio2 (512GB, macOS Tahoe 26.3)
- 연결: Thunderbolt 5 RDMA (80 Gb/s x 2 channels)

---

## 2. 환경 정보

### 하드웨어

| 항목 | hwStudio1 | hwStudio2 |
|------|-----------|-----------|
| 기종 | Mac Studio | Mac Studio |
| RAM | 512GB | 512GB |
| WiFi IP | 192.168.0.105 (en1) | 192.168.0.107 (en1) |
| Thunderbolt 5 | Bus 1 (Receptacle 2) + Bus 3 (Receptacle 4), 각 80 Gb/s | 동일 |
| TB 인터페이스 | en3, en5 | en3, en5 |
| RDMA 디바이스 | rdma_en2 ~ rdma_en7 | rdma_en2 ~ rdma_en7 |

### 소프트웨어

| 항목 | 값 |
|------|-----|
| macOS | Tahoe 26.3 (Darwin 25.3.0) |
| MLX | 0.30.6 (distributed=True) |
| Python | `.venv/bin/python` (프로젝트 venv) |
| mlx.launch | `.venv/bin/mlx.launch` |

### 모델 위치

| 노드 | 경로 | 비고 |
|------|------|------|
| hwStudio1 | `~/models/Kimi-K2.5/` (원본, 612GB) | `~/mlx-lm-server/models/Kimi-K2.5` -> symlink |
| hwStudio2 | `~/mlx-lm-server/models/Kimi-K2.5/` (직접 복사본, 614GB) | |

---

## 3. 핵심 구현

### Memory-Safe Chunked Eval (`mlx_lm/utils.py`)

핵심 함수 목록:

| 함수 | 설명 |
|------|------|
| `_chunked_eval_params(model)` | 레이어별 `mx.eval` + `gc.collect` |
| `_check_memory_guard(label)` | 물리 메모리 대비 활성 메모리 체크 (기본: `max(10%, 5GB)`) |
| `_extract_layer_index(name)` | 파라미터 이름에서 레이어 인덱스 추출 |
| `_get_total_physical_memory()` | `sysctl hw.memsize`로 물리 메모리 조회 |

동작 방식:
- `sharded_load()`에서 `mx.eval(model.parameters())` 대신 `_chunked_eval_params(model)` 호출
- 환경변수 `MLX_MEMORY_GUARD_GB`로 가드 임계값을 GB 단위로 설정 가능

### Exo-Style Eval-Shard-Eval (`mlx_lm/utils.py`)
- `_eval_shard_eval(model, group)`: exo 프로젝트 방식의 레이어별 eval→shard→eval
- `_EvalShardEvalIter`: `model.shard()` 순회 시 끼어드는 리스트 래퍼
- `_find_layers_container(model)`: 모델 구조에서 transformer layers 위치 자동 탐색
- **핵심 원리**: shard 전에 레이어 가중치를 materialize → shard가 concrete 데이터 참조 → eval 후 원본 해제 가능
- **피크 메모리**: ~1 full layer + 1 sharded layer (기존 방식 대비 600GB → 15GB)

### 이전 방식이 실패한 이유
- `model.shard()` 가 모든 레이어에 lazy shard 그래프를 한번에 생성
- 모든 lazy 그래프가 원본 mmap 텐서를 참조 → 레이어별 eval해도 원본 해제 불가
- 결과: 원본 + sharded가 누적되어 600GB+ → OOM (exit 255)

### 메모리 로깅 (`mlx_lm_server/__main__.py`)

- 분산 모델 로드 전/후 active + peak memory 로깅 (GB 단위)

### 테스트 (`tests/test_chunked_eval.py`)

- 31개 테스트: chunked eval, memory guard, layer index extraction 등

---

## 4. JACCL 설정

### 중요: JACCL은 MLX에 내장

JACCL은 별도 pip 패키지가 아니다. MLX에 내장되어 있으며, 지원 백엔드는 다음과 같다:
- `ring`, `mpi`, `nccl`, `jaccl`, `jaccl-ring`

> **절대 ring으로 전환하지 말 것. 반드시 jaccl을 사용해야 한다.**

### RDMA 활성화 (한 번만, Recovery 모드에서)

```bash
rdma_ctl enable  # Recovery 모드에서 실행

# 확인
rdma_ctl status  # -> enabled
```

### 네트워크 설정 (매 부팅 시 필요 -- 일시적)

macOS가 TB 인터페이스를 `bridge0`에 자동으로 묶기 때문에 RDMA가 직접 접근할 수 없다. 매 부팅 시 다음 명령을 양쪽 노드에서 실행해야 한다.

> ⚠️ WiFi 대역(192.168.0.x/24)과 겹치는 IP를 TB 인터페이스에 절대 사용하지 말 것. route 명령으로 게이트웨이 IP를 TB로 리다이렉트하면 커널 패닉 발생.

#### hwStudio1

```bash
sudo ifconfig bridge0 down
sudo ifconfig bridge0 deletem en3
sudo ifconfig bridge0 deletem en5
sudo ifconfig en3 10.10.0.1/30 up
sudo ifconfig en5 10.10.1.1/30 up
```

#### hwStudio2

```bash
sudo ifconfig bridge0 down
sudo ifconfig bridge0 deletem en3
sudo ifconfig bridge0 deletem en5
sudo ifconfig en3 10.10.0.2/30 up
sudo ifconfig en5 10.10.1.2/30 up
```

> **주의:** 이 설정은 재부팅 시 초기화된다. 영구 설정을 위해서는 LaunchDaemon 작성이 필요하다 (미구현).

> ⚠️ `route add` / `route change` 명령은 사용하지 않음. `/30` 서브넷이 자동으로 라우트를 생성하며, WiFi 게이트웨이 IP를 TB로 리다이렉트하면 커널 패닉 발생.

### 호스트 파일 (`examples/distributed/hosts-hwstudio.json`)

```json
{
    "backend": "jaccl",
    "envs": ["MLX_METAL_FAST_SYNCH=1"],
    "hosts": [
        {
            "ssh": "hwStudio1.local",
            "ips": ["192.168.0.105"],
            "rdma": [null, ["rdma_en3", "rdma_en5"]]
        },
        {
            "ssh": "hwStudio2.local",
            "ips": ["192.168.0.107"],
            "rdma": [["rdma_en3", "rdma_en5"], null]
        }
    ]
}
```

#### 필드 설명

| 필드 | 설명 |
|------|------|
| `ssh` | mDNS 호스트명 (SSH 접속용) |
| `ips` | WiFi IP (coordinator용). **TB IP가 아님!** |
| `rdma` | RDMA 디바이스 매트릭스. `rdma[i][j]` = 노드 i에서 노드 j로의 RDMA 인터페이스 목록. 자기 자신은 `null`. |

> **참고 (exo 프로젝트 기준):** coordinator IP는 WiFi/Ethernet (안정적) 사용. TB IP는 사용하지 않는다.

---

## 5. 연결 테스트

### 테스트 스크립트 (`/tmp/test_distributed.py`)

```python
"""Minimal distributed connectivity test."""
import mlx.core as mx

group = mx.distributed.init()

rank = group.rank()
world = group.size()

# Simple all_sum test
data = mx.array([rank + 1.0])
mx.eval(data)
result = mx.distributed.all_sum(data, group=group)
mx.eval(result)

expected = world * (world + 1) / 2
print(f"[Rank {rank}/{world}] all_sum result: {result.item()} (expected: {expected})")

# Memory info
try:
    active = mx.metal.get_active_memory()
    print(f"[Rank {rank}/{world}] Active memory: {active / (1024**3):.2f} GB")
except AttributeError:
    pass

print(f"[Rank {rank}/{world}] Distributed test PASSED")
```

### 실행 명령

```bash
cd /Users/hw/mlx-lm-server
.venv/bin/mlx.launch \
  --backend jaccl \
  --hostfile examples/distributed/hosts-hwstudio.json \
  --verbose \
  --python /Users/hw/mlx-lm-server/.venv/bin/python \
  -- /tmp/test_distributed.py
```

### 기대 결과

```
[Rank 0/2] all_sum result: 3.0 (expected: 3.0)
[Rank 1/2] all_sum result: 3.0 (expected: 3.0)
[Rank 0/2] Distributed test PASSED
[Rank 1/2] Distributed test PASSED
```

---

## 6. 모델 로딩 테스트

JACCL 연결 테스트가 성공한 후에 진행한다.

### 실행 명령

```bash
cd /Users/hw/mlx-lm-server
.venv/bin/mlx.launch \
  --backend jaccl \
  --hostfile examples/distributed/hosts-hwstudio.json \
  --verbose \
  --python /Users/hw/mlx-lm-server/.venv/bin/python \
  -- -m mlx_lm.generate \
  --model models/Kimi-K2.5 \
  --prompt "Hello, world!"
```

### 확인 사항

- **메모리 safe 로딩:** 커널 패닉 없이 완료
- **chunked eval 로그:** 레이어별 eval 로그 출력
- **memory guard:** 임계값 이하로 유지
- **추론 결과:** 정상적인 텍스트 생성

---

## 7. 트러블슈팅

### errno 22 (EINVAL) -- "Changing queue pair to RTR failed"

- **원인:** `bridge0`이 UP 상태이거나 `en3`/`en5`가 bridge member로 남아있음
- **해결:** `bridge0 down` + `deletem en3` + `deletem en5`

### errno 60 (ETIMEDOUT)

- **원인:** `en3`/`en5`가 아직 bridge member (`bridge0 down`만으로는 부족)
- **해결:** `sudo ifconfig bridge0 deletem en3` + `sudo ifconfig bridge0 deletem en5` 실행

### mlx.launch not found

- **원인:** 시스템 PATH에 없음
- **해결:** `.venv/bin/mlx.launch` 사용 (venv 안에 있음)

### mx.distributed.Group() TypeError

- **원인:** `Group()`은 생성자가 없음
- **해결:** `group = mx.distributed.init()` 사용

### WiFi 끊김

- **원인:** 네트워크 location 변경 시 WiFi도 영향받음
- **해결:** `bridge0 down`/`deletem`만 실행. 네트워크 location을 건드리지 말 것.

### 커널 패닉 (route 명령)

- **원인:** WiFi 게이트웨이 IP(192.168.0.1)를 `route change -interface en3`로 TB에 할당 시 null pointer dereference 발생
- **해결:** TB IP는 WiFi와 다른 대역(10.10.0.x) 사용, route 명령 사용하지 않음

### exit code 255 (OOM kill during model load)
- 원인: `model.shard()` + `_chunked_eval_params()` 조합으로 lazy 그래프가 원본 텐서를 전부 참조
- 해결: `_eval_shard_eval()` 사용 (exo-style eval→shard→eval)
- `mx.clear_cache()`만으로는 불충분 — lazy 참조가 살아있으면 Metal 버퍼 해제 불가

### 토크나이저 에러 (Kimi K2.5)
- 증상: `ValueError: type of None unknown: <class 'NoneType'>`
- 원인: `generate()` → `stream_generate()`가 `tokenizer.encode(prompt, add_special_tokens=True)` 호출 시, `add_special_tokens`가 `**kwargs`로 전달되어 Kimi 토크나이저의 `super().encode()` 경로로 빠짐 (`tokenization_kimi.py:172-174`). parent의 `encode()`는 tiktoken vocab과 호환되지 않아 `input_ids=None` 반환
- 해결: `tokenizer.encode(prompt)` 로 kwargs 없이 호출하면 Kimi 자체 tiktoken 경로 사용. 또는 `stream_generate()`에 이미 인코딩된 `mx.array(prompt_tokens)`를 전달

---

## 8. 참고: exo 프로젝트의 JACCL 구현

exo 프로젝트에서 참고한 핵심 패턴:

- `bridge0`을 파괴(destroy)하고 "exo" network location을 생성
- coordinator IP로 WiFi IP를 사용 (TB IP 아님)
- RDMA 디바이스 매트릭스를 JSON 파일로 작성, `MLX_IBV_DEVICES` 환경변수에 파일 경로 설정
- `mx.distributed.init(backend="jaccl", strict=True)` 직접 호출

> **차이점:** 우리는 exo처럼 network location을 변경하지 않는다 (WiFi 끊김 문제 때문에 `bridge0 down`/`deletem`만 수행).

---

## 로딩 벤치마크 결과

| 항목 | 값 |
|------|-----|
| 모델 | Kimi K2.5 (612 GB, 4-bit, 182 safetensors) |
| 노드 | 2× Mac Studio (512 GB RAM) |
| 연결 | JACCL over TB5 RDMA (80 Gb/s × 2) |
| 로딩 시간 | ~96초 |
| 노드당 Active Memory | 309.91 GB |
| 노드당 Peak Memory | 319.98 GB |
| 여유 메모리 | ~192 GB (512 - 320) |
| 커널 패닉 | 없음 |

### 추론 벤치마크

| 항목 | 값 |
|------|-----|
| 프롬프트 | "What is 2+2? Answer in one sentence." |
| Prompt TPS | 49.6 tok/s |
| Generation TPS | 23.9 tok/s |
| 생성 토큰 수 | 8 |
| 응답 | "2+2 equals 4." |
| 총 추론 시간 | ~3초 |

---

## 9. 진행 상태

### 완료

- [x] Memory-safe chunked eval 구현 (`_chunked_eval_params`, `_check_memory_guard`)
- [x] 테스트 31개 작성 및 통과
- [x] 모델 백업 (`~/models/Kimi-K2.5`)
- [x] hwStudio2 환경 셋업 (repo sync, venv 복사, MLX/mlx.launch 확인)
- [x] RDMA 활성화 확인 (양쪽 모두)
- [x] TB 인터페이스 식별 (en3, en5)
- [x] Ring 백엔드 연결 테스트 성공 (ring은 참고용, 실제는 JACCL 사용)
- [x] exo JACCL 구현 방식 연구
- [x] bridge0 down + deletem en3/en5 (양쪽 모두)
- [x] 호스트 파일 작성 (WiFi IP + RDMA 매트릭스)
- [x] `_eval_shard_eval` 구현 (exo-style interleaved eval-shard-eval)
- [x] OOM 근본 원인 분석 (lazy shard graph 참조 수명 문제)
- [x] JACCL 연결 테스트 성공 (10.10.0.x 대역)
- [x] `sharded_load()` 직접 호출 성공 (309.91 GB, 96초, peak 319.98 GB)
- [x] `_probe` 영향 없음 확인 (격리 테스트 통과)
- [x] 토크나이저 에러 원인 분석 및 우회 (kwargs 없이 encode 호출)
- [x] 추론(generation) 테스트 성공 ("2+2 equals 4.", 23.9 tok/s)

### 미완료

- [ ] `generate()` 함수의 Kimi 토크나이저 호환성 수정 (add_special_tokens 처리)
- [ ] 메모리 사용량 벤치마크 (다양한 프롬프트)
- [ ] LaunchDaemon 작성 (네트워크 설정 영구화)

---

## 10. Git 커밋 이력

| 커밋 | 설명 |
|------|------|
| `8662b1b` | chunked eval + memory guard 구현 |
| `c69e897` | .gitignore 강화 + 예제 호스트 설정 정리 |
| `c13642f` | feat(distributed): exo-style eval-shard-eval for memory-safe TP loading |

- 브랜치: `develop` (origin/develop에 push 완료)
