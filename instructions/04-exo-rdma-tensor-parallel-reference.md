# exo 기반 RDMA + Tensor Parallel 활성화 적용 플랜 (mlx-lm-server)

작성일: 2026-02-09  
레퍼런스 exo 커밋: `2fbdb27`

## 1) 핵심 결론

- `exo`의 핵심은 시작 시점에 분산 backend를 명시적으로 초기화하고(`mx.distributed.init`), 그 다음 shard-aware 로더를 타는 것입니다.
- 우리 `mlx_lm_server`는 현재 엔트리포인트에서 무조건 `mlx_lm.load(...)` 단일 경로를 타므로(`mlx_lm_server/__main__.py:14-17`), RDMA/JACCL + TP 경로가 실제로 활성화되지 않습니다.
- 단순히 init만 추가하면 끝이 아닙니다. TP에서는 모든 rank가 동일한 요청 시퀀스를 함께 forward해야 하므로, rank0 HTTP 입력을 다른 rank에 동기화하는 런타임 경로가 필수입니다.

## 2) exo에서 확인한 활성화 패턴

- backend 선택 + 환경 변수 주입 + init
  - ring: `MLX_HOSTFILE`, `MLX_RANK` 설정 후 `mx.distributed.init(backend="ring", strict=True)`
  - jaccl(RDMA): `MLX_IBV_DEVICES`, `MLX_RANK`, `MLX_JACCL_COORDINATOR` 설정 후 `mx.distributed.init(backend="jaccl", strict=True)`
  - 근거: `/tmp/exo/src/exo/worker/engines/mlx/utils_mlx.py:119-176`
- shard 타입별 로딩 분기
  - Tensor shard면 `tensor_auto_parallel(...)`
  - Pipeline shard면 `pipeline_auto_parallel(...)`
  - 근거: `/tmp/exo/src/exo/worker/engines/mlx/utils_mlx.py:270-277`
- 로드 후 동기화 barrier 수행
  - 근거: `/tmp/exo/src/exo/worker/engines/mlx/utils_mlx.py:290-292`

요약하면, `exo`는 "초기화와 로딩을 분산 1급 경로로 취급"하고 있습니다.

## 3) 우리 코드와의 갭

- 엔트리포인트 단일 경로
  - `mlx_lm_server/__main__.py:14-17`에서 `load(...)`만 사용.
- CLI/설정에서 분산 파라미터 부재
  - `mlx_lm_server/server.py:728-825`에 distributed backend/sharding 인자 없음.
  - `mlx_lm_server/config.py:60`의 `use_distributed`는 실질적으로 미사용.
- TP 런타임 동기화 부재
  - 현재 scheduler는 로컬 request queue만 소비.
  - TP는 모든 rank가 같은 요청/취소/종료 이벤트를 동일 순서로 받아야 안전.

## 4) 제안 아키텍처

### A. 부트스트랩 계층 신설 (`mlx_lm_server/distributed.py`)

목표: 시작점에서 분산 모드/백엔드/샤딩을 명시적으로 결정.

필드 제안 (`ServerConfig` + CLI):

- `distributed_mode`: `off | ring | jaccl`
- `distributed_sharding`: `tensor | pipeline` (기본 `tensor`)
- `distributed_strict`: bool (기본 `True`)
- `distributed_hostfile`: str | None (ring)
- `distributed_ibv_devices`: str | None (jaccl)
- `distributed_jaccl_coordinator`: str | None (jaccl)

초안 스니펫:

```python
# mlx_lm_server/distributed.py
from dataclasses import dataclass
import os
import mlx.core as mx

@dataclass
class DistributedContext:
    enabled: bool
    group: object | None
    rank: int
    world_size: int
    pipeline_group: object | None
    tensor_group: object | None


def init_distributed(config) -> DistributedContext:
    if config.distributed_mode == "off":
        return DistributedContext(False, None, 0, 1, None, None)

    if config.distributed_mode == "ring":
        if config.distributed_hostfile:
            os.environ["MLX_HOSTFILE"] = config.distributed_hostfile
        group = mx.distributed.init(backend="ring", strict=config.distributed_strict)
    elif config.distributed_mode == "jaccl":
        if config.distributed_ibv_devices:
            os.environ["MLX_IBV_DEVICES"] = config.distributed_ibv_devices
        if config.distributed_jaccl_coordinator:
            os.environ["MLX_JACCL_COORDINATOR"] = config.distributed_jaccl_coordinator
        group = mx.distributed.init(backend="jaccl", strict=config.distributed_strict)
    else:
        raise ValueError(f"Unknown distributed_mode: {config.distributed_mode}")

    rank = group.rank()
    world_size = group.size()
    pipeline_group = group if config.distributed_sharding == "pipeline" and world_size > 1 else None
    tensor_group = group if config.distributed_sharding == "tensor" and world_size > 1 else None
    return DistributedContext(True, group, rank, world_size, pipeline_group, tensor_group)
```

### B. 모델 로딩 분기 (`mlx_lm_server/__main__.py`)

목표: 분산이면 `sharded_load`, 단일이면 기존 `load` 유지.

초안 스니펫:

```python
from mlx_lm import load
from mlx_lm.utils import sharded_load
from mlx_lm_server.distributed import init_distributed

ctx = init_distributed(config)
if ctx.enabled and ctx.world_size > 1:
    if config.adapter_path is not None:
        raise ValueError("distributed mode currently does not support adapter_path")
    model, tokenizer = sharded_load(
        config.model,
        pipeline_group=ctx.pipeline_group,
        tensor_group=ctx.tensor_group,
    )
else:
    model, tokenizer = load(config.model, adapter_path=config.adapter_path)
```

### C. 런타임 역할 분리

목표: 포트 충돌 방지 + TP 동작 보장.

- rank0만 HTTP 서버(`uvicorn.run`) 실행.
- 모든 rank는 scheduler inference loop 실행.
- rank>0 프로세스도 종료 신호를 받아 loop를 멈출 수 있게 shutdown 이벤트 브로드캐스트 필요.

참고: upstream `mlx_lm.server`도 rank0만 HTTP를 띄우고 worker rank는 generation loop만 수행합니다(`mlx_lm/server.py:1709-1716`).

### D. 요청 동기화 버스 추가 (필수)

목표: rank0에서 받은 요청/취소/종료 이벤트를 전 rank에 동일 순서로 전파.

- `DistributedControlBus` 신설 제안 (`mlx_lm_server/distributed_bus.py`)
- `Scheduler`의 queue pop 직전에 버스에서 이벤트를 pull해서 로컬 queue/cancel set에 반영.
- 이벤트 타입: `submit`, `cancel`, `shutdown`.
- 전송 payload는 `InferenceRequest` dataclass + request_id.

초안 스니펫(개념):

```python
# rank0: local event -> broadcast
# rank>0: broadcast receive -> local apply

def share_object(obj, group, rank):
    if rank == 0:
        if obj is None:
            mx.eval(mx.distributed.all_sum(0, group=group))
            return None
        data = mx.array(pickle.dumps(obj), dtype=mx.uint8)
        mx.eval(mx.distributed.all_sum(data.size, group=group))
        mx.eval(mx.distributed.all_sum(data, group=group))
        return obj
    size = mx.distributed.all_sum(0, group=group).item()
    if size == 0:
        return None
    buf = mx.zeros(size, dtype=mx.uint8)
    buf = mx.distributed.all_sum(buf, group=group)
    return pickle.loads(buf)
```

참고 구현 패턴: `mlx_lm/server.py:609-643`.

### E. SSD/KV cache 네임스페이스 분리 (중요)

TP에서는 rank별 KV tensor shard가 서로 다릅니다. 현재 block hash만으로 파일명을 만들면 rank 간 충돌/오염 위험이 있습니다.

필수 가드:

- SSD 경로를 rank별 분리: `.../<fingerprint>/rank_<rank>/...`
- 혹은 block key에 `rank/world_size/sharding` salt 포함.
- 권장: 경로 분리(가시성/운영성 좋음).

초안 스니펫:

```python
ssd_dir = config.ssd_cache_dir / fingerprint
if dist_ctx.enabled:
    ssd_dir = ssd_dir / f"rank_{dist_ctx.rank}"
```

## 5) 구현 단계 (실행 순서)

1. Config/CLI 확장 + 검증 규칙 추가.
2. `distributed.py` 추가, `__main__.py` 로더 분기 적용.
3. rank0-only HTTP 실행 분기 적용.
4. `distributed_bus.py` + scheduler 이벤트 동기화 연결.
5. SSD rank namespace 분리.
6. 통합 테스트 및 실패 복구 시나리오 점검.

## 6) 테스트 계획

- 단위 테스트
  - CLI 파싱: backend/sharding 조합 검증.
  - distributed off/on에 따른 로더 분기 검증(`load` vs `sharded_load`).
  - rank 분기: rank0만 HTTP, rank>0는 HTTP 미기동.
  - SSD 경로가 rank별로 분리되는지 검증.
- 통합 테스트
  - 2-rank local ring에서 동일 요청 결과 일관성.
  - cancel/shutdown 이벤트가 전 rank 동기화되는지.
  - cache miss 후 prefill 결과가 각 rank SSD에 누적되는지.

## 7) 운영 관점 권장사항

- 1차 릴리즈는 `tensor`만 먼저 지원하고 `pipeline`은 feature flag로 보류.
- distributed 모드에서 `adapter_path`는 명시적으로 차단 후 추후 지원.
- 장애 복구 시 rank별 SSD index validate를 startup에 수행.

## 8) 적용 시 기대 효과

- 시작 포인트에서 RDMA(JACCL) + TP 경로가 명시적으로 활성화됨.
- 단순 init-only가 아니라 실제 요청 처리까지 분산 안전성이 확보됨.
- 향후 exo처럼 backend와 shard 전략을 분리 확장하기 쉬운 구조가 됨.

## 9) 참고 소스

- exo repo: <https://github.com/exo-explore/exo>
- exo distributed init: <https://github.com/exo-explore/exo/blob/2fbdb27/src/exo/worker/engines/mlx/utils_mlx.py#L119-L176>
- exo shard load 분기: <https://github.com/exo-explore/exo/blob/2fbdb27/src/exo/worker/engines/mlx/utils_mlx.py#L270-L277>
- exo auto-parallel 구현: <https://github.com/exo-explore/exo/blob/2fbdb27/src/exo/worker/engines/mlx/auto_parallel.py>
- mlx_lm sharded load: `mlx_lm/utils.py:492-580`
- mlx_lm distributed request sharing 패턴: `mlx_lm/server.py:609-643`

## 10) CLI 플래그 상세 명세 (구현용)

아래는 `mlx_lm_server/server.py:parse_args`에 바로 추가할 플래그 제안입니다.

- `--distributed-mode {off,ring,jaccl}`: 기본 `off`
- `--distributed-sharding {tensor,pipeline}`: 기본 `tensor`
- `--distributed-strict` / `--no-distributed-strict`: 기본 `True`
- `--distributed-hostfile PATH`: ring에서 사용
- `--distributed-ibv-devices PATH`: jaccl에서 사용 (`MLX_IBV_DEVICES`)
- `--distributed-jaccl-coordinator HOST:PORT`: jaccl에서 사용 (`MLX_JACCL_COORDINATOR`)

`ServerConfig` 필드 추가안:

```python
distributed_mode: str = "off"          # off|ring|jaccl
distributed_sharding: str = "tensor"   # tensor|pipeline
distributed_strict: bool = True
distributed_hostfile: str | None = None
distributed_ibv_devices: str | None = None
distributed_jaccl_coordinator: str | None = None
```

검증 규칙(필수):

1. `distributed_mode == "ring"`일 때 `distributed_hostfile` 필수.
2. `distributed_mode == "jaccl"`일 때 `distributed_ibv_devices`와 `distributed_jaccl_coordinator` 필수.
3. `distributed_mode == "off"`일 때 나머지 distributed 인자가 들어오면 warning 로그 출력.
4. `adapter_path` + `distributed_mode != "off"` 조합은 1차에서 명시적 에러.
5. `distributed_sharding == "pipeline"`은 1차 릴리즈에서 feature flag (`--enable-pipeline-distributed`) 없으면 에러.

## 11) 파일별 구현 포인트 (정확한 수정 위치)

1. `mlx_lm_server/config.py`
   - distributed 관련 필드 추가.
   - 기존 `use_distributed`는 deprecated 처리(유지하되 미사용 경고).
2. `mlx_lm_server/server.py`
   - `parse_args`에 CLI 추가.
   - argparse validation 규칙 추가.
   - `kwargs` 매핑에 distributed 필드 반영.
3. `mlx_lm_server/distributed.py` (신규)
   - `DistributedContext` 정의.
   - `init_distributed(config)` 구현.
   - `finalize_distributed(ctx)` 또는 no-op 정리 함수 추가.
4. `mlx_lm_server/__main__.py`
   - 시작 시 `ctx = init_distributed(config)` 호출.
   - 로더 분기: distributed면 `sharded_load`, 아니면 `load`.
   - rank0만 `uvicorn.run(...)` 실행.
   - rank>0은 scheduler loop만 유지 후 shutdown 이벤트 대기.
5. `mlx_lm_server/scheduler.py`
   - distributed control bus 주입 파라미터 추가.
   - `_batch_inference_step` 초반에 bus pull/적용.
   - `submit_request`, `cancel_request`, `shutdown`에서 rank0 이벤트 publish.
6. `mlx_lm_server/kv_cache_manager.py` 또는 `mlx_lm_server/__main__.py`
   - SSD 경로 rank namespace 분리 적용.
7. `tests/`
   - parse_args 분산 케이스 테스트 추가.
   - distributed 로더 분기 테스트 추가(mock 기반).
   - rank role 분기 테스트 추가(rank0/http, rank>0/no-http).

## 12) DistributedControlBus 상세 인터페이스 (구현 초안)

```python
from dataclasses import dataclass
from typing import Literal

ControlEventType = Literal["submit", "cancel", "shutdown", "noop"]

@dataclass
class ControlEvent:
    typ: ControlEventType
    request: InferenceRequest | None = None
    request_id: str | None = None

class DistributedControlBus:
    def publish_from_rank0(self, event: ControlEvent) -> None: ...
    def recv_on_all_ranks(self) -> ControlEvent: ...
```

핵심 구현 포인트:

1. 모든 step에서 `recv_on_all_ranks()`를 호출해 이벤트 순서를 맞춤.
2. work가 없으면 `noop` 이벤트를 보내 collective deadlock 방지.
3. `submit` 이벤트는 `InferenceRequest` 전체 payload 전송.
4. `cancel` 이벤트는 `request_id`만 전송.
5. `shutdown` 이벤트 수신 즉시 `_running=False`로 전환.

## 13) __main__ 진입 흐름 (구체 실행 순서)

1. `config = parse_args()`
2. `dist_ctx = init_distributed(config)` (여기서 `mx.distributed.init()` 1회)
3. model/tokenizer 로딩 분기
4. KV/SSD 초기화 (`ssd_dir` rank namespace 포함)
5. scheduler 생성 시 `dist_ctx` 및 `control_bus` 주입
6. `scheduler.run_inference_loop()` 실행
7. `rank == 0`: `uvicorn.run(app, ...)`
8. `rank > 0`: shutdown 이벤트 대기 루프 진입
9. 프로세스 종료 시 `scheduler.stop()`, SSD flush

참고용 스니펫:

```python
dist_ctx = init_distributed(config)
control_bus = None if not dist_ctx.enabled else DistributedControlBus(dist_ctx)

if dist_ctx.enabled and dist_ctx.world_size > 1:
    model, tokenizer = sharded_load(
        config.model,
        pipeline_group=dist_ctx.pipeline_group,
        tensor_group=dist_ctx.tensor_group,
    )
else:
    model, tokenizer = load(config.model, adapter_path=config.adapter_path)

scheduler = Scheduler(..., dist_ctx=dist_ctx, control_bus=control_bus)
scheduler.run_inference_loop()

if (not dist_ctx.enabled) or dist_ctx.rank == 0:
    uvicorn.run(app, host=config.host, port=config.port)
else:
    scheduler.join_worker_loop()  # shutdown event까지 대기
```

## 14) 실행 예시 (운영자 가이드)

ring + tensor:

```bash
mlx.launch \
  --backend ring \
  --hostfile /path/to/hosts.json \
  -n 2 \
  python -m mlx_lm_server \
  --model mlx-community/Qwen3-4B-4bit \
  --distributed-mode ring \
  --distributed-hostfile /path/to/hosts.json \
  --distributed-sharding tensor
```

jaccl(RDMA) + tensor:

```bash
mlx.launch \
  --backend jaccl \
  --env MLX_METAL_FAST_SYNCH=1 \
  -n 2 \
  python -m mlx_lm_server \
  --model mlx-community/Qwen3-4B-4bit \
  --distributed-mode jaccl \
  --distributed-ibv-devices /path/to/ibv_devices.json \
  --distributed-jaccl-coordinator 10.0.0.10:55000 \
  --distributed-sharding tensor
```

## 15) 수용 기준 (Definition of Done)

1. distributed off에서 기존 동작/성능 회귀 없음.
2. distributed on + world_size=2에서 rank0만 포트 점유.
3. 동일 요청 10회 반복 시 rank 불일치/교착 없음.
4. cancel/shutdown 이벤트가 모든 rank에서 일관 반영.
5. SSD 파일이 `rank_<n>` 하위에 분리 저장됨.
6. server restart 후 cache index validate 에러 없이 통과.

## 16) 리스크와 완화

1. collective deadlock 리스크
   - 완화: 매 step `noop` 브로드캐스트로 호출 횟수 강제 정렬.
2. payload 직렬화 비용
   - 완화: `submit` 시에만 full payload, 평소 `noop`은 빈 이벤트.
3. rank별 SSD 데이터 오염
   - 완화: rank namespace + fingerprint 동시 적용.
4. graceful shutdown 누락
   - 완화: shutdown 이벤트 mandatory + `atexit` stop 유지.
