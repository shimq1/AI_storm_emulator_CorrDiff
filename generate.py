# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- 1. 필수 라이브러리 임포트 ---
import contextlib # with 문을 위한 컨텍스트 관리 유틸리티
from concurrent.futures import ThreadPoolExecutor # 병렬 처리를 위한 스레드 풀
from functools import partial # 함수의 인자를 미리 고정하는 기능

import hydra # NVIDIA의 설정 관리 프레임워크 (YAML 파일로 설정 관리)
from omegaconf import OmegaConf, DictConfig # Hydra가 사용하는 설정 객체
from hydra.utils import to_absolute_path # 상대 경로를 절대 경로로 변환
import torch # 파이토치
import torch._dynamo # 파이토치 2.0의 JIT 컴파일러
from torch.distributed import gather # 분산 처리에서 여러 GPU의 결과를 모으는 함수
import numpy as np
import nvtx # NVIDIA Nsight 프로파일링을 위한 마커
import netCDF4 as nc # NetCDF 파일(.nc)을 다루기 위한 라이브러리

# NVIDIA Modulus(Physics-NEMO)의 내부 유틸리티 및 모듈 임포트
from physicsnemo.distributed import DistributedManager # 분산 처리(다중 GPU) 관리자
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper # 로깅 설정
from physicsnemo.utils.patching import GridPatching2D # 이미지를 작은 패치로 나누는 유틸리티
from physicsnemo import Module # Modulus의 기본 모델 클래스
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler # 확산 모델 샘플러
from physicsnemo.utils.corrdiff import ( # CorrDiff 모델 관련 유틸리티
    NetCDFWriter, # NetCDF 파일 쓰기 래퍼 클래스
    get_time_from_range, # 시간 범위에서 시간 목록 생성
    regression_step, # 회귀(Regression) 모델 예측 단계
    diffusion_step, # 확산(Diffusion) 모델 예측 단계
)

# 사용자가 직접 만든 헬퍼(helper) 및 데이터셋 스크립트 임포트
from helpers.generate_helpers import (
    get_dataset_and_sampler, # 데이터셋과 샘플러를 가져오는 함수
    save_images, # 생성된 이미지를 저장하는 함수
)
from helpers.train_helpers import set_patch_shape # 패치 크기 설정 함수
from datasets.dataset import register_dataset # 커스텀 데이터셋 등록 함수


# --- 2. 메인 함수 정의 ---
# @hydra.main: Hydra를 사용해 YAML 설정 파일을 읽어 cfg 객체로 전달
@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """
    설정 파일(cfg)에 따라 모델을 로드하고, 데이터셋으로부터 입력을 받아
    이미지(결과장)를 생성하여 NetCDF 파일로 저장하는 메인 스크립트.
    """

    # --- 3. 분산 처리 및 로깅 초기화 ---
    # 여러 GPU 또는 여러 노드에서 코드를 실행하기 위한 설정
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device # 현재 프로세스에 할당된 GPU 장치

    # 로깅 설정 (실행 과정 기록용)
    logger = PythonLogger("generate")
    # Rank 0 (보통 마스터 GPU)에서만 로그를 출력하도록 래핑
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log") # 로그를 파일로도 저장

    # --- 4. 앙상블 및 배치 설정 ---
    # 설정 파일에 지정된 앙상블 개수만큼 시드(seed) 생성
    seeds = list(np.arange(cfg.generation.num_ensembles))
    # 전체 시드를 여러 GPU에 분배하기 위해 배치로 나눔
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # 현재 GPU(rank)에 할당된 배치만 선택
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # 모든 GPU 프로세스가 이 지점까지 도달할 때까지 대기 (동기화)
    if dist.world_size > 1:
        torch.distributed.barrier()

    # --- 5. 데이터셋 및 모델 설정 파싱 ---
    # 예측을 수행할 시간 목록을 설정 파일에서 가져옴
    if cfg.generation.times_range and cfg.generation.times:
        raise ValueError("times_range와 times 둘 중 하나만 지정해야 합니다.")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    else:
        times = cfg.generation.times

    # 데이터셋 설정 로드
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type) # 커스텀 데이터셋을 시스템에 등록
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    # 예측 시간(lead time) 사용 여부 확인
    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    
    # 설정에 맞는 데이터셋과 데이터 로더 샘플러 생성
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    img_shape = dataset.image_shape() # 이미지 크기
    img_out_channels = len(dataset.output_channels()) # 출력 채널 수

    # 패치 기반 예측 설정 (큰 이미지를 작은 조각으로 나눠 처리)
    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")

    # 추론 모드 설정 (회귀 모델만, 확산 모델만, 또는 둘 다 사용)
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # --- 6. 사전 학습된 모델 로드 ---
    # 확산 모델(net_res) 로드
    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(to_absolute_path(res_ckpt_filename))
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last) # 평가 모드로 설정 후 GPU로 이동
        if cfg.generation.perf.force_fp16: # 16비트 부동소수점 사용 여부
            net_res.use_fp16 = True
        if hasattr(net_res, "amp_mode"): # 자동 혼합 정밀도(AMP) 비활성화
            net_res.amp_mode = False
    else:
        net_res = None

    # 회귀 모델(net_reg) 로드
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(to_absolute_path(reg_ckpt_filename))
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    # --- 7. 성능 최적화 설정 ---
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.reset()
        # torch.compile으로 모델을 JIT 컴파일하여 속도 향상
        if net_res:
            net_res = torch.compile(net_res, mode="reduce-overhead")

    # 확산 모델 샘플러 함수 설정
    if cfg.sampler.type == "deterministic": # 결정론적 샘플러
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            solver=cfg.sampler.solver,
        )
    elif cfg.sampler.type == "stochastic": # 확률적 샘플러
        sampler_fn = partial(stochastic_sampler, patching=patching)
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")

    # --- 8. 핵심 예측 함수 정의 ---
    # 이 함수는 한 번의 예측을 수행하는 로직을 담고 있음
    def generate_fn():
        with nvtx.annotate("generate_fn", color="green"):
            # 데이터로더에서 받은 입력 이미지(image_lr)를 GPU 메모리 형식에 맞게 변환
            img_lr = image_lr.to(memory_format=torch.channels_last)

            # 회귀 모델 실행 (저해상도 -> 고해상도 기본 예측)
            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=( # 생성할 출력의 크기
                            cfg.generation.seed_batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),
                        lead_time_label=lead_time_label,
                    )
            # 확산 모델 실행 (디테일 추가 및 현실성 향상)
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1] # 회귀 결과를 조건으로 사용
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=img_lr.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last),
                        rank=dist.rank,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                    )
            # 최종 출력 결정
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else: # all 모드: 회귀 결과와 확산 결과를 더함
                image_out = image_reg + image_res

            # 다중 GPU 환경일 경우, 모든 GPU의 결과를 rank 0 GPU로 모음
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [torch.zeros_like(image_out) for _ in range(dist.world_size)]
                else:
                    gathered_tensors = None
                torch.distributed.barrier()
                gather(image_out, gather_list=gathered_tensors if dist.rank == 0 else None, dst=0)
                if dist.rank == 0:
                    return torch.cat(gathered_tensors)
                else:
                    return None
            else:
                return image_out

    # --- 9. 메인 실행 루프 ---
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2) # 성능 측정 전 워밍업 스텝

    # rank 0 프로세스에서만 NetCDF 파일을 생성
    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        f.cfg = str(cfg) # 파일에 사용된 설정 정보를 속성으로 저장

    # 성능 프로파일링을 위한 컨텍스트 관리자
    torch_cuda_profiler = (torch.cuda.profiler.profile() if torch.cuda.is_available() else contextlib.nullcontext())
    torch_nvtx_profiler = (torch.autograd.profiler.emit_nvtx() if torch.cuda.is_available() else contextlib.nullcontext())
    
    with torch_cuda_profiler:
        with torch_nvtx_profiler:
            # 데이터셋을 순회하기 위한 데이터 로더 생성
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                # NetCDF 파일 쓰기를 위한 writer 객체 생성
                writer = NetCDFWriter(
                    f,
                    lat=dataset.latitude(),
                    lon=dataset.longitude(),
                    input_channels=dataset.input_channels(),
                    output_channels=dataset.output_channels(),
                    has_lead_time=has_lead_time,
                )
                # 파일 쓰기 작업을 백그라운드에서 처리할 스레드 풀 초기화
                # 이를 통해 다음 스텝 계산과 현재 스텝 결과 저장을 동시에 진행하여 속도 향상
                writer_executor = ThreadPoolExecutor(max_workers=cfg.generation.perf.num_writer_workers)
                writer_threads = []

            # GPU 타이머 객체 생성
            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            else: # CPU 환경을 위한 더미 타이머
                class DummyEvent:
                    def record(self): pass
                    def synchronize(self): pass
                    def elapsed_time(self, _): return 0
                start, end = DummyEvent(), DummyEvent()

            times = dataset.time()
            # 데이터 로더를 순회하며 각 시간 단계에 대한 예측 수행
            for index, (image_tar, image_lr, *lead_time_label) in enumerate(iter(data_loader)):
                time_index += 1
                if dist.rank == 0: logger0.info(f"starting index: {time_index}")
                if time_index == warmup_steps: start.record() # 워밍업 끝나면 시간 측정 시작

                # 입력 데이터를 GPU로 이동 및 타입 변환
                if lead_time_label:
                    lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                else:
                    lead_time_label = None
                image_lr = image_lr.to(device=device).to(torch.float32).to(memory_format=torch.channels_last)
                image_tar = image_tar.to(device=device).to(torch.float32)

                # 핵심 예측 함수 호출
                image_out = generate_fn()

                # rank 0 프로세스에서만 결과 저장
                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    # 파일 쓰기 작업을 별도의 스레드에 제출
                    writer_threads.append(
                        writer_executor.submit(
                            save_images, # 실제 저장 로직이 담긴 함수
                            writer, dataset, list(times),
                            image_out.cpu(), image_tar.cpu(), image_lr.cpu(),
                            time_index, index, has_lead_time,
                        )
                    )
            
            # --- 10. 종료 및 정리 ---
            end.record() # 시간 측정 종료
            end.synchronize()
            elapsed_time = (start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0)
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0 and use_cuda_timing:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s")
                logger.info(f"Average time per batch element = {average_time_per_batch_element} s")

            # 모든 백그라운드 쓰기 스레드가 끝날 때까지 대기
            if dist.rank == 0:
                for thread in list(writer_threads):
                    thread.result() # 스레드 완료 대기
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    # NetCDF 파일 닫기
    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()