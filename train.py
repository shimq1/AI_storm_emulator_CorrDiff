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

# 필요한 라이브러리들을 임포트합니다.
import os
import time
import psutil  # 시스템 및 프로세스 유틸리티 (메모리 사용량 확인 등)
from contextlib import nullcontext  # 'with' 문에서 아무 작업도 하지 않는 컨텍스트 매니저

import hydra  # 설정 관리를 위한 라이브러리
from hydra.utils import to_absolute_path  # 상대 경로를 절대 경로로 변환
from hydra.core.hydra_config import HydraConfig  # Hydra 설정 정보에 접근
from omegaconf import DictConfig, OmegaConf  # Hydra에서 사용하는 설정 객체
import torch
from torch.nn.parallel import DistributedDataParallel  # 분산 학습을 위한 DDP
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅
import nvtx  # NVIDIA Tools Extension for profiling
import wandb  # Weights & Biases 로깅

# Modulus(Physics-NEMO) 관련 모듈 임포트
from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
from physicsnemo.distributed import DistributedManager
# from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss, RegressionLossCE
from loss import ResidualLoss, RegressionLoss, RegressionLossCE
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

# 사용자 정의 데이터셋 및 훈련 헬퍼 함수 임포트
from datasets.dataset import init_train_valid_datasets_from_config, register_dataset
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)

# PyTorch 2.0의 `torch.compile` 관련 설정
torch._dynamo.reset()
# 캐시 크기 제한을 늘려 컴파일 오버헤드를 줄입니다.
torch._dynamo.config.cache_size_limit = 264
torch._dynamo.config.verbose = True  # 상세 로깅 활성화
torch._dynamo.config.suppress_errors = False  # 에러 발생 시 모든 세부 정보 표시
torch._logging.set_logs(recompiles=True, graph_breaks=True)  # 재컴파일 및 그래프 중단 정보 로깅


def checkpoint_list(path, suffix=".mdlus"):
    """
    주어진 경로에서 체크포인트 파일 목록을 오름차순으로 정렬하여 반환하는 헬퍼 함수입니다.
    파일 이름의 숫자 인덱스를 기준으로 정렬합니다.
    """
    checkpoints = []
    for file in os.listdir(path):
        if file.endswith(suffix):
            # 파일 이름에서 인덱스를 추출합니다.
            try:
                index = int(file.split(".")[-2])
                checkpoints.append((index, file))
            except ValueError:
                continue

    # 인덱스를 기준으로 정렬하고 파일 이름만 반환합니다.
    checkpoints.sort(key=lambda x: x[0])
    return [file for _, file in checkpoints]


# CUDA가 사용 가능할 때만 동작하는 안전한 프로파일러 유틸리티 함수들을 정의합니다.
def cuda_profiler():
    """CUDA 프로파일러 컨텍스트를 반환합니다. CUDA가 없으면 아무 작업도 하지 않습니다."""
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    """CUDA 프로파일링을 시작합니다."""
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    """CUDA 프로파일링을 중지합니다."""
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    """NVTX 프로파일링 이벤트를 생성하는 컨텍스트를 반환합니다."""
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()


# Hydra를 사용하여 "conf/config_training.yaml" 설정으로 CorrDiff 모델을 학습합니다.
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    """메인 학습 함수"""

    # 분산 학습 환경을 초기화합니다.
    DistributedManager.initialize()
    dist = DistributedManager()

    # 로거를 초기화합니다.
    if dist.rank == 0:  # 랭크 0 프로세스에서만 TensorBoard writer를 생성합니다.
        writer = SummaryWriter(log_dir=f"tensorboard/{HydraConfig.get().job.name}/{time.strftime('%m.%d_%H.%M', time.localtime())}")
    logger = PythonLogger("main")  # 일반 파이썬 로거
    logger0 = RankZeroLoggingWrapper(logger, dist)  # 랭크 0에서만 로그를 출력하는 래퍼
    
    # Weights & Biases (wandb)를 초기화합니다.
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name=f"CorrDiff-Training-{HydraConfig.get().job.name}",
        group="CorrDiff-DDP-Group",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg), # 설정 객체를 wandb에 기록하기 위해 컨테이너로 변환
        results_dir=cfg.wandb.results_dir,
    )

    # 설정 값들을 확정하고 파싱합니다.
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO: 더 나은 처리 방식 필요

    # 설정 파일에 지정된 사용자 정의 데이터셋을 등록합니다.
    register_dataset(cfg.dataset.type)
    logger0.info(f"사용 데이터셋: {cfg.dataset.type}")

    # 검증 데이터셋 설정이 있는지 확인합니다.
    if hasattr(cfg, "validation"):
        validation = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        validation = False
        validation_dataset_cfg = None

    # 성능 최적화 관련 설정을 파싱합니다.
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"  # 순수 FP16 사용 여부
    enable_amp = fp_optimizations.startswith("amp")  # 자동 혼합 정밀도(AMP) 사용 여부
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16 # AMP 데이터 타입 설정
    
    logger.info(f"결과물 저장 위치: {os.getcwd()}")
    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
    )
    
    # GPU당 배치 크기가 'auto'로 설정된 경우, 전체 배치 크기를 GPU 수로 나누어 자동 계산합니다.
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    # 중단된 학습을 재개하기 위해 현재까지 처리된 이미지 수를 불러옵니다.
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
        )
    except Exception:
        cur_nimg = 0

    # 재현성을 위해 시드를 설정하고, CUDA 및 cuDNN 설정을 구성합니다.
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()

    # 데이터셋과 데이터로더를 인스턴스화합니다.
    data_loader_kwargs = {
        "pin_memory": True,  # GPU로 데이터 전송 시 속도 향상
        "num_workers": cfg.training.perf.dataloader_workers, # 데이터 로딩에 사용할 워커 수
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None, # 미리 데이터를 가져올 양
    }
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        validation=validation,
        sampler_start_idx=cur_nimg, # 학습 재개를 위한 샘플러 시작 인덱스
    )

    # 이미지 설정 정보를 파싱하고 모델 인자를 업데이트합니다.
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    
    # 고해상도(HR) 이미지의 평균을 조건으로 사용하는 경우, 입력 채널 수를 늘립니다.
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # 패치 기반 학습 관련 설정을 처리합니다.
    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None
        
    # 패치 모양을 파싱합니다.
    if (
        cfg.model.name == "patched_diffusion"
        or cfg.model.name == "lt_aware_patched_diffusion"
    ):
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None
        
    # 패치 크기가 원본 이미지 크기보다 크거나 같으면 경고를 출력하고 패칭을 사용하지 않습니다.
    if (
        patch_shape_x
        and patch_shape_y
        and patch_shape_y >= img_shape[0]
        and patch_shape_x >= img_shape[1]
    ):
        logger0.warning(
            f"패치 크기 {patch_shape_y}x{patch_shape_x}가 이미지 크기 {img_shape[0]}x{img_shape[1]}보다 큽니다. 패칭을 사용하지 않습니다."
        )
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    
    if use_patching:
        # 이미지에서 랜덤 패치를 추출하고 배치하는 유틸리티
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("패치 기반 학습이 활성화되었습니다.")
    else:
        patching = None
        logger0.info("패치 기반 학습이 비활성화되었습니다.")
        
    # 패치 기반 모델을 사용할 경우, 전역 채널을 보간하기 위해 입력 채널 수를 늘립니다.
    if use_patching:
        img_in_channels += dataset_channels

    # 모델을 인스턴스화하고 디바이스로 이동시킵니다.
    model_args = {  # 모든 네트워크의 기본 파라미터
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }
    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels
    if hasattr(cfg.model, "model_args"):  # 설정 파일에서 기본값을 덮어씁니다.
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    use_torch_compile = False
    use_apex_gn = False
    profile_mode = False

    # `torch.compile` 사용 여부 설정
    if hasattr(cfg.training.perf, "torch_compile"):
        use_torch_compile = cfg.training.perf.torch_compile
    # Apex GroupNorm 사용 여부 설정
    if hasattr(cfg.training.perf, "use_apex_gn"):
        use_apex_gn = cfg.training.perf.use_apex_gn
        model_args["use_apex_gn"] = use_apex_gn
    # 프로파일링 모드 사용 여부 설정
    if hasattr(cfg.training.perf, "profile_mode"):
        profile_mode = cfg.training.perf.profile_mode
        model_args["profile_mode"] = profile_mode

    if enable_amp:
        model_args["amp_mode"] = enable_amp

    # 설정된 모델 이름에 따라 적절한 모델 클래스를 인스턴스화합니다.
    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif (
        cfg.model.name == "lt_aware_ce_regression"
        or cfg.model.name == "lt_aware_regression"
    ):
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name == "patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    else:
        raise ValueError(f"잘못된 모델 이름입니다: {cfg.model.name}")

    # 모델을 학습 모드로 설정하고, 그래디언트 계산을 활성화하며, 디바이스로 이동시킵니다.
    model.train().requires_grad_(True).to(dist.device)

    # Apex GroupNorm을 사용하는 경우, 메모리 포맷을 channels_last로 변경하여 성능을 향상시킵니다.
    if use_apex_gn:
        model.to(memory_format=torch.channels_last)

    # 회귀 모델과 패치 기반 학습을 동시에 사용하는 경우 에러를 발생시킵니다.
    if (
        cfg.model.name
        in ["regression", "lt_aware_regression", "lt_aware_ce_regression"]
        and patching is not None
    ):
        raise ValueError(
            f"회귀 모델({cfg.model.name})은 패치 기반 학습과 함께 사용할 수 없습니다."
        )

    # 분산 학습(DDP)을 설정합니다.
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,  # 사용되지 않는 파라미터 찾기 활성화
            bucket_cap_mb=35, # 그래디언트 버킷 크기
            gradient_as_bucket_view=True, # 그래디언트 버킷 뷰 사용
        )
    
    # wandb.watch를 사용하여 모델의 그래디언트와 파라미터를 추적합니다. (랭크 0에서만)
    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(model)

    # 저장된 모델 체크포인트가 있으면 불러옵니다.
    try:
        load_checkpoint(path=checkpoint_dir, models=model)
    except Exception:
        pass

    # 사전 학습된 회귀 모델 체크포인트가 있으면 불러옵니다. (주로 디퓨전 모델의 조건으로 사용)
    if (
        hasattr(cfg.training.io, "regression_checkpoint_path")
        and cfg.training.io.regression_checkpoint_path is not None
    ):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"회귀 모델 체크포인트를 찾을 수 없습니다: {regression_checkpoint_path}"
            )
        
        logger0.info(f"회귀 모델 체크포인트 경로: {regression_checkpoint_path}")
        logger0.info(f"use_apex_gn: {use_apex_gn}, enable_amp: {enable_amp}, profile_mode: {profile_mode}")
        regression_net = Module.from_checkpoint(
            regression_checkpoint_path
            #! If override_args activate, model malfunction occurs.
            # , override_args={"use_apex_gn": use_apex_gn}
        )
        regression_net.amp_mode = enable_amp
        regression_net.profile_mode = profile_mode
        regression_net.eval().requires_grad_(False).to(dist.device) # 평가 모드로 설정하고 그래디언트 계산 비활성화
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        logger0.success("사전 학습된 회귀 모델을 성공적으로 불러왔습니다.")
    else:
        regression_net = None

    # `torch.compile`을 사용하여 모델을 컴파일합니다. (성능 향상)
    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    # 그래디언트 누적(accumulation) 횟수를 계산합니다.
    # (GPU당 배치 크기 * GPU 수)가 전체 배치 크기보다 작을 때 자동으로 사용됩니다.
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"그래디언트 누적 횟수: {num_accumulation_rounds}")

    # 반복 당 처리할 패치 수를 계산합니다.
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    if hasattr(cfg.training.hp, "max_patch_per_gpu"):
        max_patch_per_gpu = cfg.training.hp.max_patch_per_gpu
        if max_patch_per_gpu // batch_size_per_gpu < 1:
            raise ValueError(
                f"max_patch_per_gpu({max_patch_per_gpu})는 batch_size_per_gpu({batch_size_per_gpu})보다 크거나 같아야 합니다."
            )
        max_patch_num_per_iter = min(
            patch_num, (max_patch_per_gpu // batch_size_per_gpu)
        )
        patch_iterations = (
            patch_num + max_patch_num_per_iter - 1
        ) // max_patch_num_per_iter
        patch_nums_iter = [
            min(max_patch_num_per_iter, patch_num - i * max_patch_num_per_iter)
            for i in range(patch_iterations)
        ]
        logger0.info(
            f"반복 당 최대 패치 수: {max_patch_num_per_iter}, 패치 반복 횟수: {patch_iterations}, 반복별 패치 수: {patch_nums_iter}"
        )
    else:
        patch_nums_iter = [patch_num]

    # 패치 그래디언트 누적을 설정합니다. (패치 기반 디퓨전 모델에만 해당)
    if cfg.model.name in {
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    }:
        if len(patch_nums_iter) > 1:
            if not patching:
                logger0.info(
                    "패칭이 비활성화되어 패치 그래디언트 누적이 자동으로 비활성화됩니다."
                )
                use_patch_grad_acc = False
            else:
                use_patch_grad_acc = True
        else:
            use_patch_grad_acc = False
    # 패치 기반이 아닌 모델의 경우 자동으로 비활성화합니다.
    else:
        logger0.info(
            "패치 기반이 아닌 모델을 학습하므로 패치 그래디언트 누적이 자동으로 비활성화됩니다."
        )
        use_patch_grad_acc = None

    # 손실 함수를 인스턴스화합니다.
    if cfg.model.name in (
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        loss_fn = ResidualLoss(
            regression_net=regression_net,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression" or cfg.model.name == "lt_aware_regression":
        loss_fn = RegressionLoss()
    elif cfg.model.name == "lt_aware_ce_regression":
        loss_fn = RegressionLossCE(prob_channels=prob_channels)

    # 옵티마이저를 인스턴스화합니다. (Adam 사용)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True, # CUDA 커널 퓨전을 통해 성능 향상
    )

    # 후속 작업의 소요 시간을 측정하기 위해 현재 시간을 기록합니다.
    start_time = time.time()

    # 옵티마이저 체크포인트가 있으면 불러옵니다.
    if dist.world_size > 1:
        torch.distributed.barrier() # 모든 프로세스가 동기화될 때까지 대기
    try:
        load_checkpoint(
            path=checkpoint_dir,
            optimizer=optimizer,
            device=dist.device,
        )
    except Exception:
        pass

    ############################################################################
    #                            메인 학습 루프                                  #
    ############################################################################

    logger0.info(f"총 {cfg.training.hp.training_duration}개의 이미지에 대해 학습을 시작합니다...")
    done = False

    # 주기적 작업 이후의 평균 손실을 모니터링하기 위한 변수 초기화
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    # 프로파일러 활성화
    with cuda_profiler():
        with profiler_emit_nvtx():

            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                # 특정 시점에서 프로파일러를 시작하고 중지합니다. (디버깅/분석용)
                if cur_nimg - start_nimg == 24 * cfg.training.hp.total_batch_size:
                    logger0.info(f"{cur_nimg}에서 프로파일러를 시작합니다.")
                    cuda_profiler_start()

                if cur_nimg - start_nimg == 25 * cfg.training.hp.total_batch_size:
                    logger0.info(f"{cur_nimg}에서 프로파일러를 중지합니다.")
                    cuda_profiler_stop()

                with nvtx.annotate("Training iteration", color="green"):
                    # 그래디언트를 계산하고 누적합니다.
                    optimizer.zero_grad(set_to_none=True) # 메모리 효율을 위해 그래디언트를 None으로 설정
                    loss_accum = 0
                    for n_i in range(num_accumulation_rounds):
                        with nvtx.annotate(
                            f"accumulation round {n_i}", color="Magenta"
                        ):
                            with nvtx.annotate("loading data", color="green"):
                                # 데이터로더에서 다음 배치를 가져옵니다.
                                img_clean, img_lr, *lead_time_label = next(
                                    dataset_iterator
                                )
                                # Apex GroupNorm 사용 시 메모리 포맷을 변경합니다.
                                if use_apex_gn:
                                    img_clean = img_clean.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                    img_lr = img_lr.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                else:
                                    img_clean = (
                                        img_clean.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                                    img_lr = (
                                        img_lr.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                            # 손실 함수에 전달할 인자들을 설정합니다.
                            loss_fn_kwargs = {
                                "net": model,
                                "img_clean": img_clean,
                                "img_lr": img_lr,
                                "augment_pipe": None,
                            }
                            if use_patch_grad_acc is not None:
                                loss_fn_kwargs[
                                    "use_patch_grad_acc"
                                ] = use_patch_grad_acc

                            # 리드 타임 레이블이 있는 경우 추가합니다.
                            if lead_time_label:
                                lead_time_label = (
                                    lead_time_label[0].to(dist.device).contiguous()
                                )
                                loss_fn_kwargs.update(
                                    {"lead_time_label": lead_time_label}
                                )
                            else:
                                lead_time_label = None
                            if use_patch_grad_acc:
                                loss_fn.y_mean = None

                            # 패치 반복
                            for patch_num_per_iter in patch_nums_iter:
                                if patching is not None:
                                    patching.set_patch_num(patch_num_per_iter)
                                    loss_fn_kwargs.update({"patching": patching})
                                
                                # 순전파(forward pass) 및 손실 계산
                                with nvtx.annotate(f"loss forward", color="green"):
                                    with torch.autocast(
                                        "cuda", dtype=amp_dtype, enabled=enable_amp
                                    ): # AMP 컨텍스트
                                        loss = loss_fn(**loss_fn_kwargs)

                                # 손실을 정규화하고 누적합니다.
                                loss = loss.sum() / batch_size_per_gpu
                                loss_accum += (
                                    loss
                                    / num_accumulation_rounds
                                    / len(patch_nums_iter)
                                )
                                
                                # 역전파(backward pass)
                                with nvtx.annotate(f"loss backward", color="yellow"):
                                    loss.backward()

                    # 모든 프로세스에서 계산된 손실을 집계합니다.
                    with nvtx.annotate(f"loss aggregate", color="green"):
                        loss_sum = torch.tensor([loss_accum], device=dist.device)
                        if dist.world_size > 1:
                            torch.distributed.barrier()
                            torch.distributed.all_reduce(
                                loss_sum, op=torch.distributed.ReduceOp.SUM
                            )
                        average_loss = (loss_sum / dist.world_size).cpu().item()

                    # 이동 평균 손실을 업데이트합니다.
                    average_loss_running_mean += (
                        average_loss - average_loss_running_mean
                    ) / n_average_loss_running_mean
                    n_average_loss_running_mean += 1

                    # 랭크 0에서 TensorBoard에 손실 값을 기록합니다.
                    if dist.rank == 0:
                        writer.add_scalar("training_loss", average_loss, cur_nimg)
                        writer.add_scalar(
                            "training_loss_running_mean",
                            average_loss_running_mean,
                            cur_nimg,
                        )

                    # 주기적인 작업(예: 로그 출력)을 수행할 시간인지 확인합니다.
                    ptt = is_time_for_periodic_task(
                        cur_nimg,
                        cfg.training.io.print_progress_freq,
                        done,
                        cfg.training.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    )
                    if ptt:
                        # 이동 평균 손실을 리셋합니다.
                        average_loss_running_mean = 0
                        n_average_loss_running_mean = 1

                    # 가중치를 업데이트합니다.
                    with nvtx.annotate("update weights", color="blue"):
                        lr_rampup = (
                            cfg.training.hp.lr_rampup
                        )  # 학습률을 점진적으로 증가시키는 기간
                        for g in optimizer.param_groups:
                            if lr_rampup > 0:
                                g["lr"] = cfg.training.hp.lr * min(
                                    cur_nimg / lr_rampup, 1
                                )
                            # 특정 시점 이후 학습률을 감소시킵니다 (decay).
                            if cur_nimg >= lr_rampup:
                                g["lr"] *= cfg.training.hp.lr_decay ** (
                                    (cur_nimg - lr_rampup)
                                    // cfg.training.hp.lr_decay_rate
                                )
                            current_lr = g["lr"]
                            if dist.rank == 0:
                                writer.add_scalar("learning_rate", current_lr, cur_nimg)
                        
                        # 그래디언트 클리핑을 수행합니다.
                        handle_and_clip_gradients(
                            model,
                            grad_clip_threshold=cfg.training.hp.grad_clip_threshold,
                        )
                    
                    # 옵티마이저 스텝을 실행하여 모델 파라미터를 업데이트합니다.
                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()

                    # 처리된 이미지 수를 업데이트하고 종료 조건을 확인합니다.
                    cur_nimg += cfg.training.hp.total_batch_size
                    done = cur_nimg >= cfg.training.hp.training_duration

                with nvtx.annotate("validation", color="red"):
                    # 검증(Validation)
                    if validation_dataset_iterator is not None:
                        valid_loss_accum = 0
                        # 주기적으로 검증을 수행할 시간인지 확인합니다.
                        if is_time_for_periodic_task(
                            cur_nimg,
                            cfg.training.io.validation_freq,
                            done,
                            cfg.training.hp.total_batch_size,
                            dist.rank,
                        ):
                            with torch.no_grad(): # 그래디언트 계산 비활성화
                                for _ in range(cfg.training.io.validation_steps):
                                    (
                                        img_clean_valid,
                                        img_lr_valid,
                                        *lead_time_label_valid,
                                    ) = next(validation_dataset_iterator)

                                    if use_apex_gn:
                                        img_clean_valid = img_clean_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)
                                        img_lr_valid = img_lr_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)
                                    else:
                                        img_clean_valid = (
                                            img_clean_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )
                                        img_lr_valid = (
                                            img_lr_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )

                                    loss_valid_kwargs = {
                                        "net": model,
                                        "img_clean": img_clean_valid,
                                        "img_lr": img_lr_valid,
                                        "augment_pipe": None,
                                        "use_patch_grad_acc": use_patch_grad_acc,
                                    }
                                    if use_patch_grad_acc is not None:
                                        loss_valid_kwargs[
                                            "use_patch_grad_acc"
                                        ] = use_patch_grad_acc
                                    if lead_time_label_valid:
                                        lead_time_label_valid = (
                                            lead_time_label_valid[0]
                                            .to(dist.device)
                                            .contiguous()
                                        )
                                        loss_valid_kwargs.update(
                                            {"lead_time_label": lead_time_label_valid}
                                        )
                                    if use_patch_grad_acc:
                                        loss_fn.y_mean = None

                                    for patch_num_per_iter in patch_nums_iter:
                                        if patching is not None:
                                            patching.set_patch_num(patch_num_per_iter)
                                            loss_fn_kwargs.update(
                                                {"patching": patching}
                                            )
                                        with torch.autocast(
                                            "cuda", dtype=amp_dtype, enabled=enable_amp
                                        ):
                                            loss_valid = loss_fn(**loss_valid_kwargs)

                                        loss_valid = (
                                            (loss_valid.sum() / batch_size_per_gpu)
                                            .cpu()
                                            .item()
                                        )
                                        valid_loss_accum += (
                                            loss_valid
                                            / cfg.training.io.validation_steps
                                        )
                                # 모든 프로세스에서 검증 손실을 집계합니다.
                                valid_loss_sum = torch.tensor(
                                    [valid_loss_accum], device=dist.device
                                )
                                if dist.world_size > 1:
                                    torch.distributed.barrier()
                                    torch.distributed.all_reduce(
                                        valid_loss_sum,
                                        op=torch.distributed.ReduceOp.SUM,
                                    )
                                average_valid_loss = valid_loss_sum / dist.world_size
                                if dist.rank == 0:
                                    writer.add_scalar(
                                        "validation_loss", average_valid_loss, cur_nimg
                                    )

                # 주기적으로 학습 진행 상황을 출력합니다.
                if is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.print_progress_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    tick_end_time = time.time()
                    fields = []
                    fields += [f"samples {cur_nimg:<9.1f}"]
                    fields += [f"training_loss {average_loss:<7.2f}"]
                    fields += [
                        f"training_loss_running_mean {average_loss_running_mean:<7.2f}"
                    ]
                    fields += [f"learning_rate {current_lr:<7.8f}"]
                    fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
                    fields += [
                        f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"
                    ]
                    fields += [
                        f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"
                    ]
                    fields += [
                        f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
                    ]
                    if torch.cuda.is_available():
                        fields += [
                            f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
                        ]
                        fields += [
                            f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
                        ]
                        torch.cuda.reset_peak_memory_stats()
                    logger0.info(" ".join(fields))

                # 주기적으로 체크포인트를 저장합니다.
                if dist.world_size > 1:
                    torch.distributed.barrier()
                if is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.save_checkpoint_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    save_checkpoint(
                        path=checkpoint_dir,
                        models=model,
                        optimizer=optimizer,
                        epoch=cur_nimg,
                    )

            # 지정된 수의 최근 체크포인트만 남기고 오래된 체크포인트를 삭제합니다.
            if cfg.training.io.save_n_recent_checkpoints > 0:
                for suffix in [".mdlus", ".pt"]:
                    ckpts = checkpoint_list(checkpoint_dir, suffix=suffix)
                    while len(ckpts) > cfg.training.io.save_n_recent_checkpoints:
                        os.remove(os.path.join(checkpoint_dir, ckpts[0]))
                        ckpts = ckpts[1:]

    # 학습 완료
    logger0.info("학습이 완료되었습니다.")


if __name__ == "__main__":
    main()
