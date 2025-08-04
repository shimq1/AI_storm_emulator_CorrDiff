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

import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import torch
import torch._dynamo
from torch.distributed import gather
import numpy as np
import nvtx
import netCDF4 as nc
import pandas as pd

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
    regression_step,
    diffusion_step,
)

from helpers.generate_helpers import (
    get_dataset_and_sampler,
    save_images,
)
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """
    설정 파일(cfg)에 따라 모델을 로드하고, 데이터셋으로부터 입력을 받아
    이미지(결과장)를 생성하여 NetCDF 파일로 저장하는 메인 스크립트.
    Autoregressive 모드를 지원.
    """
    # --- 2. 초기 설정 및 분산 처리 초기화 ---
    # Autoregressive 모드 활성화 여부를 설정 파일에서 확인
    is_autoregressive = cfg.generation.get("autoregressive", {}).get("enabled", False)

    # 다중 GPU/노드를 위한 분산 처리 관리자 초기화
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # 로깅 설정 (실행 과정 기록용)
    logger = PythonLogger("generate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # 앙상블 및 배치 크기 설정 (여러 GPU에 작업을 분배하기 위함)
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # 모든 GPU 프로세스 동기화
    if dist.world_size > 1:
        torch.distributed.barrier()

    # --- 3. 데이터 로딩을 위한 시간 정보 파싱 ---
    if cfg.generation.times_range and cfg.generation.times:
        raise ValueError("times_range와 times 둘 중 하나만 지정해야 합니다.")
    
    # Autoregressive 모드일 경우, 예측 시작 시간만 가져옴
    if is_autoregressive:
        logger0.info("Autoregressive mode enabled. Using only the first timestep as initial input.")
        if cfg.generation.times_range:
            start_time_str = cfg.generation.times_range['start']
            times_for_initial_load = get_time_from_range({'start': start_time_str, 'end': start_time_str})
        else:
            start_time_str = cfg.generation.times[0]
            times_for_initial_load = [start_time_str]
    else: # 일반 모드일 경우, 모든 요청된 시간을 사용
        if cfg.generation.times_range:
            times = get_time_from_range(cfg.generation.times_range)
        else:
            times = cfg.generation.times

    # --- 4. 데이터셋 및 모델 관련 설정 ---
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type) # 커스텀 데이터셋 등록
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    has_lead_time = cfg.generation.get("has_lead_time", False)
    
    # 데이터셋 객체와 샘플러 생성 (Autoregressive 모드에서는 초기 데이터 로딩용)
    dataset_times = times_for_initial_load if is_autoregressive else times
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=dataset_times, has_lead_time=has_lead_time
    )
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Parse the patch shape
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

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(to_absolute_path(res_ckpt_filename))
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True
        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(to_absolute_path(reg_ckpt_filename))
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res, mode="reduce-overhead")

    # Partially instantiate the sampler based on the configs
    if cfg.sampler.type == "deterministic":
        if cfg.generation.hr_mean_conditioning:
            raise NotImplementedError(
                "High-res mean conditioning is not yet implemented for the deterministic sampler"
            )
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            solver=cfg.sampler.solver,
        )
    elif cfg.sampler.type == "stochastic":
        sampler_fn = partial(stochastic_sampler, patching=patching)
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")

    # --- AUTOREGRESSIVE: generate_fn을 수정하여 image_lr을 인자로 받도록 함 ---
    def generate_fn(current_image_lr, lead_time_label):
        with nvtx.annotate("generate_fn", color="green"):
            img_lr = current_image_lr.to(device=device).to(torch.float32).to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=(
                            cfg.generation.seed_batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),
                        lead_time_label=lead_time_label,
                    )
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
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
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg + image_res

            # Gather tensors on rank 0 (분산 처리)
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [torch.zeros_like(image_out, dtype=image_out.dtype, device=image_out.device) for _ in range(dist.world_size)]
                else:
                    gathered_tensors = None
                torch.distributed.barrier()
                gather(image_out, gather_list=gathered_tensors if dist.rank == 0 else None, dst=0)
                return torch.cat(gathered_tensors) if dist.rank == 0 else None
            else:
                return image_out

    # --- 메인 로직 분기 (Autoregressive 모드 vs 일반 모드) ---
    if is_autoregressive:
        # --------------------------------------------------
        # --- AUTOREGRESSIVE 예측 로직 ---
        # --------------------------------------------------
        num_steps = cfg.generation.autoregressive.num_steps
        logger0.info(f"Starting autoregressive forecast for {num_steps} steps.")

        # --- 7a. 정답(Truth) 데이터 미리 로드 (수정된 핵심 로직) ---
        # 예측할 모든 시간 단계에 대한 실제 정답 데이터를 미리 불러옴
        logger0.info("Loading ground truth data for all forecast steps...")
        start_time = pd.to_datetime(start_time_str)
        time_delta = pd.Timedelta(minutes=30) # 30분 간격
        # 예측에 필요한 모든 시간 목록 생성
        truth_times = [start_time + i * time_delta for i in range(num_steps)]
        truth_times_str = [t.strftime("%Y-%m-%dT%H:%M:%S") for t in truth_times]

        # 정답 데이터를 로드하기 위한 별도의 데이터셋과 데이터로더 생성
        truth_dataset, truth_sampler = get_dataset_and_sampler(
            dataset_cfg=dataset_cfg, times=truth_times_str, has_lead_time=has_lead_time
        )
        truth_data_loader = torch.utils.data.DataLoader(dataset=truth_dataset, sampler=truth_sampler, batch_size=1, pin_memory=True)
        
        # 모든 정답 데이터를 순회하며 리스트에 저장
        truths_list = []
        for _, image_tar, *_ in iter(truth_data_loader):
            truths_list.append(image_tar.cpu())
        # 리스트를 하나의 텐서로 결합
        all_truths = torch.cat(truths_list, dim=0)
        logger0.info(f"Loaded {len(all_truths)} ground truth steps.")


        # --- 7b. 연쇄 예측 수행 ---
        # 초기 입력 데이터 로드 (t=0)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True)
        try:
            _, initial_lr, *initial_lead_time = next(iter(data_loader))
        except StopIteration:
            logger0.error("Dataset is empty. Cannot get initial state for autoregressive forecast.")
            return

        if initial_lead_time:
            lead_time_label = initial_lead_time[0].to(dist.device).contiguous()
        else:
            lead_time_label = None

        current_lr = initial_lr
        
        predictions_list = []
        inputs_list = []

        # 연쇄 예측 루프
        for step in range(num_steps):
            logger0.info(f"  - Autoregressive step {step + 1}/{num_steps}...")
            predicted_out = generate_fn(current_lr, lead_time_label)

            if dist.rank == 0:
                predictions_list.append(predicted_out.cpu())
                inputs_list.append(current_lr.cpu())

            # 앙상블 평균을 다음 입력으로 사용
            current_lr = predicted_out.mean(dim=0, keepdim=True)

        # --- 7c. 결과 저장 ---
        if dist.rank == 0:
            output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
            logger0.info(f"Autoregressive forecast finished. Saving results to {output_path}...")
            
            all_predictions = torch.stack(predictions_list) 
            all_inputs = torch.stack(inputs_list) 
            
            with nc.Dataset(output_path, "w") as f:
                f.cfg = str(cfg)
                writer = NetCDFWriter(
                    f, lat=dataset.latitude(), lon=dataset.longitude(),
                    input_channels=dataset.input_channels(), output_channels=dataset.output_channels(),
                    has_lead_time=has_lead_time,
                )
                
                # Prediction 데이터 저장 (이전과 동일)
                output_channels_meta = dataset.output_channels()
                prediction_to_save = all_predictions.permute(1, 0, 3, 4, 2).numpy()
                num_ensembles = prediction_to_save.shape[0]
                num_times = prediction_to_save.shape[1]
                
                for ens_idx in range(num_ensembles):
                    for time_idx in range(num_times):
                        for chan_idx, chan_meta in enumerate(output_channels_meta):
                            data_slice = prediction_to_save[ens_idx, time_idx, :, :, chan_idx]
                            writer.write_prediction(val=data_slice, time_index=time_idx, ensemble_index=ens_idx, channel_name=chan_meta.name)

                # Input 데이터 저장 (이전과 동일)
                input_channels_meta = dataset.input_channels()
                input_to_save = all_inputs.squeeze(1).permute(0, 2, 3, 1).numpy()
                for time_idx in range(num_times):
                    for chan_idx, chan_meta in enumerate(input_channels_meta):
                        data_slice = input_to_save[time_idx, :, :, chan_idx]
                        writer.write_input(val=data_slice, time_index=time_idx, channel_name=chan_meta.name)

                # Truth 데이터 저장 (수정된 로직)
                truth_to_save = all_truths.permute(0, 2, 3, 1).numpy()
                for time_idx in range(num_steps):
                    for chan_idx, chan_meta in enumerate(output_channels_meta):
                        data_slice = truth_to_save[time_idx, :, :, chan_idx]
                        writer.write_truth(val=data_slice, time_index=time_idx, channel_name=chan_meta.name)


    else:
        # --------------------------------------------------
        # --- 기존의 단일 예측 로직 ---
        # --------------------------------------------------
        output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
        logger0.info(f"Generating images, saving results to {output_path}...")
        batch_size = 1
        warmup_steps = min(len(times) - 1, 2)
        if dist.rank == 0:
            f = nc.Dataset(output_path, "w")
            f.cfg = str(cfg)

        data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True)
        time_index = -1
        if dist.rank == 0:
            writer = NetCDFWriter(f, lat=dataset.latitude(), lon=dataset.longitude(), input_channels=dataset.input_channels(), output_channels=dataset.output_channels(), has_lead_time=has_lead_time)
            writer_executor = ThreadPoolExecutor(max_workers=cfg.generation.perf.num_writer_workers)
            writer_threads = []

        times_from_dataset = dataset.time()
        for index, (image_tar, image_lr, *lead_time_label) in enumerate(iter(data_loader)):
            time_index += 1
            if dist.rank == 0: logger0.info(f"starting index: {time_index}")

            if lead_time_label:
                lead_time_label = lead_time_label[0].to(dist.device).contiguous()
            else:
                lead_time_label = None

            # generate_fn은 current_image_lr, lead_time_label 인자를 받도록 수정되었으므로 맞춰서 호출
            image_out = generate_fn(image_lr, lead_time_label)
            
            if dist.rank == 0:
                batch_size = image_out.shape[0]
                writer_threads.append(
                    writer_executor.submit(
                        save_images, writer, dataset, list(times_from_dataset), 
                        image_out.cpu(), image_tar.cpu(), image_lr.cpu(), 
                        time_index, index, has_lead_time
                    )
                )
        
        if dist.rank == 0:
            for thread in list(writer_threads):
                thread.result()
                writer_threads.remove(thread)
            writer_executor.shutdown()
        
        if dist.rank == 0:
            f.close()
            
    logger0.info("Generation Completed.")

if __name__ == "__main__":
    main()
