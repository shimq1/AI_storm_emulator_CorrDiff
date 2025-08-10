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

import sys

sys.path.insert(0, "/data03/SAM/physicsnemo")
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from utils import (
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

forecast_time_min = 6 #! <--- 예측 시간 (분) 최종적으로 입력을 받을 수 있게 바꿔야 할 듯

@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """
    설정 파일(cfg)에 따라 모델을 로드하고, 데이터셋으로부터 입력을 받아
    이미지(결과장)를 생성하여 NetCDF 파일로 저장하는 메인 스크립트.
    Autoregressive 모드를 지원.
    """
    # --- 2. 초기 설정 및 분산 처리 초기화 ---
    is_autoregressive = cfg.generation.get("autoregressive", {}).get("enabled", False)
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    logger = PythonLogger("generate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # 앙상블 및 배치 크기 설정
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    if dist.world_size > 1:
        torch.distributed.barrier()

    # --- 3. 데이터 로딩을 위한 시간 정보 파싱 ---
    if getattr(cfg.generation, "times_range", None) and getattr(cfg.generation, "times", None):
        raise ValueError("times_range와 times 둘 중 하나만 지정해야 합니다.")
    
    # Use times even in case of autoregressive generation
    times = get_time_from_range(cfg.generation.times_range) if cfg.generation.times_range else cfg.generation.times

    logger0.info(f"Using times: {times}")
    # --- 4. 데이터셋 및 모델 관련 설정 ---
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")
    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
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
        """한 스텝의 예측을 수행하고, 최종 결과와 회귀 결과를 반환하는 함수"""
        with nvtx.annotate("generate_fn", color="green"):
            img_lr = current_image_lr.to(device=device).to(torch.float32).to(memory_format=torch.channels_last)
            
            image_reg, image_res = None, None # 결과 변수 초기화

            # 회귀 모델 실행
            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg, img_lr=img_lr,
                        latents_shape=(cfg.generation.seed_batch_size, img_out_channels, img_shape[0], img_shape[1]),
                        lead_time_label=lead_time_label,
                    )
            # 확산 모델 실행
            if net_res:
                with nvtx.annotate("diffusion model", color="purple"):
                    mean_hr = image_reg[0:1] if cfg.generation.hr_mean_conditioning else None
                    image_res = diffusion_step(
                        net=net_res, sampler_fn=sampler_fn, img_shape=img_shape, img_out_channels=img_out_channels,
                        rank_batches=rank_batches, img_lr=img_lr.expand(cfg.generation.seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                        rank=dist.rank, device=device, mean_hr=mean_hr, lead_time_label=lead_time_label,
                    )

            # 최종 출력 결정
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else: # "all" 모드
                image_out = image_reg + image_res

            # 💡 수정된 부분: image_reg도 함께 반환 (all 모드일 때만)
            image_reg_to_return = image_reg if cfg.generation.inference_mode == "all" else None
            
            # 다중 GPU 처리 로직 (복잡성으로 인해 여기서는 생략, 단일 GPU 가정)
            # TODO: 다중 GPU 환경에서 튜플을 gather 하려면 별도 처리 필요
            return image_out, image_reg_to_return

    # --- 메인 로직 분기 (Autoregressive 모드 vs 일반 모드) ---
    if is_autoregressive:
        # --------------------------------------------------
        # --- AUTOREGRESSIVE 예측 로직 ---
        # --------------------------------------------------
        num_auto_steps = cfg.generation.autoregressive.num_auto_steps
        logger0.info(f"Starting autoregressive forecast for {num_auto_steps} steps.")

        output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
        logger0.info(f"Generating images, saving results to {output_path}...")

        # DataLoader according to designated times array
        data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True)
        time_index = -1

        times_from_dataset = dataset.time()

        predictions_total_list = []
        inputs_total_list = []
        regressions_total_list = []
        truths_total_list = []
        for index, (image_tar, initial_lr, *lead_time_label) in enumerate(iter(data_loader)):
            if lead_time_label:
                lead_time_label = lead_time_label[0].to(dist.device).contiguous()
            else:
                lead_time_label = None

            # --- 7a. 정답(Truth) 데이터 미리 로드 (수정된 핵심 로직) ---
            # 예측할 모든 시간 단계에 대한 실제 정답 데이터를 미리 불러옴
            logger0.info("Loading ground truth data for all forecast steps...")
            start_time = pd.to_datetime(times[index])
            time_delta = pd.Timedelta(minutes=forecast_time_min) # 30분 간격
            # 예측에 필요한 모든 시간 목록 생성
            truth_times = [start_time + i * time_delta for i in range(num_auto_steps)]
            truth_times_str = [t.strftime("%Y-%m-%dT%H:%M:%S") for t in truth_times]

            # 정답 데이터를 로드하기 위한 별도의 데이터셋과 데이터로더 생성
            truth_dataset, truth_sampler = get_dataset_and_sampler(
                dataset_cfg=dataset_cfg, times=truth_times_str, has_lead_time=has_lead_time
            )
            truth_data_loader = torch.utils.data.DataLoader(dataset=truth_dataset, sampler=truth_sampler, batch_size=1, pin_memory=True)
            logger0.info(f"Loaded ground truth dataset with {len(truth_dataset)} samples for times: {truth_times_str}")
            # 모든 정답 데이터를 순회하며 리스트에 저장
            truths_list = []
            for _, image_tar, *_ in iter(truth_data_loader):
                truths_list.append(image_tar.cpu())
            # 리스트를 하나의 텐서로 결합
            truths_single_time = torch.cat(truths_list, dim=0)
            logger0.info(f"Loaded {len(truths_single_time)} ground truth steps.")


            # --- 7b. 연쇄 예측 수행 ---
            current_lr = initial_lr
            
            predictions_list = []
            regressions_list = []

            # 연쇄 예측 루프
            for step in range(num_auto_steps):
                logger0.info(f"  - Autoregressive step {step + 1}/{num_auto_steps}...")
                # 💡 수정: 이제 2개의 값을 반환받음
                predicted_out, predicted_reg = generate_fn(current_lr, lead_time_label)

                if dist.rank == 0:
                    predictions_list.append(predicted_out.cpu())
                    if predicted_reg is not None:
                        regressions_list.append(predicted_reg.cpu())

                        
                # Which output to use as input for the next step (Default: ENS_mean = Diffusion ens mean, regression = U-Net, diffusion = Diffusion 1st member)
                out2input = getattr(cfg.generation.autoregressive, "out2input", "ENS_mean")
                if out2input == "regression":
                    current_lr = predicted_reg
                elif out2input == "diffusion":
                    current_lr = predicted_out[0].unsqueeze(0)
                elif out2input == "ENS_mean":
                    current_lr = predicted_out.mean(dim=0, keepdim=True)

            predictions_total_list.append(torch.stack(predictions_list))
            inputs_total_list.append(initial_lr.squeeze(0).cpu())
            if predicted_reg is not None:
                regressions_total_list.append(torch.stack(regressions_list))
            truths_total_list.append(truths_single_time)

        # --- 7c. 결과 저장 ---
        # --------------------------------------------------
        # Data Dimensions
        # Input: (Time, X, Y, Channel)
        # Truth: (Time, Step, X, Y, Channel)
        # Unet : (Time, Step, X, Y, Channel)
        # Pred : (Ens , Time, Step, X, Y, Channel)
        # --------------------------------------------------
        if dist.rank == 0:
            output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
            logger0.info(f"Autoregressive forecast finished. Saving results to {output_path}...")
            
            all_predictions = torch.stack(predictions_total_list, dim=0)
            all_inputs = torch.stack(inputs_total_list, dim=0)
            all_truths = torch.stack(truths_total_list, dim=0)
            logger0.info(f"Total predictions shape: {all_predictions.shape}, inputs shape: {all_inputs.shape}, truths shape: {all_truths.shape}")

            with nc.Dataset(output_path, "w") as f:
                f.cfg = str(cfg)
                writer = NetCDFWriter(
                    f, lat=dataset.latitude(), lon=dataset.longitude(),
                    input_channels=dataset.input_channels(), output_channels=dataset.output_channels(),
                    has_lead_time=has_lead_time, autoregressive_opt=is_autoregressive  # ✅ _opt 맞추기
                )
                
                # Prediction 데이터 저장 (이전과 동일)
                output_channels_meta = dataset.output_channels()
                prediction_to_save = all_predictions.permute(2, 0, 1, 4, 5, 3).numpy() # (time, step, ens, channel, x, y) -> (ensemble, time, step, x ,y ,channel)
                num_ensembles = prediction_to_save.shape[0]
                num_times = prediction_to_save.shape[1]
                num_auto_steps = prediction_to_save.shape[2]

                print(f"num_auto_steps = {num_auto_steps}")
                for ens_idx in range(num_ensembles):
                    for time_idx in range(num_times):
                        for step_idx in range(num_auto_steps):
                            for chan_idx, chan_meta in enumerate(output_channels_meta):
                                data_slice = prediction_to_save[ens_idx, time_idx, step_idx, :, :, chan_idx]
                                writer.write_prediction(val=data_slice, time_index=time_idx, ensemble_index=ens_idx, step_index=step_idx, channel_name=chan_meta.name)

                # Input 데이터 저장 (이전과 동일)
                input_channels_meta = dataset.input_channels()
                input_to_save = all_inputs.permute(0, 2, 3, 1).numpy() # (time, channel, x, y) -> (time, x ,y ,channel)
                for time_idx in range(num_times):
                    for chan_idx, chan_meta in enumerate(input_channels_meta):
                        data_slice = input_to_save[time_idx, :, :, chan_idx]
                        writer.write_input(val=data_slice, time_index=time_idx, channel_name=chan_meta.name)

                # Truth 데이터 저장 (수정된 로직)
                truth_to_save = all_truths.permute(0, 1, 3, 4, 2).numpy() # (time, step, channel, x, y)) -> (time, step, x ,y ,channel)
                for time_idx in range(num_times):
                    for step_idx in range(num_auto_steps):
                        for chan_idx, chan_meta in enumerate(output_channels_meta):
                            data_slice = truth_to_save[time_idx, step_idx, :, :, chan_idx]
                            writer.write_truth(val=data_slice, time_index=time_idx, step_index=step_idx, channel_name=chan_meta.name)

                # 💡 추가: Regression 그룹에 결과 저장
                if cfg.generation.inference_mode == 'all' and regressions_total_list:
                    logger0.info("Saving regression-only output to 'regression' group...")
                    all_regressions = torch.stack(regressions_total_list, dim=0)
                    logger0.info(f"Total regression shape: {all_regressions.shape}")
                    reg_to_save = all_regressions.permute(2, 0, 1, 4, 5, 3).squeeze(0).numpy()
                    
                    # NetCDF4 라이브러리를 직접 사용하여 그룹과 변수 생성
                    reg_group = f.createGroup('regression')
                    reg_group.createDimension('time', num_times)
                    reg_group.createDimension('step', num_auto_steps)
                    reg_group.createDimension('y', img_shape[0])
                    reg_group.createDimension('x', img_shape[1])

                    output_channels_meta = dataset.output_channels()
                    for chan_idx, chan_meta in enumerate(output_channels_meta):
                        var = reg_group.createVariable(chan_meta.name, 'f4', ('time', 'step', 'y', 'x'))
                        var[:] = reg_to_save[:, :, :, :, chan_idx]


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
