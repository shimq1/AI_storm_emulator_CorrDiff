import xarray as xr
import matplotlib.pyplot as plt
import os

# --- ⚙️ 설정 변수 ---
# 1. 분석할 .nc 파일 경로
nc_file_path = '/scratch/x3108a06/generate_output/corrdiff_autoregressive_output.nc'

# 2. 시각화할 변수 이름
variable_to_plot = 'PW'

# 3. 결과물 저장 폴더
output_dir = 'visualization_results'

file_name = 'unet_regressive'
# ---------------------

# 결과물 저장 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)
print(f"결과물은 '{output_dir}' 폴더에 저장됩니다.")

try:
    # 1. 데이터 로딩 (파일을 한 번만 엽니다)
    ds_truth = xr.open_dataset(nc_file_path, group='truth', decode_times=False)
    ds_pred = xr.open_dataset(nc_file_path, group='prediction', decode_times=False)
    ds_input = xr.open_dataset(nc_file_path, group='input', decode_times=False)
    ds_coords = xr.open_dataset(nc_file_path, decode_times=False)
    lat = ds_coords['lat'].values
    lon = ds_coords['lon'].values

    # 파일에 있는 전체 타임스텝 개수 확인
    num_timesteps = ds_truth.sizes['time']
    print(f"총 {num_timesteps}개의 타임스텝에 대한 시각화를 시작합니다.")

    # 2. 모든 타임스텝에 대해 반복 실행
    for t_idx in range(num_timesteps):
        print(f"  - 타임스텝 {t_idx} 처리 중...")

        # 현재 타임스텝(t_idx)의 데이터 선택
        truth_data = ds_truth[variable_to_plot].isel(time=t_idx)
        input_data = ds_input[variable_to_plot].isel(time=t_idx)
        pred_data_all_ensembles = ds_pred[variable_to_plot].isel(time=t_idx)
        pred_data_1st_member = ds_pred[variable_to_plot].isel(time=t_idx, ensemble=0)

        # 앙상블 통계 계산
        pred_mean = pred_data_all_ensembles.mean(dim='ensemble', keep_attrs=True)
        pred_std = pred_data_all_ensembles.std(dim='ensemble', keep_attrs=True)
        difference = pred_mean - truth_data

        # 시각화
        fig, axes = plt.subplots(1, 5, figsize=(28, 5), constrained_layout=True)
        axes = axes.flatten()

        # min_val = min(truth_data.min(), pred_mean.min())
        # max_val = max(truth_data.max(), pred_mean.max())
        min_val = -3
        max_val = 8

        # 플롯 0: Input
        im0 = axes[0].pcolormesh(lon, lat, input_data, vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        axes[0].set_title('Input')
        fig.colorbar(im0, ax=axes[0], orientation='vertical')
        
        # 플롯 1: Ground Truth
        im1 = axes[1].pcolormesh(lon, lat, truth_data, vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        axes[1].set_title('Ground Truth')
        fig.colorbar(im1, ax=axes[1], orientation='vertical')

        # 플롯 2: Prediction (Ensemble Mean)
        im2 = axes[2].pcolormesh(lon, lat, pred_mean, vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        # axes[2].set_title('Diffusion Ensemble mean Prediction')
        axes[2].set_title('UNet Prediction')
        fig.colorbar(im2, ax=axes[2], orientation='vertical')

        # 플롯 3: Prediction (Ensemble 1st member)
        im3 = axes[3].pcolormesh(lon, lat, pred_data_1st_member, vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        axes[3].set_title('Diffusion 1st member Prediction')
        fig.colorbar(im3, ax=axes[3], orientation='vertical')

        # 플롯 4: Uncertainty
        im4 = axes[4].pcolormesh(lon, lat, pred_std, cmap='magma', shading='auto')
        axes[4].set_title('Uncertainty (Ensemble Std Dev)')
        fig.colorbar(im4, ax=axes[4], orientation='vertical', label='Standard Deviation')

        # 전체 제목 설정
        fig.suptitle(f'Model Output for "{variable_to_plot}" at Time Index {t_idx}', fontsize=16)

        # 각 타임스텝별로 다른 이름의 파일로 저장
        output_filename = os.path.join(output_dir, f'{file_name}_{variable_to_plot}_t{t_idx}.png')
        plt.savefig(output_filename, dpi=300)
        
        # 메모리 관리를 위해 그림 닫기
        plt.close(fig)

    print(f"✅ 모든 시각화가 완료되었습니다.")

except FileNotFoundError:
    print(f"❌ Error: '{nc_file_path}' 파일을 찾을 수 없습니다.")