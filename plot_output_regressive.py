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

# 4. 저장할 최종 이미지 파일 이름
output_filename = 'unet_autoregressive_forecast_comparison.png'
# ---------------------

# 결과물 저장 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)
print(f"결과물은 '{output_dir}' 폴더에 저장됩니다.")

try:
    # --- 1. 데이터 로딩 ---
    ds_truth = xr.open_dataset(nc_file_path, group='truth', decode_times=False)
    ds_pred = xr.open_dataset(nc_file_path, group='prediction', decode_times=False)
    ds_coords = xr.open_dataset(nc_file_path, decode_times=False)
    
    lat = ds_coords['lat'].values
    lon = ds_coords['lon'].values

    # 전체 타임스텝 개수 확인
    num_timesteps = ds_pred.sizes['time']
    print(f"총 {num_timesteps}개의 타임스텝에 대한 시각화를 시작합니다.")

    # --- 2. 시각화할 전체 데이터 준비 ---
    # 모든 타임스텝의 truth와 prediction 데이터를 미리 불러옵니다.
    all_truth_data = ds_truth[variable_to_plot].isel(time=range(num_timesteps))
    all_pred_data_1st_member = ds_pred[variable_to_plot].isel(time=range(num_timesteps), ensemble=0)
    
    # --- 3. 전체 플롯에 대한 공통 컬러맵 범위 계산 ---
    # 모든 데이터에 걸쳐 최소/최대값을 계산하여 색상 범위를 통일합니다.
    min_val = min(all_truth_data.min(), all_pred_data_1st_member.min())
    max_val = max(all_truth_data.max(), all_pred_data_1st_member.max())
    print(f"전체 데이터의 값 범위 (min/max): {min_val.item():.2f} / {max_val.item():.2f}")
    
    # --- 4. 💡 하나의 큰 Figure 생성 ---
    # 2행 num_timesteps열의 그리드를 만듭니다. figsize로 전체 그림 크기를 조정합니다.
    fig, axes = plt.subplots(
        nrows=2, 
        ncols=num_timesteps, 
        figsize=(4 * num_timesteps, 8), 
        constrained_layout=True
    )
    
    # --- 5. 💡 각 subplot에 데이터 채우기 ---
    # 각 열(timestep)을 순회하며 그림을 그립니다.
    for t_idx in range(num_timesteps):
        # 1행: Prediction (1st member)
        ax_pred = axes[0, t_idx]
        im = ax_pred.pcolormesh(lon, lat, all_pred_data_1st_member.isel(time=t_idx), 
                                vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        ax_pred.set_xticks([]) # x축 눈금 제거
        ax_pred.set_yticks([]) # y축 눈금 제거
        ax_pred.set_title(f't = {t_idx}') # 각 열의 제목 (시간 인덱스)
        
        # 2행: Ground Truth
        ax_truth = axes[1, t_idx]
        ax_truth.pcolormesh(lon, lat, all_truth_data.isel(time=t_idx), 
                            vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        ax_truth.set_xticks([]) # x축 눈금 제거
        ax_truth.set_yticks([]) # y축 눈금 제거

    # --- 6. 💡 행(Row) 제목 설정 ---
    # 첫 번째 열에만 각 행의 제목을 표시합니다.
    axes[0, 0].set_ylabel('Prediction\n(UNet)', fontsize=14, labelpad=10)
    axes[1, 0].set_ylabel('Ground Truth', fontsize=14, labelpad=10)
    
    # --- 7. 💡 전체 Figure에 대한 공통 설정 ---
    fig.suptitle(f'Autoregressive Forecast vs. Ground Truth for "{variable_to_plot}"', fontsize=16)

    # 그림 오른쪽에 공통 컬러바 추가
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=f'{variable_to_plot} units')

    # 최종 이미지 파일로 저장
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"✅ 모든 타임스텝을 포함한 그림이 '{save_path}' 파일로 저장되었습니다.")
    
    plt.close(fig)

except FileNotFoundError:
    print(f"❌ Error: '{nc_file_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ An error occurred: {e}")