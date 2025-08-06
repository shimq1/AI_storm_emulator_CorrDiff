import xarray as xr
import matplotlib.pyplot as plt
import properscoring as ps
import numpy as np
import os

# --- ⚙️ 설정 변수 ---
# 1. 분석할 .nc 파일 경로
nc_file_path = '/scratch/x3108a06/generate_output/corrdiff_autoregressive_output.nc'

# 2. 시각화할 변수 이름
variable_to_plot = 'PW'

# 3. 결과물 저장 폴더
output_dir = 'visualization_results'

# 4. 저장할 최종 이미지 파일 이름
output_filename = 'res_autoregressive_forecast_comparison_cropped.png'

# 5. 잘라낼 픽셀 수
CROP_SIZE = 16
# ---------------------

# 결과물 저장 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)
print(f"결과물은 '{output_dir}' 폴더에 저장됩니다.")

try:
    # --- 1. 데이터 로딩 ---
    ds_truth = xr.open_dataset(nc_file_path, group='truth', decode_times=False)
    ds_pred = xr.open_dataset(nc_file_path, group='prediction', decode_times=False)
    ds_res = xr.open_dataset(nc_file_path, group='regression', decode_times=False)
    ds_coords = xr.open_dataset(nc_file_path, decode_times=False)
    
    # --- 좌표축 데이터 로딩 및 잘라내기 ('y', 'x' 사용) ---
    y_full = ds_coords['y'].values
    x_full = ds_coords['x'].values
    y_cropped = y_full[CROP_SIZE:-CROP_SIZE]
    x_cropped = x_full[CROP_SIZE:-CROP_SIZE]

    height = len(y_cropped)
    width = len(x_cropped)
    print(f"✅ 원본 이미지 크기: {len(y_full)} x {len(x_full)}")
    print(f"✅ 잘라낸 후 이미지 크기(세로 x 가로): {height} x {width}")

    # 전체 타임스텝 개수 확인
    num_timesteps = ds_pred.sizes['time']
    print(f"총 {num_timesteps}개의 타임스텝에 대한 시각화를 시작합니다.")

    # --- 2. 시각화할 전체 데이터 준비 및 잘라내기 ---
    # isel과 slice를 사용하여 모든 데이터를 미리 잘라냅니다.
    slicer = {
        'y': slice(CROP_SIZE, -CROP_SIZE),
        'x': slice(CROP_SIZE, -CROP_SIZE)
    }
    
    all_truth_data = ds_truth[variable_to_plot].isel(time=range(num_timesteps), **slicer)
    all_pred_data_1st_member = ds_pred[variable_to_plot].isel(time=range(num_timesteps), ensemble=0, **slicer)
    all_pred_data_ensembles = ds_pred[variable_to_plot].isel(time=range(num_timesteps), **slicer)
    all_res_data = ds_res[variable_to_plot].isel(time=range(num_timesteps), ensemble=0, **slicer)
    
    # --- 3. 전체 플롯에 대한 공통 컬러맵 범위 계산 ---
    # '잘라낸' 데이터에 대해 최소/최대값을 계산합니다.
    min_val = min(all_truth_data.min(), all_pred_data_1st_member.min(), all_res_data.min())
    max_val = max(all_truth_data.max(), all_pred_data_1st_member.max(), all_res_data.max())
    print(f"전체 데이터의 값 범위 (min/max): {min_val.item():.2f} / {max_val.item():.2f}")
    
    # --- 4. 하나의 큰 Figure 생성 ---
    fig, axes = plt.subplots(
        nrows=3, 
        ncols=num_timesteps, 
        figsize=(4 * num_timesteps, 12), 
        constrained_layout=True
    )

    mse_scores = []
    crps_scores = []
    
    # --- 5. '잘라낸' 데이터로 점수 계산 및 시각화 ---
    # 각 열(timestep)을 순회하며 그림을 그립니다.
    for t_idx in range(num_timesteps):
        # 현재 타임스텝의 잘라낸 데이터를 선택합니다.
        truth_t = all_truth_data.isel(time=t_idx)
        pred_1st_t = all_pred_data_1st_member.isel(time=t_idx)
        pred_ens_t = all_pred_data_ensembles.isel(time=t_idx)
        res_t = all_res_data.isel(time=t_idx)

        # MSE score (잘라낸 데이터 기준)
        mse = np.mean((pred_1st_t.values - truth_t.values)**2)
        mse_scores.append(mse)

        # CRPS score (잘라낸 데이터 기준)
        crps = ps.crps_ensemble(truth_t.values, pred_ens_t.values, axis=0).mean()
        crps_scores.append(crps)

        # 1행: Prediction (UNet)
        ax_res = axes[0, t_idx]
        im = ax_res.pcolormesh(x_cropped, y_cropped, res_t, 
                                vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        ax_res.set_xticks([])
        ax_res.set_yticks([])
        ax_res.set_title(f't = {t_idx}')

        # 2행: Prediction (1st member)
        ax_pred = axes[1, t_idx]
        ax_pred.pcolormesh(x_cropped, y_cropped, pred_1st_t, 
                            vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        
        # 3행: Ground Truth
        ax_truth = axes[2, t_idx]
        ax_truth.pcolormesh(x_cropped, y_cropped, truth_t, 
                            vmin=min_val, vmax=max_val, cmap='jet', shading='auto')
        ax_truth.set_xticks([])
        ax_truth.set_yticks([])

        # 3행 그래프 아래에 점수 텍스트 추가
        score_text = f"MSE: {mse:.3f}\nCRPS: {crps:.3f}"
        ax_truth.text(0.5, -0.2, score_text, 
                      ha='center', va='top', fontsize=10, 
                      transform=ax_truth.transAxes)

    # --- 6. 행(Row) 제목 설정 ---
    axes[0, 0].set_ylabel('Prediction\n(UNet)', fontsize=14, labelpad=10)
    axes[1, 0].set_ylabel('Prediction\n(1st member)', fontsize=14, labelpad=10)
    axes[2, 0].set_ylabel('Ground Truth', fontsize=14, labelpad=10)
    
    # --- 7. 전체 Figure에 대한 공통 설정 ---
    fig.suptitle(f'Cropped (96x96) Autoregressive Forecast vs. Ground Truth for "{variable_to_plot}"', fontsize=16)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=f'{variable_to_plot} units')

    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 모든 타임스텝을 포함한 그림이 '{save_path}' 파일로 저장되었습니다.")
    
    plt.close(fig)

except FileNotFoundError:
    print(f"❌ Error: '{nc_file_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
