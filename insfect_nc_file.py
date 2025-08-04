import xarray as xr
import h5py
import os

def inspect_nc_structure(file_path):
    """
    NetCDF 파일의 전체 구조(그룹 포함)를 자세히 출력합니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"--- '{file_path}' 파일 구조 분석 ---")
    try:
        # 최상위(root) 데이터셋 정보 출력
        print("\n[최상위 (root) 데이터셋 정보]")
        with xr.open_dataset(file_path) as ds_root:
            print(ds_root)

        # h5py를 사용하여 파일 내 그룹 목록을 가져옵니다.
        with h5py.File(file_path, 'r') as f:
            # 최상위 데이터셋에 포함된 항목(좌표, 변수)들을 제외하여 순수 그룹 목록을 찾습니다.
            with xr.open_dataset(file_path) as ds_root:
                root_items = list(ds_root.variables) + list(ds_root.coords)
                groups = [key for key in f.keys() if key not in root_items]

            if groups:
                print("\n--- 그룹별 상세 정보 ---")
                for group_name in groups:
                    print(f"\n[그룹: '{group_name}']")
                    try:
                        with xr.open_dataset(file_path, group=group_name) as ds_group:
                            print(ds_group)
                    except Exception as e:
                        print(f"  '{group_name}' 그룹을 여는 중 오류 발생: {e}")
            else:
                print("\n[알림] 파일 내에 별도의 그룹은 없습니다.")

    except Exception as e:
        print(f"파일을 분석하는 중 오류가 발생했습니다: {e}")

    print("\n--- 분석 완료 ---")


# --- ⚙️ 설정 ---
# 분석하고 싶은 .nc 파일의 경로를 여기에 입력하세요.
nc_file_to_inspect = 'corrdiff_output.nc'
# --- -------- ---

inspect_nc_structure(nc_file_to_inspect)