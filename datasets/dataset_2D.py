import os
import torch
import numpy as np
import cftime
from datetime import datetime, timedelta
import json
from datasets.base import ChannelMetadata, DownscalingDataset

class Dataset_2D(DownscalingDataset):
    def __init__(self, data_path, stats_path=None, input_variables=None, output_variables=None, invariant_variables=None):
        print("✅ dataset2d init")

        self.data_path = data_path
        self.fnames = sorted([f for f in os.listdir(data_path) if f.endswith('.pt')])
        # 여기서 한 번만 첫 샘플 불러서 shape 정보 저장
        first_sample = torch.load(os.path.join(self.data_path, self.fnames[0]))
        
        n_channels = first_sample[0].shape[0]  # x의 채널 수 기준 (C, H, W)일 때

        self.input_vars = input_variables if input_variables else list(range(n_channels))
        self.output_vars = output_variables if output_variables else list(range(n_channels))
     
        self._img_shape = first_sample[0].shape[1:]  # (H, W)
                # --- latitude, longitude mesh 생성 ---
        
        H, W = self._img_shape
        # 16으로 padding
        self._lat = np.linspace(-16, 112, H)   # shape (H,)
        self._lon = np.linspace(-16, 112, W)   # shape (W,)

        if stats_path is not None and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.input_mean = np.array([stats[v]["mean"] for v in self.input_vars])[:, None, None]
            self.input_std = np.array([stats[v]["std"] for v in self.input_vars])[:, None, None]
        else:
            self.input_mean = self.input_std = None


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        """
        Returns:
        Tuple containing:
        - img_clean: Target high-resolution data [output_channels, height, width]
        - img_lr: Input low-resolution data [input_channels, height, width]
        - lead_time_label: (Optional) Lead time information [1]
        """
        fpath = os.path.join(self.data_path, self.fnames[idx])
        data = torch.load(fpath)
        x = data[0]
        y = data[1]
        # return img_clean, img_lr
        return y, x

    
    def input_channels(self):
        # return [ChannelMetadata(name=str(v), level='') for v in self.input_vars]
        return [ChannelMetadata(name=v) for v in self.input_vars]

    def output_channels(self):
        # return [ChannelMetadata(name=str(v), level='') for v in self.output_vars]
        return [ChannelMetadata(name=v) for v in self.output_vars]

    def image_shape(self):
        return self._img_shape

    def time(self):
        print("✅ time")
        base_time = datetime(2000, 1, 1, 0, 0, 0)
        times = []
        for i in range(len(self)):
            # 간격 하드코딩 되어있는듯 ?
            t = base_time + timedelta(minutes=i*1)
            # cftime 객체로!
            times.append(cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second))
        return times
    
    def latitude(self):
        # return 2D array (H, W)
        lat2d, _ = np.meshgrid(self._lat, self._lon, indexing='ij')
        return lat2d

    def longitude(self):
        # return 2D array (H, W)
        _, lon2d = np.meshgrid(self._lat, self._lon, indexing='ij')
        return lon2d
    
    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return x

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return x

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.input_std + self.input_mean
