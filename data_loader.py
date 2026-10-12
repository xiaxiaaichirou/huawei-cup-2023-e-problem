# ============================================================
# 数据加载模块
# ============================================================
import pandas as pd
import numpy as np
from typing import Tuple, Dict


class DataLoader:
    """
    2023华为杯E题数据加载器
    用于加载临床信息、影像数据、随访信息等多个数据表
    """

    def __init__(self):
        self.clinical_data = None
        self.imaging_data = None
        self.followup_data = None
        self.merged_data = None

    def load_clinical_data(self, file_path: str) -> pd.DataFrame:
        """
        加载临床信息表
        常见字段：patient_id, age, gender, nihss, gcs, sbp, dbp, glucose, 
                  platelets, inr, diabetes, hypertension, prior_stroke
        """
        if file_path.endswith('.xlsx'):
            self.clinical_data = pd.read_excel(file_path)
        else:
            self.clinical_data = pd.read_csv(file_path)
        
        print(f"✓ 加载临床数据: {self.clinical_data.shape[0]} 病人, {self.clinical_data.shape[1]} 字段")
        print(f"  字段: {list(self.clinical_data.columns)}\n")
        return self.clinical_data

    def load_imaging_data(self, file_path: str) -> pd.DataFrame:
        """
        加载影像数据表（可能含多次检查）
        常见字段：patient_id, exam_time/timepoint_day, hematoma_volume_ml, 
                  edema_volume_ml, location, density, midline_shift_mm, ventricle_involvement
        """
        if file_path.endswith('.xlsx'):
            self.imaging_data = pd.read_excel(file_path)
        else:
            self.imaging_data = pd.read_csv(file_path)
        
        print(f"✓ 加载影像数据: {self.imaging_data.shape[0]} 检查记录, {self.imaging_data.shape[1]} 字段")
        print(f"  字段: {list(self.imaging_data.columns)}\n")
        return self.imaging_data

    def load_followup_data(self, file_path: str) -> pd.DataFrame:
        """
        加载随访数据表
        常见字段：patient_id, mrs_90d, surgery_type, surgery_time_hour, 
                  medication_type, outcome, discharge_status
        """
        if file_path.endswith('.xlsx'):
            self.followup_data = pd.read_excel(file_path)
        else:
            self.followup_data = pd.read_csv(file_path)
        
        print(f"✓ 加载随访数据: {self.followup_data.shape[0]} 病人, {self.followup_data.shape[1]} 字段")
        print(f"  字段: {list(self.followup_data.columns)}\n")
        return self.followup_data

    def merge_all_data(self, 
                       clinical: pd.DataFrame = None,
                       imaging: pd.DataFrame = None, 
                       followup: pd.DataFrame = None) -> pd.DataFrame:
        """
        合并所有数据表
        按 patient_id 进行 left merge
        """
        if clinical is None:
            clinical = self.clinical_data
        if imaging is None:
            imaging = self.imaging_data
        if followup is None:
            followup = self.followup_data

        # 先合并临床 + 影像
        merged = clinical.merge(imaging, on='patient_id', how='left')
        
        # 再合并随访
        merged = merged.merge(followup, on='patient_id', how='left')
        
        # 按病人和时间排序
        if 'timepoint_day' in merged.columns:
            merged = merged.sort_values(['patient_id', 'timepoint_day']).reset_index(drop=True)
        else:
            merged = merged.sort_values(['patient_id']).reset_index(drop=True)
        
        self.merged_data = merged
        print(f"✓ 数据合并完成: {merged.shape[0]} 行, {merged.shape[1]} 列\n")
        
        return merged

    def create_synthetic_data(self, n_patients: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        生成合成数据用于测试（如果你没有真实数据）
        """
        np.random.seed(42)
        
        # 1. 临床数据
        clinical_data = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'age': np.random.normal(65, 12, n_patients).astype(int),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'nihss': np.random.randint(5, 25, n_patients),
            'gcs': np.random.randint(8, 15, n_patients),
            'sbp': np.random.normal(160, 25, n_patients).astype(int),
            'dbp': np.random.normal(95, 15, n_patients).astype(int),
            'glucose': np.random.normal(140, 40, n_patients).astype(int),
            'platelets': np.random.normal(200, 50, n_patients).astype(int),
            'inr': np.random.normal(1.0, 0.3, n_patients),
            'diabetes': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'hypertension': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
            'prior_stroke': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
            'onset_to_ct_hour': np.random.uniform(0.5, 24, n_patients),
        })
        
        # 2. 影像数据（每个病人2-3次扫描）
        imaging_records = []
        for patient_id in range(1, n_patients + 1):
            n_scans = np.random.randint(2, 4)
            for scan_idx in range(n_scans):
                timepoint_day = scan_idx * 1  # 0天, 1天, 2天等
                baseline_volume = np.random.uniform(10, 100)
                
                # 模拟血肿扩张
                expand_prob = 0.3
                if np.random.random() < expand_prob and scan_idx > 0:
                    hematoma_vol = baseline_volume + np.random.uniform(6, 30)
                else:
                    hematoma_vol = baseline_volume + np.random.uniform(-2, 5)
                
                edema_vol = hematoma_vol * np.random.uniform(0.5, 2.0)
                
                imaging_records.append({
                    'patient_id': patient_id,
                    'timepoint_day': timepoint_day,
                    'hematoma_volume_ml': max(0, hematoma_vol),
                    'edema_volume_ml': max(0, edema_vol),
                    'location': np.random.choice(['basal_ganglia', 'thalamus', 'brainstem', 'cerebellum'], 1)[0],
                    'density_hu': np.random.randint(40, 80),
                    'midline_shift_mm': np.random.uniform(0, 15),
                    'ventricle_involvement': np.random.choice([0, 1], p=[0.7, 0.3]),
                })
        
        imaging_data = pd.DataFrame(imaging_records)
        
        # 3. 随访数据
        followup_data = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'mrs_90d': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_patients, 
                                       p=[0.15, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05]),
            'surgery_type': np.random.choice(['none', 'aspiration', 'craniotomy', 'decompression'], 
                                            n_patients, p=[0.6, 0.2, 0.1, 0.1]),
            'surgery_time_hour': np.random.uniform(0, 72, n_patients),
            'medication_type': np.random.choice(['standard', 'intensive'], n_patients),
            'outcome': np.random.choice(['died', 'poor', 'fair', 'good'], n_patients, 
                                       p=[0.15, 0.25, 0.35, 0.25]),
        })
        
        self.clinical_data = clinical_data
        self.imaging_data = imaging_data
        self.followup_data = followup_data
        
        print("✓ 合成数据生成完成")
        print(f"  临床数据: {clinical_data.shape}")
        print(f"  影像数据: {imaging_data.shape}")
        print(f"  随访数据: {followup_data.shape}\n")
        
        return clinical_data, imaging_data, followup_data

    def get_merged_data(self) -> pd.DataFrame:
        """获取合并后的数据"""
        if self.merged_data is None:
            raise ValueError("请先调用 merge_all_data()")
        return self.merged_data

    def get_data_summary(self) -> Dict:
        """获取数据汇总统计"""
        if self.merged_data is None:
            raise ValueError("请先调用 merge_all_data()")
        
        summary = {
            'n_patients': self.merged_data['patient_id'].nunique(),
            'n_records': len(self.merged_data),
            'n_columns': len(self.merged_data.columns),
            'missing_values': self.merged_data.isnull().sum().to_dict(),
            'dtypes': self.merged_data.dtypes.to_dict(),
        }
        return summary
