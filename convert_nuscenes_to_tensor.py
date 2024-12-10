from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.input_representation.interface import AgentRepresentation
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFriction
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.utils.splits import train, val
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
import numpy as np
import os

# NuScenes 데이터 경로 및 출력 경로
NUSCENES_DATA_PATH = '../data/mini/v1.0-mini'
OUTPUT_PATH = './converted_data'

def load_nuscenes_data(nusc_path):
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_path, verbose=True)
    data = []

    # 샘플 레코드 순회
    for sample in nusc.sample:
        sample_token = sample['token']

        # 중심 차량의 과거 및 미래 궤적 추출
        instance_token = sample['instance_token']
        annotation = nusc.get('sample_annotation', instance_token)

        # 과거와 미래 데이터를 수집
        past_traj = nusc.get_past_for_agent(instance_token, seconds=2, in_agent_frame=True)
        future_traj = nusc.get_future_for_agent(instance_token, seconds=3, in_agent_frame=True)

        if past_traj is not None and future_traj is not None:
            center_gt_trajs = past_traj[['x', 'y', 'z']].to_numpy()
            center_gt_trajs_future = future_traj[['x', 'y', 'z']].to_numpy()

            data.append({
                "center_gt_trajs": center_gt_trajs,  # 과거 궤적
                "center_gt_trajs_future": center_gt_trajs_future  # 미래 궤적
            })

    return data

def convert_and_save_data():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    data = load_nuscenes_data(NUSCENES_DATA_PATH)
    np.save(f"{OUTPUT_PATH}/converted_data.npy", data)
    print(f"Data saved to {OUTPUT_PATH}/converted_data.npy")

if __name__ == "__main__":
    convert_and_save_data()