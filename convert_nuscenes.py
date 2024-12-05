import os
import numpy as np
import json
from nuscenes.nuscenes import NuScenes

def calculate_velocity(translation_current, translation_previous, time_diff):
    """
    Calculate velocity based on the change in translation and time difference.

    Args:
        translation_current (list): [x, y, z] position in the current frame.
        translation_previous (list): [x, y, z] position in the previous frame.
        time_diff (float): Time difference between frames in seconds.

    Returns:
        list: [vx, vy] velocity components (z-component is ignored).
    """
    if time_diff == 0:
        return [0.0, 0.0]
    velocity = [(translation_current[i] - translation_previous[i]) / time_diff for i in range(2)]  # x, y only
    return velocity

def calculate_acceleration(velocity_current, velocity_previous, time_diff):
    """
    Calculate acceleration based on the change in velocity and time difference.

    Args:
        velocity_current (list): [vx, vy] velocity in the current frame.
        velocity_previous (list): [vx, vy] velocity in the previous frame.
        time_diff (float): Time difference between frames in seconds.

    Returns:
        list: [ax, ay] acceleration components.
    """
    if time_diff == 0:
        return [0.0, 0.0]
    acceleration = [(velocity_current[i] - velocity_previous[i]) / time_diff for i in range(2)]  # x, y only
    return acceleration

def process_nuscenes_data(nusc, save_path):
    for scene in nusc.scene:
        scene_data = []
        current_sample_token = scene['first_sample_token']

        # 객체 추적용 저장소
        previous_translations = {}
        previous_velocities = {}
        previous_timestamps = {}

        while current_sample_token != "":
            sample = nusc.get('sample', current_sample_token)
            timestamp = sample['timestamp']  # 현재 타임스탬프

            for ann_token in sample['anns']:
                annotation = nusc.get('sample_annotation', ann_token)
                translation = annotation['translation']
                instance_token = annotation['instance_token']

                # 속도와 가속도 초기화
                velocity = 0.0
                acceleration = 0.0

                if instance_token in previous_translations:
                    # 이전 프레임 정보 가져오기
                    prev_translation = previous_translations[instance_token]
                    prev_velocity = previous_velocities[instance_token]
                    prev_timestamp = previous_timestamps[instance_token]

                    # 속도 계산
                    delta_time = (timestamp - prev_timestamp) / 1e6  # 밀리초 → 초
                    velocity = (
                        (
                            (translation[0] - prev_translation[0])**2
                            + (translation[1] - prev_translation[1])**2
                            + (translation[2] - prev_translation[2])**2
                        )**0.5 / delta_time
                    )

                    # 가속도 계산
                    acceleration = (velocity - prev_velocity) / delta_time

                # 업데이트
                previous_translations[instance_token] = translation
                previous_velocities[instance_token] = velocity
                previous_timestamps[instance_token] = timestamp

                # 데이터를 기록
                scene_data.append({
                    'instance_token': instance_token,
                    'translation': translation,
                    'velocity': velocity,
                    'acceleration': acceleration,
                })

            current_sample_token = sample['next']

        # 파일 저장
        scene_file = os.path.join(save_path, f"scene-{scene['name']}.json")
        with open(scene_file, 'w') as f:
            json.dump(scene_data, f, indent=4)
        print(f"Saved scene data to {scene_file}")

# Main function to execute the processing
if __name__ == "__main__":
    # Path to nuScenes dataset
    # nusc_path = "./v1.0-mini"
    nusc_path = "../data/meta/v1.0-trainval_meta"

    # Path to save processed data
    save_path = "./processed_data"

    # Load nuScenes dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_path, verbose=True)

    # Process data and save
    process_nuscenes_data(nusc, save_path)
