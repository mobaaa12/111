import pandas as pd
import numpy as np

def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def extract_joint_angles(kp_data, steps=2):
    left_elbow_shoulder_hip = np.asarray([cal_angle(kp_data[i, 7*steps:(7*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 11*steps:(11*steps+2)])
                                          for i in range(len(kp_data))])
    left_elbow_shoulder_hip = np.nan_to_num(left_elbow_shoulder_hip)

    right_elbow_shoulder_hip = np.asarray([cal_angle(kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 12*steps:(12*steps+2)])
                                            for i in range(len(kp_data))])
    right_elbow_shoulder_hip = np.nan_to_num(right_elbow_shoulder_hip)

    left_wrist_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 9*steps:(9*steps+2)], kp_data[i, 7*steps:(7*steps+2)], kp_data[i, 5*steps:(5*steps + 2)])
                                            for i in range(len(kp_data))])
    left_wrist_elbow_shoulder = np.nan_to_num(left_wrist_elbow_shoulder)

    right_wrist_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 10*steps:(10*steps+2)], kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)])
                                              for i in range(len(kp_data))])
    right_wrist_elbow_shoulder = np.nan_to_num(right_wrist_elbow_shoulder)

    right_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 5*steps:(5*steps+2)])
                                              for i in range(len(kp_data))])
    right_elbow_shoulder = np.nan_to_num(right_elbow_shoulder)

    left_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 7*steps:(7*steps+2)])
                                              for i in range(len(kp_data))])
    left_elbow_shoulder = np.nan_to_num(left_elbow_shoulder)

    joint_angles = np.array([left_elbow_shoulder_hip,
                             right_elbow_shoulder_hip, left_wrist_elbow_shoulder, right_wrist_elbow_shoulder, right_elbow_shoulder, left_elbow_shoulder]).T

    return joint_angles


def extract_velocity(kp_data):
    velocity = np.diff(kp_data, axis=0)
    return velocity


def extract_feature(data, fs):
    mean_ft = np.mean(data, axis=0)
    std_ft = np.std(data, axis=0)
    max_ft = np.max(data, axis=0)
    min_ft = np.min(data, axis=0)
    var_ft = np.var(data, axis=0)
    med_ft = np.median(data, axis=0)
    sum_ft = np.sum(data, axis=0)
    features = np.array([mean_ft, std_ft, max_ft, min_ft, var_ft, med_ft, sum_ft]).T.flatten()
    features = np.nan_to_num(features)
    return features


WINDOW_SIZE = 2  # seconds
OVERLAP_RATE = 0.5 * WINDOW_SIZE  # overlap 50% of window size
FS = 30

# 定义文件夹路径
output_folder = r"C:\Users\86152\Desktop\Nurse Care Activity recognization\features and labels"

# 定义ID列表
ids = [
    "N01T1", "N01T2", "N02T1", "N02T2",
    "N04T1", "N04T2", "N06T1", "N06T2",
    "N07T1", "N07T2", "N11T1", "N11T2", "N12T1",
    "S01T1", "S01T2", "S02T1", "S02T2",
    "S03T1", "S03T2", "S05T1", "S05T2",
    "S07T1", "S07T2", "S08T1", "S08T2",
    "S09T1", "S09T2", "S10T1", "S10T2",
    "S11T1", "S11T2"
]


for id in ids:
    # 定义文件路径
    annotation_file = fr"C:\Users\86152\Desktop\Nurse Care Activity recognization\ann\{id}_ann.csv"
    keypoint_file = fr"C:\Users\86152\Desktop\Nurse Care Activity recognization\keypoints_smoothed\{id}_keypoint_smoothed.csv"
    output_file = fr"{output_folder}\{id}_features_and_labels.csv"

    # 加载注释信息文件和关键点文件
    annotation_df = pd.read_csv(annotation_file)  # 加载注释信息文件为DataFrame
    keypoint_df = pd.read_csv(keypoint_file)  # 加载关键点文件为DataFrame

    # 提取特征和标签
    features = []  # 存储特征的列表
    labels = []  # 存储标签的列表

    for i in range(len(annotation_df)):
        start_time = int(annotation_df['start_time'].iloc[i] * FS)  # 获取注释的开始时间并转换为样本索引
        stop_time = int(annotation_df['stop_time'].iloc[i] * FS)  # 获取注释的结束时间并转换为样本索引

        for t in range(start_time, stop_time, FS):  # 每秒提取一次数据
            segment = keypoint_df.iloc[t:t+FS]  # 提取1秒的关键点数据段
            label = annotation_df["annotation"].iloc[i]  # 获取当前行的标签

            if len(segment) > 0:  # 如果数据段的长度大于0，则进行以下操作
                # 从关键点数据中计算关节角度
                joint_angles = extract_joint_angles(np.array(segment))

                # 从关键点数据中计算速度
                velocity = extract_velocity(np.array(segment))

                # 从关键点数据、关节角度和速度中提取特征
                feature = extract_feature(np.array(segment), FS)
                joint_angles_feature = extract_feature(joint_angles, FS)
                velocity_feature = extract_feature(velocity, FS)

                # 连接所有特征
                feature = np.concatenate([feature, joint_angles_feature, velocity_feature])

                features.append(feature)  # 将特征向量添加到特征列表中
                labels.append(label)  # 将标签添加到标签列表中

    # 创建特征和标签的DataFrame
    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels, columns=["label"])

    # 将特征和标签合并为一个DataFrame
    merged_df = pd.concat([features_df, labels_df], axis=1)

    # 保存合并后的DataFrame为CSV文件
    merged_df.to_csv(output_file, index=False)
