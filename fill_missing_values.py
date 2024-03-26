import pandas as pd

file_paths = [
    "N01T1", "N01T2", "N02T1", "N02T2", "N04T1", "N04T2", "N06T1", "N06T2", "N07T1", "N07T2", 
    "N11T1", "N11T2", "N12T1", "S01T1", "S01T2", "S02T1", "S02T2", "S03T1", "S03T2", "S05T1", 
    "S05T2", "S07T1", "S07T2", "S08T1", "S08T2", "S09T1", "S09T2", "S10T1", "S10T2", "S11T1", 
    "S11T2"
]

folder_path = r"C:\Users\86152\Desktop\Nurse Care Activity recognization\features and labels\\"

# 处理训练集文件
for file in file_paths:
    file_path = folder_path + file + "_features_and_labels.csv"
    df = pd.read_csv(file_path)
    df_filled = df.fillna(0)  # 将缺失值替换为0
    df_filled.to_csv(file_path, index=False)

