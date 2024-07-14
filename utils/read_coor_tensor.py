import os
import json
import numpy as np
import torch
import re  # 导入正则表达式库


def process_coordinates(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return {}

    index_to_coords = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()[2:]  # Skip the first two lines
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # 使用正则表达式提取字符串中的数字部分
                    match = re.search(r'\d+', parts[0])
                    if match:
                        index = int(match.group())  # 转换为整数
                        coordinates = tuple(map(float, parts[-3:]))
                        if index not in index_to_coords:
                            index_to_coords[index] = []
                        index_to_coords[index].append(coordinates)
                except ValueError:
                    continue

    # Calculate the average coordinates
    averaged_coords = {index: np.mean(coords, axis=0) for index, coords in index_to_coords.items()}
    return averaged_coords


def read_all_files(directory, letter, k, t):
    file_name = f"{k}_{t}ns-{letter}.gro"
    file_path = os.path.join(directory, file_name)
    averaged_coords = process_coordinates(file_path)

    # Convert to tensor and sort by index
    indices = sorted(averaged_coords.keys())
    tensor_coords = torch.tensor([averaged_coords[i] for i in indices], dtype=torch.float)

    return tensor_coords

#tenasor1 = read_all_files('../raw_data/monomer\structure', 'A', 1, -0.2)
#print(tenasor1)