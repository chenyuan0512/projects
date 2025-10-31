import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
CSV_PATH = ".\\NewCSV"


def read_csv_file(file_path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path, header=None)


def find_turning_points(df):
    # 找法:
    # 要找最低點的方法可以從斜率來看, 找前後一正一負的
    # 如果是, sign_change == 1
    df['diff'] = df[1].diff()
    sign_change = (df['diff'] > 0) & (df['diff'].shift(1) <= 0)
    return sign_change[sign_change == 1].index - 1


def find_closest_points(max_index, indices):
    # 找法:
    # 如果要找峰值旁邊的兩個最低點, 應該看sign_change == 1
    # 且index離峰值index最近的點
    differences = np.abs(indices - max_index)
    sorted_indices = indices[np.argsort(differences)]
    return sorted_indices[:2]


def calculate_average_pulse_depth(max_value, min_value, closest_points_values):
    # 論文上的公式抄下來, 得到的是平均脈衝深度
    pulse_valley_height_left = closest_points_values[0]
    pulse_valley_height_right = closest_points_values[1]
    return (2 * max_value - pulse_valley_height_left - pulse_valley_height_right) / (2 * (max_value - min_value))


def plot_data(df, indices, closest_points, file):
    # 就畫圖, 並找出我要的點
    plt.figure(figsize=(10, 6))
    plt.plot(df[0], df[1], label='Waveform Data')
    plt.scatter(df[0].iloc[indices], df[1].iloc[indices],
                color='red', label='Turning Points')
    plt.scatter(df[0].iloc[closest_points], df[1].iloc[closest_points],
                color='green', label='Closest Points', s=100)
    plt.title(f'Waveform Plot for {file}')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.legend()
    plt.show()


def process_file(file):
    # 要讀csv檔, 所以把檔名寫進路徑裡

    file_path = os.path.join(CSV_PATH, file)
    df = read_csv_file(file_path)

    # 找到最大最小值和最大值index
    max_value = df[1].max()
    max_value_index = df[1].idxmax()
    min_value = df[1].min()

    indices = find_turning_points(df)  # 找到轉折點
    closest_points = find_closest_points(
        max_value_index, indices)  # 找到最近點的index
    closest_points_values = df[1].iloc[closest_points].values  # 找最近點的值

    average_pulse_depth = calculate_average_pulse_depth(
        max_value, min_value, closest_points_values)  # 平均脈衝深度

    # print(
    #     f"{file} - Found turning points at indices: {closest_points[0]}, {closest_points[1]}. Pulse depth: {average_pulse_depth*100:.4f}%")
    print(
        f"{file} - 取樣到的兩個波谷index為: {closest_points[0]}, {closest_points[1]}. 脈衝深度為: {average_pulse_depth*100:.4f}%")
    return average_pulse_depth
    plot_data(df, indices, closest_points, file)


def main():
    """Main function to process all files in the directory."""
    for file in os.listdir(CSV_PATH):
        if not (file.startswith("C2Trace") and file.endswith(".csv")):
            print(
                f"{file} does not match the required naming convention and will be skipped.")
            continue
        process_file(file)


if __name__ == "__main__":
    main()
