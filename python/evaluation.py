import os
import argparse
from joblib import Parallel, delayed

import numpy as np 

import raplace_config
from calculate_similarity import fast_dft
from generate_raplace_descriptor import generate_raplace_descriptor
import matplotlib.pyplot as plt
import seaborn as sns

from tic import tic


def compute_similarity_parallel(descriptors):
    """
    Compute similarity matrix using FFT-based cross-correlation in parallel.

    Args:
        descriptors (list of numpy.ndarray): List of descriptors, each of shape (N, M).

    Returns:
        numpy.ndarray: Similarity matrix of shape (num_descriptors, num_descriptors).
    """
    num_descriptors = len(descriptors)
    similarity_matrix = np.zeros((num_descriptors, num_descriptors))

    def compute_pair(i, j):
        _, max_corr = fast_dft(descriptors[i], descriptors[j])
        return (i, j, max_corr)

    # Create all unique pairs (i, j) where i <= j
    pairs = [(i, j) for i in range(num_descriptors) for j in range(i, num_descriptors)]

    # Compute all pairs in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_pair)(i, j) for i, j in pairs)

    # Fill the similarity matrix
    for i, j, max_corr in results:
        similarity_matrix[i, j] = max_corr
        similarity_matrix[j, i] = max_corr

    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, title="Similarity Matrix"):
    """
    Plot the similarity matrix using seaborn heatmap.

    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix of shape (N, N).
        title (str): Title of the plot.
    """
    # similarity_matrix 저장
    np.save("similarity_matrix.npy", similarity_matrix)  # 파일 이름을 원하는 대로 지정 가능

    # plt.ion()  # 인터랙티브 모드 활성화
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(similarity_matrix, cmap="inferno_r", vmin=-0.01, vmax=0.1)
    cbar = heatmap.collections[0].colorbar  # 컬러바 객체 가져오기

    plt.title(title)
    plt.xlabel("Descriptor Index")
    plt.ylabel("Descriptor Index")
    plt.show()

    # clim을 재조정하는 예시
    # cbar.set_clim(0.2, 0.8)  # 새로운 컬러 범위로 설정
    # plt.draw()  # 업데이트된 값 반영


@tic
def match_descriptors(query_sinofft, candidate_sinoffts):
    """
    Match a query descriptor against all candidate descriptors.

    Args:
        query_sinofft (numpy.ndarray): Query descriptor of shape (N, M).
        candidate_sinoffts (list of numpy.ndarray): List of candidate descriptors.

    Returns:
        tuple:
            best_match_idx (int): Index of the best matching candidate.
            best_match_val (float): Maximum correlation value.
    """
    max_correlation = -np.inf
    best_match_idx = -1
    for idx, candidate in enumerate(candidate_sinoffts):
        _, current_max = fast_dft(query_sinofft, candidate)
        if current_max > max_correlation:
            max_correlation = current_max
            best_match_idx = idx
    return best_match_idx, max_correlation


def compute_similarity_matrix(descriptors):
    """
    Compute a similarity matrix using FFT-based cross-correlation.

    Args:
        descriptors (list of numpy.ndarray): List of descriptors, each of shape (N, M).

    Returns:
        numpy.ndarray: Similarity matrix of shape (num_descriptors, num_descriptors).
    """
    num_descriptors = len(descriptors)
    similarity_matrix = np.zeros((num_descriptors, num_descriptors))

    def compute_pair(i, j):
        _, max_corr = fast_dft(descriptors[i], descriptors[j])
        return (i, j, max_corr)

    # Create all unique pairs (i, j) where i <= j
    pairs = [(i, j) for i in range(num_descriptors) for j in range(i, num_descriptors)]

    # Compute all pairs in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_pair)(i, j) for i, j in pairs)

    # Fill the similarity matrix
    for i, j, max_corr in results:
        similarity_matrix[i, j] = max_corr
        similarity_matrix[j, i] = max_corr  # Symmetric matrix

    return similarity_matrix


def evaluate(sinoffts):
    """
    Compare sinofft descriptors and visualize the similarity matrix.

    Args:
        sinoffts (list of numpy.ndarray): List of sinofft descriptors.
        query_poses (list of iterable): List of pose information for each query descriptor.
        exp_poses (list of iterable): List of pose information for each candidate descriptor.

    Returns:
        distance_matrix (numpy.ndarray): Matrix of computed sim.
    """
    num_queries = len(sinoffts)
    distance_matrix = np.zeros((num_queries, num_queries))

    for query_idx in range(num_queries):

        current_min_distance = np.inf
        current_nearest_target_idx = None

        query_sinofft = sinoffts[query_idx]

        _, self_corr = fast_dft(query_sinofft, query_sinofft)
        for target_idx, target_sinofft in enumerate(sinoffts):
            _, corr = fast_dft(query_sinofft, target_sinofft)

            # calibrated_corr = (self_corr - corr) / 1000000
            mutably_calibrated_distance = abs((self_corr - corr) / self_corr)
            print(f" calibrated_corr btn {query_idx} and {target_idx}: {mutably_calibrated_distance:.4f}")
            if mutably_calibrated_distance < current_min_distance:
                current_min_distance = mutably_calibrated_distance
                current_nearest_target_idx = target_idx

            # gather all results for drawing sim mat
            distance_matrix[query_idx, target_idx] = mutably_calibrated_distance

        # argmax disp 
        nearest_idx = current_min_distance
        nearest_similarity = current_nearest_target_idx 
        print(f"Query {query_idx}: Nearest Index = {nearest_idx}, Max Distance = {nearest_similarity}")

    print("Plotting distance matrix...")
    plot_similarity_matrix(distance_matrix, title="Distance Matrix")
    print("Plotting completed.")

    return distance_matrix


def generate_raplace_descriptors_offline_batch(radar_data_dir):
    """
    디렉토리 내의 여러 이미지에 대해 Radon 변환과 FFT를 수행하는 함수.

    Args:
        data_dir (str): 이미지들이 있는 디렉토리 경로.

    Returns:
        sinoffts (list): 모든 이미지의 Radon 변환 후 FFT 결과 리스트.
        rowkeys (numpy.ndarray): 모든 이미지의 Radon 변환된 결과를 1차원 배열로 변환한 리스트.
    """
    # radar data directory
    data_names = os.listdir(radar_data_dir)
    data_names.sort()

    num_data = len(data_names)
    print(f"num_data in {radar_data_dir}: {num_data}")

    theta_for_radon_tf = np.arange(180)  # Angle for radon transform
    sinoffts = []

    for data_idx, file_name in enumerate(data_names):
        if data_idx % raplace_config.config["skip_for_fast_eval"] != 0:
            continue

        print(" ")
        data_path = os.path.join(radar_data_dir, file_name)

        # 하나의 이미지에 대해 변환 수행
        print(f"processing {data_idx} th data: {data_path} ...")
        sinofft = generate_raplace_descriptor(
            data_path, theta_for_radon_tf, raplace_config.config
        )

        # 결과 저장
        sinoffts.append(sinofft)

        # Log progress every 100 iterations
        if (data_idx + 1) % 100 == 0:
            message = f"{data_idx + 1} / {num_data} processed."
            print(message)

    return sinoffts


def main():
    """
    명령어로 실행할 때 인자를 받아 Radon 변환을 수행하는 메인 함수.
    """
    parser = argparse.ArgumentParser(
        description="Perform Radon transform on radar images."
    )

    # 명령행 인자 설정
    parser.add_argument(
        "data_dir", type=str, help="The directory where the radar images are stored."
    )

    # 명령행 인자 파싱
    args = parser.parse_args()

    # Radon 변환 수행
    sinoffts = generate_raplace_descriptors_offline_batch(args.data_dir)

    # 결과를 파일로 저장하거나 처리할 수 있습니다. 여기서는 간단히 로그를 출력합니다.
    print(f"\n\nProcessed {len(sinoffts)} images.\n\n")

    # 평가 수행
    evaluate(sinoffts)


if __name__ == "__main__":
    
    # main()
    
    plot_similarity_matrix(np.load("similarity_matrix.npy"))
