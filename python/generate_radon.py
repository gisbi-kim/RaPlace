import os
import argparse
import numpy as np
from scipy.fft import fft, ifft
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from joblib import Parallel, delayed

from skimage.transform import radon
from PIL import Image
import matplotlib.pyplot as plt

from tic import tic


@tic
def fast_dft(query_descriptor, candidate_descriptor):
    """
    Perform FFT-based cross-correlation between query and candidate descriptors.

    Args:
        query_descriptor (numpy.ndarray): Query descriptor of shape (N, M).
        candidate_descriptor (numpy.ndarray): Candidate descriptor of shape (N, M).

    Returns:
        tuple:
            correlation_map (numpy.ndarray): Correlation map of shape (N,).
            max_correlation (float): Maximum correlation value.
    """
    # Perform FFT along the theta axis for both descriptors
    query_fft = fft(query_descriptor, axis=1)
    candidate_fft = fft(candidate_descriptor, axis=1)

    # Compute the cross-correlation in the frequency domain
    correlation_map_2d = ifft(query_fft * np.conj(candidate_fft), axis=1)

    # Sum the correlation map along the second axis to get a 1D correlation map
    correlation_map = np.sum(correlation_map_2d.real, axis=1)

    # Find the maximum correlation value
    max_correlation = np.max(correlation_map)

    return correlation_map, max_correlation


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
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Descriptor Index")
    plt.ylabel("Descriptor Index")
    plt.show()


@tic
def polar_to_cartesian_naive(polar_image, cartesian_size, polar_pixel_size):
    """
    Polar 좌표계 이미지를 Cartesian 좌표계 이미지로 변환하는 함수.

    Args:
        polar_image (numpy.ndarray): Polar 좌표계의 입력 이미지.
        cartesian_size (int): Cartesian 이미지의 크기 (픽셀 수).
        polar_pixel_size (float): Polar 이미지에서 하나의 픽셀이 차지하는 거리 (미터 단위).
    Returns:
        cartesian_image (numpy.ndarray): 변환된 Cartesian 좌표계 이미지.
    """
    # Polar image shape
    height, width = polar_image.shape  # height=3360 (range), width=400 (angle)

    # Cartesian 이미지 초기화
    cartesian_image = np.zeros((cartesian_size, cartesian_size))

    # Cartesian 이미지에서 각 픽셀이 몇 미터를 차지할 것인지를 설정
    cartesian_pixel_size = (polar_pixel_size * height) / (cartesian_size / 2)

    # Cartesian 이미지의 중심 좌표
    center = cartesian_size // 2

    # 각도 설정 (360도를 width로 나눔)
    theta_values = np.linspace(0, 2 * np.pi, width)

    for i in range(height):  # range loop
        for j in range(width):  # angle loop
            r = i * polar_pixel_size  # 반지름 (거리)
            theta = theta_values[j]  # 각도

            # Polar to Cartesian 변환
            x = int(center + (r / cartesian_pixel_size) * np.cos(theta))
            y = int(center + (r / cartesian_pixel_size) * np.sin(theta))

            # Cartesian 좌표계에 값 할당
            if 0 <= x < cartesian_size and 0 <= y < cartesian_size:
                cartesian_image[y, x] = polar_image[i, j]

    return cartesian_image


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


@tic
def polar_to_cartesian_interpolated(polar_image, cartesian_size, polar_pixel_size):
    """
    보간법을 사용하여 Polar 좌표계 이미지를 Cartesian 좌표계 이미지로 변환하는 함수.

    Args:
        polar_image (numpy.ndarray): Polar 좌표계의 입력 이미지.
        cartesian_size (int): Cartesian 이미지의 크기 (픽셀 수).
        polar_pixel_size (float): Polar 이미지에서 하나의 픽셀이 차지하는 거리 (미터 단위).
    Returns:
        cartesian_image (numpy.ndarray): 변환된 Cartesian 좌표계 이미지.
    """
    # Polar image shape
    height, width = polar_image.shape  # height=3360 (range), width=400 (angle)

    # Cartesian 이미지에서 각 픽셀이 몇 미터를 차지할 것인지를 설정
    cartesian_pixel_size = (polar_pixel_size * height) / (cartesian_size / 2)

    # Cartesian 이미지의 중심 좌표
    center = cartesian_size // 2

    # 각도 및 반지름 설정
    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)
    r = np.linspace(0, polar_pixel_size * height, height)

    # 생성할 Cartesian 좌표 그리드
    x = (
        np.linspace(-cartesian_size / 2, cartesian_size / 2, cartesian_size)
        * cartesian_pixel_size
    )
    y = (
        np.linspace(-cartesian_size / 2, cartesian_size / 2, cartesian_size)
        * cartesian_pixel_size
    )
    X, Y = np.meshgrid(x, y)

    # 변환된 좌표에서 r과 theta 계산
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X) % (2 * np.pi)

    # 원래 Polar 이미지의 좌표에 맞추기 위해 r과 theta를 인덱스로 변환
    # 보간을 위해 각 점의 (r, theta)를 평탄화된 형태로 변환
    points = np.vstack((r.repeat(width), np.tile(theta, height))).T
    values = polar_image.flatten()

    # 보간을 통해 Cartesian 이미지에 값 할당
    cartesian_image = griddata(
        points, values, (R, Theta), method="linear", fill_value=0
    )

    return cartesian_image


@tic
def polar_to_cartesian_fast(polar_image, cartesian_size, polar_pixel_size):
    """
    보간법을 사용하여 Polar 좌표계 이미지를 Cartesian 좌표계 이미지로 변환하는 함수.
    좌표 변환을 정확하게 수행하여 이미지가 중앙에 위치하도록 합니다.

    Args:
        polar_image (numpy.ndarray): Polar 좌표계의 입력 이미지.
        cartesian_size (int): Cartesian 이미지의 크기 (픽셀 수).
        polar_pixel_size (float): Polar 이미지에서 하나의 픽셀이 차지하는 거리 (미터 단위).
    Returns:
        cartesian_image (numpy.ndarray): 변환된 Cartesian 좌표계 이미지.
    """
    height, width = polar_image.shape

    # Cartesian 이미지에서 각 픽셀이 몇 미터를 차지할 것인지를 설정
    cartesian_pixel_size = (polar_pixel_size * height) / (cartesian_size / 2)

    # Cartesian 이미지의 중심 좌표
    center = cartesian_size // 2

    # 각도 및 반지름 설정
    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)
    r = np.linspace(0, polar_pixel_size * height, height)

    # 생성할 Cartesian 좌표 그리드 (중앙 정렬)
    x = (np.arange(cartesian_size) - center) * cartesian_pixel_size
    y = (np.arange(cartesian_size) - center) * cartesian_pixel_size
    X, Y = np.meshgrid(x, y)

    # 변환된 좌표에서 r과 theta 계산
    R = np.sqrt(X**2 + Y**2) / polar_pixel_size
    Theta = np.arctan2(Y, X) % (2 * np.pi)

    # 폴라 이미지의 좌표계에 맞게 인덱스 계산
    r_idx = R
    theta_idx = Theta / (2 * np.pi) * width

    # 좌표 클리핑 (범위를 벗어나지 않도록)
    r_idx = np.clip(r_idx, 0, height - 1)
    theta_idx = np.clip(theta_idx, 0, width - 1)

    # map_coordinates는 (행, 열) 순서의 좌표를 요구합니다.
    coordinates = np.vstack((r_idx.flatten(), theta_idx.flatten()))

    # 보간 수행 (order=1은 선형 보간)
    cartesian_flat = map_coordinates(
        polar_image, coordinates, order=1, mode="constant", cval=0.0
    )

    # 평탄화를 원래의 그리드 형태로 복원
    cartesian_image = cartesian_flat.reshape(cartesian_size, cartesian_size)

    return cartesian_image


@tic
def radon_transform(cartesian_image, theta, circle=False):
    return radon(cartesian_image, theta=theta, circle=False)


@tic
def downsize_polar_image(polar_image, downscale_factor):
    """
    Polar 이미지를 다운사이즈하는 함수.

    Args:
        polar_image (numpy.ndarray): 원본 Polar 이미지.
        downscale_factor (float): 이미지를 축소할 비율 (0.5는 50% 축소).

    Returns:
        downsized_polar_image (numpy.ndarray): 축소된 Polar 이미지.
    """
    # 이미지 축소 (선형 보간 사용, order=1)
    downsized_polar_image = zoom(
        polar_image, (downscale_factor, downscale_factor), order=1
    )

    return downsized_polar_image


def draw_descriptors(cartesian_image, radon_image, sinofft, rowkey):
    # 시각화
    plt.figure(figsize=(24, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cartesian_image, cmap="gray")
    plt.title("Cartesian Image from Polar Coordinates")
    plt.colorbar()

    # 1. Radon 이미지 시각화
    plt.subplot(1, 4, 2)
    plt.imshow(radon_image, cmap="gray", aspect="auto")
    plt.title("Radon Transform Image")
    plt.colorbar()

    # 2. sinofft 시각화 (FFT 결과)
    plt.subplot(1, 4, 3)
    plt.imshow(sinofft, cmap="gray", aspect="auto")
    plt.title("Sinofft (FFT of Radon Image)")
    plt.colorbar()

    # 3. rowkey 시각화 (1차원 벡터)
    plt.subplot(1, 4, 4)
    plt.plot(rowkey)
    plt.title("Rowkey (Flattened Radon Image)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(0, 1)  # Fix y-axis range to [0, 1]

    plt.tight_layout()
    plt.show()


def generate_rowkey(radon_img):
    """
    Radon 변환 이미지의 중앙 행을 반환하는 함수.

    Args:
        radon_img (numpy.ndarray): Radon 변환된 이미지.

    Returns:
        row_key (numpy.ndarray): Radon 이미지의 중앙 행.
    """
    # Radon 이미지의 중앙 행을 선택
    row_idx = radon_img.shape[0] // 2  # 중앙 행의 인덱스
    row_key = radon_img[row_idx, :]  # 중앙 행을 선택

    return row_key


@tic
def generate_descriptor(image_path, theta, config):
    """
    하나의 polar 이미지를 받아서 Cartesian 좌표계로 변환 후 Radon 변환 및 FFT를 수행하는 함수.

    Args:
        image_path (str): 처리할 polar 이미지의 파일 경로.
        theta (numpy.ndarray): Radon 변환에서 사용할 각도 배열.
        config["down_shape"] (int): 이미지의 크기를 변경할 크기.
    Returns:
        sinofft (numpy.ndarray): 이미지의 Radon 변환 후 FFT 결과.
        rowkey (numpy.ndarray): 이미지의 Radon 변환된 결과를 1차원으로 변환한 배열.
    """

    # Load and process radar image using PIL
    polar_image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale
    print(f" polar img_gray of size {polar_image.shape} is read.")

    NAVTECH_RADAR_MAX_SENSING_RANGE = 200.0
    original_polar_pixel_size = (
        NAVTECH_RADAR_MAX_SENSING_RANGE / polar_image.shape[0]
    )  # Polar 이미지에서 한 픽셀이 차지하는 거리 (미터)

    # 다운사이즈 설정
    downsized_polar_image = downsize_polar_image(
        polar_image, config["polar_downscale_factor"]
    )
    # 다운사이즈 후의 픽셀당 거리 계산
    downsized_polar_pixel_size = (
        original_polar_pixel_size / config["polar_downscale_factor"]
    )

    # Polar to Cartesian 변환
    # cartesian_image = polar_to_cartesian_naive(
    cartesian_image = polar_to_cartesian_fast(
        downsized_polar_image, config["cartesian_size"], downsized_polar_pixel_size
    )
    print(f" cartesian_image of size {cartesian_image.shape} is made.")

    # Radon transform
    radon_image = radon_transform(cartesian_image, theta=theta, circle=False)
    print(f" radon_image (raw) of size {radon_image.shape} is made.")

    radon_image = radon_image / np.max(radon_image)
    radon_image = zoom(
        radon_image,
        (
            config["radon_zoom_shape"] / radon_image.shape[0],
            config["radon_zoom_shape"] / radon_image.shape[1],
        ),
    )
    print(f" radon_image (zoomed) of size {radon_image.shape} is made.")

    # FFT and process the result
    sinofft = np.abs(fft(radon_image, axis=0))
    sinofft = sinofft[: sinofft.shape[0] // 2, :]
    print(f" sinofft of size {radon_image.shape} is made.")

    # Flatten the Radon transform result for rowkey
    rowkey = generate_rowkey(radon_image)
    print(f" rowkey of size {rowkey.shape} is made.")

    # Cartesian 이미지 시각화
    if config["visual_debug"]:
        draw_descriptors(cartesian_image, radon_image, sinofft, rowkey)

    return sinofft, rowkey


def generate_radon(radar_data_dir):
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

    theta = np.arange(180)  # Angle for radon transform
    sinoffts = []
    rowkeys = []

    config = {
        "skip_for_fast_eval": 5,
        "visual_debug": 0,
        "polar_downscale_factor": 0.25,
        "cartesian_size": 128,
        "radon_zoom_shape": 128,
    }

    for data_idx, file_name in enumerate(data_names):
        if data_idx % config["skip_for_fast_eval"] != 0:
            continue

        print(" ")
        data_path = os.path.join(radar_data_dir, file_name)

        # 하나의 이미지에 대해 변환 수행
        print(f"processing {data_idx} th data: {data_path} ...")
        sinofft, rowkey = generate_descriptor(data_path, theta, config)

        # 결과 저장
        sinoffts.append(sinofft)
        rowkeys.append(rowkey)  # rowkey의 크기에 맞춰 동적으로 추가

        # Log progress every 100 iterations
        if (data_idx + 1) % 100 == 0:
            message = f"{data_idx + 1} / {num_data} processed."
            print(message)

    return sinoffts, rowkeys


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

    Returns:
        None
    """
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(sinoffts)
    print("Similarity matrix computation completed.")

    print("Plotting similarity matrix...")
    plot_similarity_matrix(
        similarity_matrix, title="FFT-based Cross-Correlation Similarity Matrix"
    )
    print("Plotting completed.")


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
    sinoffts, rowkeys = generate_radon(args.data_dir)

    # 결과를 파일로 저장하거나 처리할 수 있습니다. 여기서는 간단히 로그를 출력합니다.
    print(f"Processed {len(sinoffts)} images.")

    # 평가 수행
    evaluate(sinoffts)


if __name__ == "__main__":
    main()
