import os
import argparse
import numpy as np
from scipy.fftpack import fft
from scipy.ndimage import zoom
from skimage.transform import radon
from PIL import Image
import matplotlib.pyplot as plt

from tic import tic


@tic
def polar_to_cartesian(polar_image, cartesian_size, polar_pixel_size):
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
    plt.imshow(radon_image, cmap='gray', aspect='auto')
    plt.title("Radon Transform Image")
    plt.colorbar()

    # 2. sinofft 시각화 (FFT 결과)
    plt.subplot(1, 4, 3)
    plt.imshow(sinofft, cmap='gray', aspect='auto')
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
    downsized_polar_image = downsize_polar_image(polar_image, config["polar_downscale_factor"])
    # 다운사이즈 후의 픽셀당 거리 계산
    downsized_polar_pixel_size = original_polar_pixel_size / config["polar_downscale_factor"]

    # Polar to Cartesian 변환
    cartesian_image = polar_to_cartesian(
        downsized_polar_image, 
        config["cartesian_size"], 
        downsized_polar_pixel_size
    )
    print(f" cartesian_image of size {cartesian_image.shape} is made.")

    # Radon transform
    radon_image = radon_transform(cartesian_image, theta=theta, circle=False)
    print(f" radon_image (raw) of size {radon_image.shape} is made.")

    radon_image = radon_image / np.max(radon_image)
    radon_image = zoom(
        radon_image,
        (config["radon_zoom_shape"] / radon_image.shape[0], 
         config["radon_zoom_shape"] / radon_image.shape[1]),
    )
    print(f" radon_image (zoomed) of size {radon_image.shape} is made.")

    # FFT and process the result
    sinofft = np.abs(fft(radon_image, axis=0))
    sinofft = sinofft[: sinofft.shape[0] // 2, :]

    # Flatten the Radon transform result for rowkey
    rowkey = radon_image.flatten()

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
        "visual_debug": True,
        "polar_downscale_factor": 0.25,
        "cartesian_size": 128,
        "radon_zoom_shape": 128,
    }

    skip = 10
    for data_idx, file_name in enumerate(data_names):
        if data_idx % skip !=0:
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


if __name__ == "__main__":
    main()
