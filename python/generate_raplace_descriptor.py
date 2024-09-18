import numpy as np

from scipy.fft import fft
from scipy.ndimage import zoom
from skimage.transform import radon

from PIL import Image

from draw_descriptors import draw_descriptors
from polar_to_cartesian import polar_to_cartesian_fast

from tic import tic


@tic
def radon_transform(cartesian_image, theta, circle=False):
    return radon(cartesian_image, theta=theta, circle=False)


@tic
def radon_to_sinogram(radon_image):
    sinofft = np.abs(fft(radon_image, axis=0))
    sinofft = sinofft[: sinofft.shape[0] // 2, :]
    return sinofft


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


@tic
def generate_raplace_descriptor(image_path, theta, config):
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
    sinofft = radon_to_sinogram(radon_image)
    print(f" sinofft of size {sinofft.shape} is made.")

    # Cartesian 이미지 시각화
    if config["visual_debug"]:
        draw_descriptors(cartesian_image, radon_image, sinofft)

    return sinofft
