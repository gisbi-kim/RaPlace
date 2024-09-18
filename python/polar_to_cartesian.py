import numpy as np 
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

from tic import tic 


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
def polar_to_cartesian_naive_slow(polar_image, cartesian_size, polar_pixel_size):
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



