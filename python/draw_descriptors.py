import matplotlib.pyplot as plt


def draw_descriptors(cartesian_image, radon_image, sinofft):
    # 시각화
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cartesian_image, cmap="gray")
    plt.title("Cartesian Image from Polar Coordinates")
    plt.colorbar()

    # 1. Radon 이미지 시각화
    plt.subplot(1, 3, 2)
    plt.imshow(radon_image, cmap="gray", aspect="auto")
    plt.title("Radon Transform Image")
    plt.colorbar()

    # 2. sinofft 시각화 (FFT 결과)
    plt.subplot(1, 3, 3)
    plt.imshow(sinofft, cmap="gray", aspect="auto")
    plt.title("Sinofft (FFT of Radon Image)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
