import matplotlib.pyplot as plt

is_save_file = True
save_idx = 0

def draw_descriptors(cartesian_image, radon_image, sinofft):
    global save_idx  # Declare that save_idx is a global variable
    global is_save_file

    # 시각화
    plt.figure(figsize=(18, 5))

    # 0. input, downsized cart image 시각화
    plt.subplot(1, 3, 1)
    plt.imshow(cartesian_image, cmap="gray")
    plt.title("Cartesian Image from Polar Coordinates")
    plt.colorbar()
    plt.clim(0, 100)

    # 1. Radon 이미지 시각화
    plt.subplot(1, 3, 2)
    plt.imshow(radon_image, cmap="gray", aspect="auto")
    plt.title("Radon Transform Image")
    plt.colorbar()
    plt.clim(0, 1)
    
    # 2. sinofft 시각화 (FFT 결과)
    plt.subplot(1, 3, 3)
    plt.imshow(sinofft, cmap="gray", aspect="auto")
    plt.title("Sinofft (FFT of Radon Image)")
    plt.colorbar()
    plt.clim(0, 2)
    
    plt.tight_layout()
    
    # Save the figure as a PNG file
    if is_save_file:
        plt.savefig(f'python/visual_debug/{save_idx}.png', format='png', dpi=300)  # dpi=300 ensures high resolution
        save_idx += 1
    else:
        plt.show()