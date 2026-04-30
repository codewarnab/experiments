import numpy as np
import matplotlib.pyplot as plt

def conv2d_numpy(image, kernel):
    H, W = image.shape
    k = kernel.shape[0]

    out_H = H - k + 1
    out_W = W - k + 1
    output = np.zeros((out_H, out_W), dtype=np.float32)

    for i in range(out_H):
        for j in range(out_W):
            patch = image[i:i+k, j:j+k]
            output[i, j] = np.sum(patch * kernel)

    return output


# Better test image: a clear white square
img = np.zeros((100, 100), dtype=np.float32)
img[30:70, 40:60] = 1.0


vertical_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

horizontal_kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

blur_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16.0

sobel_kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])


vertical_edges = conv2d_numpy(img, vertical_kernel)
horizontal_edges = conv2d_numpy(img, horizontal_kernel)
blur = conv2d_numpy(img, blur_kernel)
sobel = conv2d_numpy(img, sobel_kernel)


def show_image(ax, image, title):
    ax.imshow(image, cmap="gray", vmin=image.min(), vmax=image.max())
    ax.set_title(title, fontsize=12)
    ax.axis("off")


plt.figure(figsize=(10, 8))

ax1 = plt.subplot(2, 3, 1)
show_image(ax1, img, "Original Image")

ax2 = plt.subplot(2, 3, 2)
show_image(ax2, vertical_edges, "Vertical Edges")

ax3 = plt.subplot(2, 3, 3)
show_image(ax3, horizontal_edges, "Horizontal Edges")

ax4 = plt.subplot(2, 3, 4)
show_image(ax4, blur, "Blurred")

ax5 = plt.subplot(2, 3, 5)
show_image(ax5, sobel, "Sobel Edges")

plt.tight_layout()
plt.show()