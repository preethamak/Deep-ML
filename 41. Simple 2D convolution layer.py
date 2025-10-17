import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 1. Pad the input
    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')

    # 2. Compute output dimensions
    out_height = (padded_input.shape[0] - kernel_height) // stride + 1
    out_width = (padded_input.shape[1] - kernel_width) // stride + 1
    output_matrix = np.zeros((out_height, out_width))

    # 3. Slide the kernel
    for i in range(out_height):
        for j in range(out_width):
            start_i = i * stride
            start_j = j * stride
            window = padded_input[start_i:start_i + kernel_height, start_j:start_j + kernel_width]
            output_matrix[i, j] = np.sum(window * kernel)

    return output_matrix
