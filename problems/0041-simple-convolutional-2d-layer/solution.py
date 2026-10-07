import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# first we need to pad the input matrix with padding
	if padding > 0:
		padded_input = np.pad(input_matrix, pad_width = ((padding,padding),(padding,padding)), mode="constant", constant_values=0)
	else:
		padded_input = input_matrix

	# then we need to allocate the output size = x+2*padding // kernel height , y + 2*padding//kernel_width
	output_height = (input_height + 2*padding - kernel_height)//stride +1 
	output_width = (input_width + 2*padding - kernel_width)//stride +1 

	output_matrix = np.zeros((output_height, output_width))

	# we do matrix-matrix dot pordoct and sum
	for row in range(output_height):
		for col in range(output_width):
			row_start = row * stride
			col_start = col * stride

			roi = padded_input[row_start:row_start+kernel_height, col_start : col_start+kernel_width]
			output_matrix[row, col] = np.sum(roi * kernel)

	return output_matrix
