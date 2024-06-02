import numpy as np

# 生成一个简单的三维张量，假设形状为(10, 20, 30)
tensor = np.arange(10 * 20 * 30).reshape(10, 20, 30)

# 对第二个维度进行切片，只取前十个数据
sliced_tensor = tensor[:, :10, :]

# 选择要打印的切片的索引，这里选择第一个切片
slice_index = 0

# 打印原始张量的某个切片
print("Original Tensor Slice:")
print(tensor[slice_index, :, :])

# 打印切片后的张量的某个切片
print("\nSliced Tensor:")
print(sliced_tensor[slice_index, :, :])
