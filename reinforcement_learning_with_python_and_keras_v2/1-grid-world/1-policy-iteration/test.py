import numpy as np

value_list = [11, 12, 13, 19]
print(np.amax(value_list))
print(np.argwhere(value_list == np.array([19])))
max_idx_list = np.argwhere(value_list == np.amax(value_list))
max_idx_list = max_idx_list.flatten().tolist()

print(max_idx_list)
