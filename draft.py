# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 20:57
# @Author  : Liang Jinaye
# @File    : draft.py
# @Description :

import numpy as np

arr = np.array([[1, 5],
                [3, 2],
                [2, 8],
                [4, 3]])

# 对每个子数组按降序排序
sorted_subarrays = np.array([np.max(subarr) for subarr in arr])
sorted_subarrays = np.sort(sorted_subarrays)[::-1]

# 对第一列进行降序排序的索引
# sort_indices = sorted_subarrays.sort()

# 根据索引重新排列数组
# sorted_arr = sorted_subarrays[sort_indices]

print("原始数组:")
print(arr)
print("先对每个子数组降序排序，再对第一列降序排序后的数组:")
print(sorted_subarrays)

