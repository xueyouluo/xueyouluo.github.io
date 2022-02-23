---
layout: post
title: "一道coding题：上三角矩阵的快速索引"
date: 2022-02-23
excerpt: "上三角矩阵快速索引"
tags: [matrix,coding]
comments: true
---


# 上三角矩阵的快速索引

这个是在做biaffine建模的时候遇到的一个问题，感觉可以作为一个leetcode的中等题，有兴趣的同学可以想想看。

## 问题定义

给定一个上三角矩阵如下所示：

$$
\begin{matrix}
1 & 2 & 3 & 4 & 5 \\
0 & 6 & 7 & 8 & 9 \\
0 & 0 & 10 & 11 & 12 \\
0 & 0 & 0 & 13 & 14 \\
0 & 0 & 0 & 0 & 15 \\
\end{matrix}
$$

> 它是一个`N*N`大小的矩阵。

我们可以把这个矩阵拍平，只保留上三角的部分，这样就得到：`nums = [1,2,3,4,5,6,7,8,9,10,...,15]`。这里我们约定下标都是从0开始的。

问题1：给定矩阵索引(s,e)如何快速定位其在数组中的索引k？

> 比如给定矩阵索引`[1,1]`，那么它对应值在nums数组中的索引应该是5。

问题2：给定数组的索引k,如何快速定位到矩阵的索引(s,e)？

> 就是上面的问题反过来，给定数组索引5，我们要返回它在矩阵的索引`[1,1]`


## 问题解法

### 问题1

问题1其实挺简单的，矩阵的每行的数据量分别为`n,n-1,...,1`。那么给定`(s,e)`，如果不是上三角矩阵，那么应该是`s*n+e`这个位置，但是要去掉下三角中0的个数，有多少个呢？应该是`0 + 1 + 2 + ... + s`个，所以最终的计算方法：

$$
position = s * n + e - s*(s+1)/2
$$

### 问题2

从问题1可以看到，只给出k是无法反推两个参数s和e的，那么我们简单变换一下
$$
e = position - s * n + s*(s+1)/2
$$
因此我们只需要定位出s，那么就可以知道e了，如何快速定位s呢？

我们设计一个前缀和数组`prefix_sum = [n,2n-1,3n-3,...,n*(n+1)/2]`，那么问题就转化为如何在一个排序的数组中快速找到第一个>=k的下标？典型的二分查找问题了。

```python
def get_position(n,k):
  nums = reversed(list(range(n)))
  for i in range(1,n):
    nums[i] += nums[i-1]
  
  left,right=0,n
  while left < right:
    mid = (left + right) // 2
    if nums[mid] < k:
      left = mid + 1
    else:
      right = mid
      
  s = left
  e = k - s * n + s * (s + 1) // 2
  return (s,e)
```

这里用前缀数组的话还是要占用$O(N)$的空间的，而且整体的时间复杂度也是$O(N)$。

因为我们知道这是一个等差数列，更优化的方法是直接用求和公式来计算前缀和，使得总体的时间复杂度降为$O(logN)$。

改进一下：

```python
def get_position(n,k):
  def prefix_sum(i):
    return (n + n - i) * (i + 1) // 2
  
  left,right=0,n
  while left < right:
    mid = (left + right) // 2
    if nums[mid] < k:
      left = mid + 1
    else:
      right = mid
      
  s = left
  e = k - s * n + s * (s + 1) // 2
  return (s,e)
```

但是，如果你的矩阵输入的N大小是固定的，那么就可以用空间换时间了，直接建立(s,e)->k的映射，到时候查表即可。虽然空间复杂度是$O(N^2)$，但是查询是`O(1)`。

```python
N = 100
mp = {}
for i in range(N):
  for j in range(i,N):
    mp[len(mp)] = (i,j)

def get_position(k):
  return mp[k]

```






