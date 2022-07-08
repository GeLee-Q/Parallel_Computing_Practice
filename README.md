- [**CMake 工程构建**](#cmake-工程构建)
- [康威生命游戏优化](#康威生命游戏优化)
  - [代码环境](#代码环境)
  - [实验优化记录](#实验优化记录)
  - [优化**方法总结**](#优化方法总结)
- [Mylib](#mylib)
  - [ticktock.h | bate.h | mtprint.h](#ticktockh--bateh--mtprinth)
  - [snode.h](#snodeh)
  - [pod.h](#podh)
  - [alignalloc.h | alignalloc_msvc.h](#alignalloch--alignalloc_msvch)
  - [ndarray.h | ndarray_msvc.h](#ndarrayh--ndarray_msvch)

# **CMake 工程构建**

- src        源码文件
- mylib     包含各种功能的头文件
- sparse_data_structure 稀疏数据结构案例代码
- 顶层CMakeLists.txt 构造项目

# 康威生命游戏优化

## 代码环境

> WSL + GCC 9.2

> src/Conway_Game_of_Life.cpp
## 实验优化记录

- 原始代码未开启openMP

```
134.008686s
```

- 原始代码开启openMP

```
49.100373s        加速：2.72x
```

- 使用指针数组这种稀疏数据结构来优化

```
OpenMp  29.86s    加速：4,48x
tbb     17.48s    加速：7.66x
```

- 在指针数组加上spin_mutex优化，效果不明显

```
 openMP 33.47s    加速：4.06x
 tbb    17.76s    加速：7.54x
```

## 优化**方法总结**

**封装稀疏网格的 Grid 类的？**

- 采用指针数组封装稀疏的Grid表格
- 为了减轻mutex的系统开销，使用了spin_mutex自旋锁
- 如果使用hash().pointer().dense的话，在WSL上回出现 out of memory

**使用位运算量化减轻内存带宽？**

- & 替代 %， >> 替代 /， | 替代 +

**使用并行访问访问库进行优化？**

- 使用OpenMP进行并行访问。

**有没有用访问者模式缓存坐标，避免重复上锁？**

- 没有使用, 在WSL里使用这个，会莫名出现segment fault

**`step()` 函数中这种插桩方式优化**

- 将step()改造成了tbb并行。






# Mylib

## ticktock.h | bate.h | mtprint.h

- 定义宏获取时间
- 基于 `std::chrono`

- frand() 获取随机数
- `mtprint.h` 用于打印多维数组

## snode.h

- 三层嵌套稀疏数据结构
- BUG：会引起内存爆炸，待修改

## pod.h

- 帮手类

- vector的初始化，写入了0, 即为一次分配，一次写入。
- 让数据结构不会初始化写入，相乘malloc一样的惰性初始化。

## alignalloc.h | alignalloc_msvc.h

- 指定对齐到任意字节
- `std::vector<int, AlignedAllocator<int, 4096>> arr(n);`
- `alignalloc_msvc.h`  适配msvc的版本  ： `aligned_alloc `并未被MSVC支持，但改用其 `_aligned_malloc`即可；

## ndarray.h | ndarray_msvc.h

- ZYX 序：(z * ny + y) * nx + x , 避免手动扁平化高维数组，封装成类
-  ndarray<2, float>  二维浮点数组，ndarray<3, int> 三维整型数组。
- 解决访问越界问题，增加额外的参数，控制边界层的大小

​	`consexptr int nblur = 8 ;  ndarray<2, float, nblur> a(nx, ny);`

- 有些 SIMD 指令要求地址对齐到一定字节数，否则会 segfault，如 _mm256_stream_ps 需要对齐到 32 字节。因此使用 AlignedAllocator 让 vector 的内存分配，始终对齐到 64 字节（缓存行）

​		`ndarray<2,float, 0 ,0 ,32> a(nx, ny)`

- MSVC 的模板元编程偏差，需要修改模板

```c++
 // template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && std::conjunction_v<std::is_integral<Ts>...> , int> = 0>
```