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
- [memory_optimization](#memory_optimization)
  - [xy_yx_loop.cpp](#xy_yx_loopcpp)
  - [matrix_alloc.cpp](#matrix_alloccpp)
  - [cache_skip.cpp](#cache_skipcpp)
  - [aosoa.cpp](#aosoacpp)
  - [prefetch.cpp](#prefetchcpp)
  - [write_read.cpp | write_read_msvc.cpp](#write_readcpp--write_read_msvccpp)

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



# memory_optimization

https://quick-bench.com/ benchmark在线测试网站

## xy_yx_loop.cpp

- 验证xy序 和 yx序

## matrix_alloc.cpp

- STL容器的多维数组alloc 和 展开的多维数组alloc数组速度比较

## cache_skip.cpp

- 验证缓存的读写机制

## aosoa.cpp

- AOS（Array of Struct）单个对象的属性紧挨着存
- SOA（Struct of Array）属性分离存储在多个数组
-  MyClass 内部是 SOA，外部仍是一个 `vector<MyClass>` 的 AOS——这种内存布局称为 AOSOA。
- **如果几个属性几乎总是同时一起用的**，比如位置矢量pos的xyz分量，可能都是同时读取同时修改的，这时用**AOS**，减轻预取压力。
- **如果几个属性有时只用到其中几个，不一定同时写入**，这时候就用**SOA**比较好，省内存带宽。
- **AOSOA**：在高层保持AOS的统一索引，底层又享受SOA带来的矢量化和缓存行预取等好处

## prefetch.cpp

> 解决随机的预取问题

- 缓存行预取技术：由硬件自动识别程序的访存规律，决定要预取的地址。一般来说只有线性的地址访问规律（包括顺序、逆序；连续、跨步）能被识别出来，而**如果访存是随机的，那就没办法预测**。
- 为了解决随机访问的问题，把分块的大小调的更大一些，比如 4KB 那么大，即64个缓存行，而不是一个。每次随机出来的是块的位置。
- 预取不能跨越页边界，否则可能会触发不必要的 page fault。所以选取页的大小，因为本来就不能跨页顺序预取，所以被我们切断掉也无所谓。

## write_read.cpp | write_read_msvc.cpp

> 为写入花的时间比读取慢：写入的粒度太小，浪费了缓存行的代码

- https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

- 绕过缓存，直接写入：`_mm_stream_si32`，代替直接赋值的写入，绕开缓存**，将一个4字节的写入操作，挂起到临时队列，等凑满64字节后，直接写入内存，从而完全避免读的带宽。只支持int做参数，要用float还得转换一下指针类型，bitcast一下参数。

- stream特点：不会读到缓存里， 最好是连续的写入
- 1.只有写入，没有读取 2.之后没有再读取该数组 才应该使用`stream`指令
- `_mm_stream_si32` 可以一次性写入4字节到挂起队列。而 `_mm_stream_ps` 可以一次性写入 16 字节到挂起队列。