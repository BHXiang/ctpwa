// cpu_shims/cub/cub.cuh — CPU 后端垫片
//
// 全库只 include <cub/cub.cuh> 而没有 cub:: 调用（块级归约在 CPU 后端由
// T2/T3 kernel 的显式循环替代），因此这里是空文件。
#pragma once
