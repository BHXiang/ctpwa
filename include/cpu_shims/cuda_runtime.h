// cpu_shims/cuda_runtime.h — CUDA Runtime API 的 CPU 垫片
//
// 只出现在 CTPWA_BACKEND=cpu 构建的 include 路径最前面（见 setup.py）。
// CUDA 构建完全看不到本文件。
//
// 设计:
//   * 全部 inline，转发到 src/Backend.cpp 的 ctpwa:: 原语——**只有一个实现**。
//   * 语义上"同地址空间"：H2D/D2H/D2D 都退化为 memmove，多卡 peer 拷贝退化为 memcpy。
//   * 不做静默降级：用到本垫片没覆盖的 API 会直接编译失败，失败点即移植点。
//
// 覆盖范围 = 全库实际用到的 28 个 cuda* 符号（见 doc/cpu-backend-plan.md §1.1）。

#pragma once

#include "CudaCompat.h"
#include "Backend.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------- 错误码
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorIllegalAddress = 2;
constexpr cudaError_t cudaErrorMemoryAllocation = 3;

inline const char* cudaGetErrorString(cudaError_t err)
{
    switch (err) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorIllegalAddress: return "illegal address";
        case cudaErrorMemoryAllocation: return "out of memory";
        default: return "unknown CPU-backend error";
    }
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }

// ---------------------------------------------------------------- 拷贝方向
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

using cudaStream_t = void*;

// ---------------------------------------------------------------- 内存
inline cudaError_t cudaMalloc(void** p, std::size_t bytes)
{
    *p = ctpwa::alloc(bytes);
    if (!*p && bytes != 0)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

// CUDA 的 cudaMalloc 是模板重载（接受 T**），这里保持一致，
// 否则 700+ 处 cudaMalloc(&typed_ptr, n) 会因 void** 不兼容而编不过。
template <class T>
inline cudaError_t cudaMalloc(T** p, std::size_t bytes)
{
    *p = static_cast<T*>(ctpwa::alloc(bytes));
    if (!*p && bytes != 0)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* p)
{
    ctpwa::dealloc(p);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* p, int value, std::size_t bytes)
{
    ctpwa::memset(p, value, bytes);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* p, int value, std::size_t bytes,
                                   cudaStream_t /*stream*/ = nullptr)
{
    ctpwa::memset(p, value, bytes);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes,
                              cudaMemcpyKind kind)
{
    ctpwa::copy(dst, src, bytes, kind == cudaMemcpyDeviceToHost ? ctpwa::Dir::D2H
                                                                : ctpwa::Dir::H2D);
    return cudaSuccess;
}

// 多卡 peer 拷贝：同一地址空间，退化为同进程 memcpy
inline cudaError_t cudaMemcpyPeer(void* dst, int /*dst_device*/, const void* src,
                                  int /*src_device*/, std::size_t bytes)
{
    ctpwa::copy(dst, src, bytes, ctpwa::Dir::D2D);
    return cudaSuccess;
}

// ---------------------------------------------------------------- 设备
inline cudaError_t cudaSetDevice(int index)
{
    return ctpwa::set_device(index) ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t cudaGetDevice(int* index)
{
    *index = ctpwa::current_device();
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int* count)
{
    *count = ctpwa::device_count();
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

inline cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes)
{
    *free_bytes = ctpwa::available_memory();
    *total_bytes = ctpwa::total_memory();
    return cudaSuccess;
}

// ---------------------------------------------------------------- 设备属性
enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
};

template <class T>
inline cudaError_t cudaFuncSetAttribute(T* /*entry*/, cudaFuncAttribute /*attr*/,
                                        int /*value*/)
{
    return cudaSuccess;   // CPU 后端没有 shared memory 上限
}

inline cudaError_t cudaFuncSetAttribute(const void* /*entry*/, cudaFuncAttribute /*attr*/,
                                        int /*value*/)
{
    return cudaSuccess;
}

constexpr int cudaDevAttrMaxSharedMemoryPerBlockOptin = 97;

inline cudaError_t cudaDeviceGetAttribute(int* value, int attr, int /*device*/)
{
    // CPU 没有 shared memory 上限：返回一个足够大的值，让调用方的
    // "动态 shared 是否超限"检查恒通过（否则会走报错分支或误降级分块策略）。
    if (attr == cudaDevAttrMaxSharedMemoryPerBlockOptin)
        *value = 1 << 20;   // 1 MiB，远超所有 kernel 的实际需求
    else
        *value = 0;
    return cudaSuccess;
}

struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    std::size_t totalGlobalMem;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int /*device*/)
{
    std::memset(prop, 0, sizeof(*prop));
    std::snprintf(prop->name, sizeof(prop->name), "CPU backend");
    prop->major = 0;
    prop->minor = 0;
    prop->totalGlobalMem = ctpwa::total_memory();
    return cudaSuccess;
}
