#ifndef CUDA_TEXTURE_CUH
#define CUDA_TEXTURE_CUH
#include <functional>
#include <iostream>

#include "vec2.cuh"


template<typename T>
class CudaTexture
{
    T *m_textureArr;
    Vec2<int> m_size;

public:
    __device__ CudaTexture(T *textureArr, Vec2<int> size)
        : m_textureArr(textureArr)
        , m_size(size)
    {}

    __device__ __inline__ T& operator[](size_t idx)
    {
        return m_textureArr[idx];
    }
    __device__ __inline__ const T& operator[](size_t idx) const
    {
        return m_textureArr[idx];
    }

    __device__ __inline__ T& operator[](const Vec2<int> &textureIndex)
    {
        return m_textureArr[textureIndex.y * m_size.x + textureIndex.x];
    }
    __device__ __inline__ const T& operator[](const Vec2<int> &textureIndex) const
    {
        return m_textureArr[textureIndex.y * m_size.x + textureIndex.x];
    }

    __device__ __inline__ const Vec2<int> &size() const { return m_size; }

};

template<typename T>
class CudaTextureHost
{
    CudaTexture<T> *m_texture;
    T *m_rawCudaPointer;
    std::vector<std::function<void()>> m_onUpdateCallbacks;
    int m_width;
    int m_height;

public:
    CudaTextureHost()
        : m_texture(nullptr)
        , m_rawCudaPointer(nullptr)
        , m_width(0)
        , m_height(0)
    {}

    void onUpdateCallback(const std::function<void()> &callback)
    {
        m_onUpdateCallbacks.push_back(callback);
    }

    void update() const
    {
        for (auto &callback : m_onUpdateCallbacks)
            callback();
    }

    ~CudaTextureHost()
    {
        if (m_rawCudaPointer != nullptr)
            if (const cudaError_t err = cudaFree(m_rawCudaPointer); err != cudaSuccess)
                // Handle memory allocation failure
                std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;

        if (m_texture != nullptr)
            if (const cudaError_t err = cudaFree(m_texture); err != cudaSuccess)
                std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
    }

    void initialize(const int width, const int height)
    {
        m_width = width;
        m_height = height;
        if (const cudaError_t err= cudaMalloc(&m_rawCudaPointer, width * height * sizeof(T)); err != cudaSuccess)
        {
            // Handle memory allocation failure
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
            m_rawCudaPointer = nullptr;  // Ensure m_textureArr remains nullptr on failure
        }

        // Allocate device memory for the CudaTexture<T> object
        if (cudaMalloc(&m_texture, sizeof(CudaTexture<T>)) != cudaSuccess)
        {
            std::cerr << "CUDA malloc failed for texture object!" << std::endl;
            cudaFree(m_texture);
            return;
        }

        // Construct the texture object directly in device memory
        CudaTexture<T> h_texture(m_rawCudaPointer, Vec2(m_width, m_height));
        cudaMemcpy(m_texture, &h_texture, sizeof(CudaTexture<T>), cudaMemcpyHostToDevice);
    }

    T *getPointer() const { return m_rawCudaPointer;}

    CudaTexture<T> *deviceTexture() { return m_texture; }

};

#endif //CUDA_TEXTURE_CUH
