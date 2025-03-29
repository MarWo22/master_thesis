#ifndef CUDA_TEXTURE_CUH
#define CUDA_TEXTURE_CUH
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

    __device__ __inline__ const T& operator[](const Vec2<float>& textureCoordinate) const
    {
        int x0 = floorf(textureCoordinate.x);
        int x1 = x0 + 1;
        int y0 = floorf(textureCoordinate.y);
        int y1 = y0 + 1;

        float weightX = textureCoordinate.x - x0;
        float weightY = textureCoordinate.y - y0;

        float I0 = (1 - weightX) * this[Vec2<int>(x0, y0)] + weightX * this[Vec2<int>(x1, y0);
        float I1 = (1 - weightX) * this[Vec2<int>(x0, y1) + weightX * this[Vec2<int>(x1, y1);
        return (1 - weightY) * I0 + weightY * I1;         //Keep in mind. I0 and I1 might need to be swapped !!!!!!!
    }



    __device__ __inline__ const Vec2<int> &size() const { return m_size; }

    __device__ const Vec2<int> indexToCoordinate(size_t idx) 
    {  
        return Vec2<int>(fmodf(idx, m_size.x), idx / m_size.x);
    }

    __device__ const size_t coordinateToIndex(Vec2<int> coordinate) 
    {  
        x = ((coordinate.x % m_size.x) + m_size.x) % m_size.x;
        y = ((coordinate.y % m_size.y) + m_size.y) % m_size.y;

        return  y * width + x;
    }

    __device__ const float Slope(Vec2<int>& textureIndex) {
        double dzdx = (this[textureIndex + Vec2<int>(1,0)] - this[textureIndex + Vec2<int>(-1,0)) / 2.0;
        double dzdy = (this[textureIndex + Vec2<int>(0,1)] - this[textureIndex + Vec2<int>(0,-1)]) / 2.0;

        return std::sqrt(dzdx * dzdx + dzdy * dzdy);
    }

};

template<typename T>
class CudaTextureHost
{
    CudaTexture<T> *m_texture;
    T *m_rawCudaPointer;
    int m_width;
    int m_height;

public:
    CudaTextureHost()
        : m_texture(nullptr)
        , m_rawCudaPointer(nullptr)
        , m_width(0)
        , m_height(0)
    {}

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
