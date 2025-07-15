#ifndef CUDA_TEXTURE_CUH
#define CUDA_TEXTURE_CUH
#include <functional>
#include <iostream>

#include "types/vec2.cuh"


template<typename T>
class CudaTexture
{
    T *m_textureArr;
    Vec2<int> m_size;

public:
    __device__ CudaTexture(T *textureArr, const Vec2<int> size)
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
        return m_textureArr[coordinateToIndex(textureIndex)];
    }
    __device__ __inline__ const T& operator[](const Vec2<int> &textureIndex) const
    {
        return m_textureArr[coordinateToIndex(textureIndex)];
    }
    __device__ __inline__ T operator[](const Vec2<float>& textureIndex) const
    {
        return interpolate(textureIndex, this);
    }

    
    



    __device__ __inline__ const Vec2<int> &size() const { return m_size; }

    __device__ __inline__ Vec2<int> indexToCoordinate(size_t idx) const
    {  
        return Vec2<int>(fmodf(idx, m_size.x), idx / m_size.x);
    }

    __device__ __inline__ size_t coordinateToIndex(const Vec2<int> &coordinate) const
    {  
        Vec2<int> coord = wrapCoordinate(coordinate);

        return  coord.y * m_size.x + coord.x;
    }

    __device__ __inline__ size_t coordinateToIndexUnwrapped(const Vec2<int> &coordinate) const
    {
        return coordinate.y * m_size.x + coordinate.x;
    }

    __device__ __inline__ Vec2<int> wrapCoordinate(const Vec2<int>& coordinate) const
    {
        int rX = coordinate.x % m_size.x;
        int rY = coordinate.y % m_size.y;

        int x = rX + (m_size.x & (rX >> 31));
        int y = rY + (m_size.y & (rY >> 31));

        // int x = fmodf(fmodf(coordinate.x, m_size.x) + m_size.x, m_size.x);
        // int y = fmodf(fmodf(coordinate.y, m_size.y) + m_size.y, m_size.y);

        return Vec2<int>(x,y);
    }

    __device__ __inline__ float Slope(const Vec2<int> &textureIndex) {
        
        int a = coordinateToIndex(textureIndex + Vec2<int>(1, 0));
        int b = coordinateToIndex(textureIndex + Vec2<int>(-1,0));
        int c = coordinateToIndex(textureIndex + Vec2<int>(0, 1));
        int d = coordinateToIndex(textureIndex + Vec2<int>(0,-1));

        float dzdx = (m_textureArr[a] - m_textureArr[b]) / 2.0;
        float dzdy = (m_textureArr[c] - m_textureArr[d]) / 2.0;

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

    ~CudaTextureHost()
    {
        free();
    }

    [[nodiscard]] int width() const { return m_width; }
    [[nodiscard]] int height() const { return m_height; }

    void free()
    {
        if (m_rawCudaPointer != nullptr)
        {
            if (const cudaError_t err = cudaFree(m_rawCudaPointer); err != cudaSuccess)
                // Handle memory allocation failure
                    std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
            m_rawCudaPointer = nullptr;
        }

        if (m_texture != nullptr)
        {
            if (const cudaError_t err = cudaFree(m_texture); err != cudaSuccess)
                std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
            m_texture = nullptr;
        }
    }

    void initialize(const int width, const int height)
    {
        m_width = width;
        m_height = height;

        allocate();

        // Construct the texture object directly in device memory
        CudaTexture<T> h_texture(m_rawCudaPointer, Vec2(m_width, m_height));
        cudaMemcpy(m_texture, &h_texture, sizeof(CudaTexture<T>), cudaMemcpyHostToDevice);
    }

    void initialize(const int width, const int height, const T &memsetValue)
    {
        m_width = width;
        m_height = height;

        allocate();
        if (const cudaError_t err = cudaMemset(m_rawCudaPointer, memsetValue, m_width * m_height * sizeof(T)); err != cudaSuccess)
            std::cerr << "Error memset cuda texture: " << cudaGetErrorString(err) << std::endl;

        // Construct the texture object directly in device memory
        CudaTexture<T> h_texture(m_rawCudaPointer, Vec2(m_width, m_height));
        cudaMemcpy(m_texture, &h_texture, sizeof(CudaTexture<T>), cudaMemcpyHostToDevice);
    }

    T *getPointer() const { return m_rawCudaPointer;}

    CudaTexture<T> *deviceTexture() { return m_texture; }
    const CudaTexture<T> *deviceTexture() const { return m_texture; }



private:
    void allocate()
    {
        if (const cudaError_t err= cudaMalloc(&m_rawCudaPointer, m_width * m_height * sizeof(T)); err != cudaSuccess)
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
        }
    }
};

#endif //CUDA_TEXTURE_CUH
