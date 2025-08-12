#ifndef CUDA_TEXTURE_CUH
#define CUDA_TEXTURE_CUH
#include <functional>
#include <iostream>
#include <memory>

#include "vec2.cuh"


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

    __device__ __inline__ T &operator[](size_t idx)
    {
        return m_textureArr[idx];
    }

    __device__ __inline__ const T &operator[](size_t idx) const
    {
        return m_textureArr[idx];
    }

    __device__ __inline__ T &operator[](const Vec2<int> &textureIndex)
    {
        return m_textureArr[coordinateToIndex(textureIndex)];
    }

    __device__ __inline__ const T &operator[](const Vec2<int> &textureIndex) const
    {
        return m_textureArr[coordinateToIndex(textureIndex)];
    }

    __device__ __inline__ T operator[](const Vec2<float> &textureIndex) const
    {
        return interpolate(textureIndex, this);
    }


    __device__ __inline__ const Vec2<int> &size() const { return m_size; }

    __device__ __inline__ Vec2<int> indexToCoordinate(const size_t idx) const
    {
        return {static_cast<int>(idx % m_size.x), static_cast<int>(idx / m_size.x)};
    }

    __device__ __inline__ size_t coordinateToIndex(const Vec2<int> &coordinate) const
    {
        const Vec2<int> coord = wrapCoordinate(coordinate);

        return coord.y * m_size.x + coord.x;
    }

    __device__ __inline__ size_t coordinateToIndexUnwrapped(const Vec2<int> &coordinate) const
    {
        return coordinate.y * m_size.x + coordinate.x;
    }

    __device__ __inline__ Vec2<int> wrapCoordinate(const Vec2<int> &coordinate) const
    {
        const int rX = coordinate.x % m_size.x;
        const int rY = coordinate.y % m_size.y;

        const int x = rX + (m_size.x & rX >> 31);
        const int y = rY + (m_size.y & rY >> 31);

        return {x, y};
    }

    __device__ __inline__ float Slope(const Vec2<int> &textureIndex)
    {
        int a = coordinateToIndex(textureIndex + Vec2(1, 0));
        int b = coordinateToIndex(textureIndex + Vec2(-1, 0));
        int c = coordinateToIndex(textureIndex + Vec2(0, 1));
        int d = coordinateToIndex(textureIndex + Vec2(0, -1));

        const float dzdx = (m_textureArr[a] - m_textureArr[b]) / 2.0;
        const float dzdy = (m_textureArr[c] - m_textureArr[d]) / 2.0;

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

    bool m_isManaged;
    std::function<void(CudaTexture<T> *, T *, int, int)> m_onRelease;

public:
    using ManagedCallback = std::function<void(CudaTexture<T> *, T *, int, int)>;


    CudaTextureHost()
        : m_texture(nullptr)
          , m_rawCudaPointer(nullptr)
          , m_width(0)
          , m_height(0)
          , m_isManaged(false)
    {}

    static std::unique_ptr<CudaTextureHost> createManaged(const int width, const int height,
                                                          ManagedCallback onReleaseCallback, const bool async = false)
    {
        std::unique_ptr<CudaTextureHost> ptr = std::unique_ptr<CudaTextureHost>(new CudaTextureHost(onReleaseCallback));
        ptr->initialize(width, height, async);

        return ptr;
    }

    static std::unique_ptr<CudaTextureHost> createManaged(CudaTexture<T> *deviceTexture, T *rawCudaPtr, const int width,
                                                          const int height, ManagedCallback onReleaseCallback)
    {
        return std::unique_ptr<CudaTextureHost>(
            new CudaTextureHost(deviceTexture, rawCudaPtr, width, height, onReleaseCallback));
    }

    static std::unique_ptr<CudaTextureHost> createManagedAndClear(const int width, const int height,
                                                                  ManagedCallback onReleaseCallback,
                                                                  const int memsetValue, const bool async = false)
    {
        std::unique_ptr<CudaTextureHost> ptr = std::unique_ptr<CudaTextureHost>(new CudaTextureHost(onReleaseCallback));
        ptr->initializeAndClear(width, height, memsetValue, async);

        return ptr;
    }

    static std::unique_ptr<CudaTextureHost> createManagedAndClear(CudaTexture<T> *deviceTexture, T *rawCudaPtr,
                                                                  const int width, const int height,
                                                                  ManagedCallback onReleaseCallback,
                                                                  const int memsetValue, const bool async = false)
    {
        std::unique_ptr<CudaTextureHost> ptr = std::unique_ptr<CudaTextureHost>(new CudaTextureHost(
            deviceTexture, rawCudaPtr, width, height, onReleaseCallback));
        ptr->memsetTexture(memsetValue, async);

        return ptr;
    }

    ~CudaTextureHost()
    {
        if (!m_isManaged)
            free();
        else
            m_onRelease(m_texture, m_rawCudaPointer, m_width, m_height);
    }

    [[nodiscard]] int width() const { return m_width; }
    [[nodiscard]] int height() const { return m_height; }

    void initialize(const int width, const int height, const bool async = false)
    {
        m_width = width;
        m_height = height;

        allocate(async);
        constructAndCopyTextureObj(async);
    }

    void memsetTexture(const int memsetValue, const bool async)
    {
        if (async)
        {
            if (const cudaError_t err = cudaMemsetAsync(m_rawCudaPointer, memsetValue, m_width * m_height * sizeof(T));
                err != cudaSuccess)
                std::cerr << "Error memset cuda texture: " << cudaGetErrorString(err) << std::endl;
        } else
        {
            if (const cudaError_t err = cudaMemset(m_rawCudaPointer, memsetValue, m_width * m_height * sizeof(T));
                err != cudaSuccess)
                std::cerr << "Error memset cuda texture: " << cudaGetErrorString(err) << std::endl;
        }
    }

    void initializeAndClear(const int width, const int height, const int memsetValue, const bool async = false)
    {
        m_width = width;
        m_height = height;

        allocate(async);
        memsetTexture(memsetValue, async);
        constructAndCopyTextureObj(async);
    }

    T *getPointer() { return m_rawCudaPointer; }
    const T* getPointer() const { return m_rawCudaPointer; }

    CudaTexture<T> *deviceTexture() { return m_texture; }
    const CudaTexture<T> *deviceTexture() const { return m_texture; }

private:
    CudaTextureHost(CudaTexture<T> *deviceTexture, T *rawCudaPtr, const int width, const int height,
                    ManagedCallback onReleaseCallback)
        : m_texture(deviceTexture)
          , m_rawCudaPointer(rawCudaPtr)
          , m_width(width)
          , m_height(height)
          , m_isManaged(true)
          , m_onRelease(onReleaseCallback)

    {}

    explicit CudaTextureHost(ManagedCallback onReleaseCallback)
        : m_texture(nullptr)
          , m_rawCudaPointer(nullptr)
          , m_width(0)
          , m_height(0)
          , m_isManaged(true)
          , m_onRelease(onReleaseCallback)

    {}

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

    void constructAndCopyTextureObj(const bool async)
    {
        // Construct the texture object directly in device memory
        CudaTexture<T> h_texture(m_rawCudaPointer, Vec2(m_width, m_height));

        if (async)
        {
            if (const cudaError_t err = cudaMemcpyAsync(m_texture, &h_texture, sizeof(CudaTexture<T>),
                                                        cudaMemcpyHostToDevice); err != cudaSuccess)
                std::cerr << "Error memcpy cuda texture: " << cudaGetErrorString(err) << std::endl;
        } else
        {
            if (const cudaError_t err = cudaMemcpy(m_texture, &h_texture, sizeof(CudaTexture<T>),
                                                   cudaMemcpyHostToDevice); err != cudaSuccess)
                std::cerr << "Error memcpy cuda texture: " << cudaGetErrorString(err) << std::endl;
        }
    }

    void allocate(const bool malloc_async)
    {
        if (malloc_async)
        {
            if (const cudaError_t err = cudaMallocAsync(&m_rawCudaPointer, m_width * m_height * sizeof(T), nullptr);
                err != cudaSuccess)
            {
                // Handle memory allocation failure
                std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
                m_rawCudaPointer = nullptr; // Ensure m_textureArr remains nullptr on failure
            }

            // Allocate device memory for the CudaTexture<T> object
            if (cudaMallocAsync(&m_texture, sizeof(CudaTexture<T>), nullptr) != cudaSuccess)
            {
                std::cerr << "CUDA malloc failed for texture object!" << std::endl;
                cudaFree(m_texture);
            }
        } else
        {
            if (const cudaError_t err = cudaMalloc(&m_rawCudaPointer, m_width * m_height * sizeof(T));
                err != cudaSuccess)
            {
                // Handle memory allocation failure
                std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
                m_rawCudaPointer = nullptr; // Ensure m_textureArr remains nullptr on failure
            }

            // Allocate device memory for the CudaTexture<T> object
            if (cudaMalloc(&m_texture, sizeof(CudaTexture<T>)) != cudaSuccess)
            {
                std::cerr << "CUDA malloc failed for texture object!" << std::endl;
                cudaFree(m_texture);
            }
        }
    }
};

#endif //CUDA_TEXTURE_CUH
