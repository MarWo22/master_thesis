//
// Created by marti on 16/07/2025.
//

#ifndef TEXTUREMANAGER_CUH
#define TEXTUREMANAGER_CUH
#include <memory>
#include <unordered_map>

#include "types/cuda_texture.cuh"


class TextureManager
{
    struct AllocatedTexture
    {
        void *deviceTexturePtr{};
        void *rawCudaPtr{};
        size_t sizeInBytes{};
    };

    struct PairHash
    {
        size_t operator()(const std::pair<int, int> &p) const noexcept
        {
            return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second) << 1;
        }
    };

    typedef std::unordered_map<std::pair<int, int>, std::vector<AllocatedTexture>, PairHash> textureMap_t;
    typedef std::unordered_map<size_t, textureMap_t> sizedTextureMap_t;

    sizedTextureMap_t m_sizedTextureMap;
    inline static size_t s_allocatedMemory = 0;

public:
    template<typename T>
    std::unique_ptr<CudaTextureHost<T> > generateTexture(unsigned int width, unsigned int height)
    {
        const size_t size = sizeof(T);

        textureMap_t &textureMap = m_sizedTextureMap[size];

        std::vector<AllocatedTexture> &allocatedTextures = textureMap[{width, height}];

        if (allocatedTextures.empty())
        {
            s_allocatedMemory += sizeof(CudaTexture<T>) + width * height * sizeof(T);
            return std::make_unique<CudaTextureHost<T> >(
                width,
                height,
                std::bind(&TextureManager::releaseTexture<T>, this, std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3, std::placeholders::_4)
            );
        }

        auto [rawTexture, rawPtr, sizeInBytes] = allocatedTextures.back();
        allocatedTextures.pop_back();


        auto tex = reinterpret_cast<CudaTexture<T> *>(rawTexture);
        auto dataPtr = reinterpret_cast<T *>(rawPtr);

        return std::make_unique<CudaTextureHost<T> >(
            tex,
            dataPtr,
            width,
            height,
            std::bind(&TextureManager::releaseTexture<T>, this, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4)
        );
    }

    template<typename T>
    std::unique_ptr<CudaTextureHost<T> > generateTexture(unsigned int width, unsigned int height, const T &memsetValue)
    {
        const size_t size = sizeof(T);

        textureMap_t &textureMap = m_sizedTextureMap[size];

        std::vector<AllocatedTexture> &allocatedTextures = textureMap[{width, height}];

        if (allocatedTextures.empty())
        {
            s_allocatedMemory += sizeof(CudaTexture<T>) + width * height * sizeof(T);
            return std::make_unique<CudaTextureHost<T> >(
                width,
                height,
                std::bind(&TextureManager::releaseTexture<T>, this, std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3, std::placeholders::_4),
                memsetValue
            );
        }

        auto [rawTexture, rawPtr, sizeInBytes] = allocatedTextures.back();
        allocatedTextures.pop_back();


        auto tex = reinterpret_cast<CudaTexture<T> *>(rawTexture);
        auto dataPtr = reinterpret_cast<T *>(rawPtr);

        return std::make_unique<CudaTextureHost<T> >(
            tex,
            dataPtr,
            width,
            height,
            std::bind(&TextureManager::releaseTexture<T>, this, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4),
            memsetValue
        );
    }


    void freeUnusedTextures()
    {
        {
            for (auto &textureMap: m_sizedTextureMap | std::views::values)
                for (auto &textures: textureMap | std::views::values)
                    for (auto &[texture, rawPtr, sizeInBytes]: textures)
                    {
                        if (rawPtr)
                            if (const cudaError_t err = cudaFree(rawPtr); err != cudaSuccess)
                                std::cerr << "Failed to free rawPtr: " << cudaGetErrorString(err) << '\n';

                        if (texture)
                            if (const cudaError_t err = cudaFree(texture); err != cudaSuccess)
                                std::cerr << "Failed to free texture: " << cudaGetErrorString(err) << '\n';

                        s_allocatedMemory -= sizeInBytes;
                    }
            m_sizedTextureMap.clear();
        }
    }

    static size_t getAllocatedMemory() { return s_allocatedMemory; }

private:
    template<typename T>
    void releaseTexture(CudaTexture<T> *texture, T *rawCudaPtr, int width, int height)
    {
        const size_t size = sizeof(T);

        textureMap_t &textureMap = m_sizedTextureMap[size];
        const size_t sizeInBytes = sizeof(CudaTexture<T>) + width * height * sizeof(T);

        std::vector<AllocatedTexture> &allocatedTextures = textureMap[{width, height}];
        allocatedTextures.push_back({texture, rawCudaPtr, sizeInBytes});
    }
};

template<typename T>
void copyAndReleaseTexture(const CudaTextureHost<T> &dst, std::unique_ptr<CudaTextureHost<T> > src, const int height,
                           const int width)
{
    if (const cudaError_t err = cudaMemcpy(dst.getPointer(), src->getPointer(),
                                           sizeof(T) * height * width,
                                           cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy texture: " << cudaGetErrorString(err) << std::endl;
}

template<typename T>
void copyTexture(const CudaTextureHost<T> &dst, const CudaTextureHost<T> &src, const int height, const int width)
{
    if (const cudaError_t err = cudaMemcpy(dst.getPointer(), src.getPointer(),
                                           sizeof(T) * height * width,
                                           cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy texture: " << cudaGetErrorString(err) << std::endl;
}


#endif //TEXTUREMANAGER_CUH
