#include "ccl.cuh"

#include "cuda_helper.cuh"

// https://www-sciencedirect-com.ezproxy.ub.gu.se/science/article/pii/S0010465515001472?via%3Dihub

__global__ void initializeCCL(const CudaTexture<uint8_t> *idsPtr, CudaTexture<unsigned int> *labelsPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();

    const CudaTexture<uint8_t> &ids = *idsPtr;
    CudaTexture<unsigned int> &labels = *labelsPtr;

    if (!isWithinBounds(invokeIndex, ids.size()))
        return;

    // Get the pixel coordinate
    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, ids.size());

    // ID of the current pixel
    const uint8_t id = ids[invokeIndex];

    // if the top left index is connected, assign that label

    Vec2 neighborTextureIndex = {textureIndex.x - 1, textureIndex.y - 1};
    if (const uint8_t neighborId = ids[neighborTextureIndex]; id == neighborId)
    {
        labels[invokeIndex] = ids.coordinateToIndex(neighborTextureIndex);
        return;
    }

    // if the top index is connected, assign that label

    neighborTextureIndex = {textureIndex.x, textureIndex.y - 1};
    if (const uint8_t neighborId = ids[neighborTextureIndex]; id == neighborId)
    {
        labels[invokeIndex] = ids.coordinateToIndex(neighborTextureIndex);
        return;
    }

    // if the left index is connected, assign that label
    neighborTextureIndex = {textureIndex.x - 1, textureIndex.y};
    if (const uint8_t neighborId = ids[neighborTextureIndex]; id == neighborId)
    {
        labels[invokeIndex] = ids.coordinateToIndex(neighborTextureIndex);
        return;
    }

    labels[invokeIndex] = invokeIndex;
}

__global__ void analysisCCL(const CudaTexture<uint8_t> *idsPtr, CudaTexture<unsigned int> *labelsPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();

    const CudaTexture<uint8_t> &ids = *idsPtr;
    CudaTexture<unsigned int> &labels = *labelsPtr;

    if (!isWithinBounds(invokeIndex, ids.size()))
        return;

    unsigned int label = labels[invokeIndex];
    unsigned int newLabel = labels[label];

    while (label != newLabel)
    {
        label = newLabel;
        newLabel = labels[label];
    }

    labels[invokeIndex] = label;
}

__global__ void labelReductionCCL(const CudaTexture<uint8_t> *idsPtr, CudaTexture<unsigned int> *labelsPtr)
{
    const unsigned int invokeIndex = getInvokeIndex();

    const CudaTexture<uint8_t> &ids = *idsPtr;
    CudaTexture<unsigned int> &labels = *labelsPtr;

    if (!isWithinBounds(invokeIndex, ids.size()))
        return;

    // Get the pixel coordinate
    const Vec2<int> textureIndex = getTextureIndex(invokeIndex, ids.size());

    const Vec2 topLeftIndex = {textureIndex.x - 1, textureIndex.y + 1};
    // const Vec2 bottomLeftIndex = {textureIndex.x - 1, textureIndex.y - 1};

    const Vec2 leftIndex = {textureIndex.x - 1, textureIndex.y};

    reduction(ids, labels, invokeIndex, topLeftIndex);
    // reduction(ids, labels, invokeIndex, bottomLeftIndex);

    reduction(ids, labels, invokeIndex, leftIndex);
}

__device__ void reduction(const CudaTexture<uint8_t> &ids, CudaTexture<unsigned int> &labels,
                          const unsigned int invokeIndex, const Vec2<int> &neighborIndex)
{
    unsigned int label1 = labels[invokeIndex];

    unsigned int newLabel = labels[label1];
    while (label1 != newLabel)
    {
        label1 = newLabel;
        newLabel = labels[label1];
    }

    unsigned int label2 = labels[neighborIndex];
    newLabel = labels[label2];
    while (label2 != newLabel)
    {
        label2 = newLabel;
        newLabel = labels[label2];
    }

    bool flag = true;
    if (ids[label1] == ids[label2] and label1 != label2)
        flag = false;

    if (label1 < label2)
    {
        const unsigned int tmp = label1;

        label1 = label2;
        label2 = tmp;
    }

    while (flag == false)
    {
        if (unsigned int label3 = atomicMin(&labels[label1], label2); label3 == label2)
            flag = true;
        else if (label3 > label2)
            label1 = label3;
        else if (label3 < label2)
        {
            label1 = label2;
            label2 = label3;
        }
    }
}

__global__ void copyToUint8Texture(const CudaTexture<unsigned int> *labelsPtr, CudaTexture<uint8_t> *writePtr,
                                   const unsigned int *labelIndices,
                                   const int len)
{
    const unsigned int invokeIndex = getInvokeIndex();

    CudaTexture<uint8_t> &write = *writePtr;
    const CudaTexture<unsigned int> &labels = *labelsPtr;

    if (!isWithinBounds(invokeIndex, labels.size()))
        return;

    const unsigned int label = labels[invokeIndex];

    for (int i = 0; i != len; ++i)
        if (labelIndices[i] == label)
        {
            write[invokeIndex] = i;
            return;
        }

    printf("Did not get a label assigned");
}
