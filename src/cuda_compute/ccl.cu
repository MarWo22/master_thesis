#include "ccl.cuh"

#include "cuda_helper.cuh"

// https://www-sciencedirect-com.ezproxy.ub.gu.se/science/article/pii/S0010465515001472?via%3Dihub


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



// Return the root of a tree
__device__ unsigned findClamped(CudaTexture<unsigned int> &labels, unsigned int index) {

    unsigned int label = labels[index];

    while (label != index && label < index) {
        index = label;
        label = labels[index];
    }

    return index;
}

__device__ unsigned findUnclamped(CudaTexture<unsigned int> &labels, unsigned int index) {

    unsigned int label = labels[index];

    while (label != index) {
        index = label;
        label = labels[index];
    }

    return index;
}

// Links together trees containing a and b
__device__ void unionStep(CudaTexture<unsigned int> &labels, unsigned int index_a, unsigned index_b) {

    bool done;

    do {

        index_a = findClamped(labels, index_a);
        index_b = findUnclamped(labels, index_b);

        if (index_a < index_b) {
            const unsigned int old = atomicMin(&labels[index_b], index_a);
            done = old == index_b;
            index_b = old;
        }
        else if (index_b < index_a) {
            const unsigned int old = atomicMin(&labels[index_a], index_b );
            done = old == index_a;
            index_a = old;
        }
        else {
            done = true;
        }

    } while (!done);

}


// Init phase.
// Labels start at value 1, to differentiate them from background, that has value 0.
__global__ void init(const CudaTexture<uint8_t> *r_inputPtr, CudaTexture<unsigned int> *w_labelsPtr) {
    const CudaTexture<uint8_t> &r_input = *r_inputPtr;
    CudaTexture<unsigned int> &w_labels = *w_labelsPtr;

    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(invokeIndex, r_input.size()))
        return;

    const Vec2<int> textureIdx = getTextureIndex(invokeIndex, r_input.size());
    const uint8_t currentId = r_input[invokeIndex];

    const Vec2<int> textureIdx_l = textureIdx + Vec2(-1, 0);
    const Vec2<int> textureIdx_t = textureIdx + Vec2(0, -1);
    const Vec2<int> textureIdx_tl = textureIdx + Vec2(-1, -1);
    const Vec2<int> textureIdx_tr = textureIdx + Vec2(1, -1);

    if (currentId == r_input[textureIdx_tl])
        w_labels[invokeIndex] = w_labels.coordinateToIndex(textureIdx_tl);
    else if (currentId == r_input[textureIdx_t])
        w_labels[invokeIndex] = w_labels.coordinateToIndex(textureIdx_t);
    else if (currentId == r_input[textureIdx_tr])
        w_labels[invokeIndex] = w_labels.coordinateToIndex(textureIdx_tr);
    else if (currentId == r_input[textureIdx_l])
        w_labels[invokeIndex] = w_labels.coordinateToIndex(textureIdx_l);
    else
        w_labels[invokeIndex] = invokeIndex;
}


// Analysis phase.
__global__ void analyzeClamped(CudaTexture<unsigned int> *w_labelsPtr) {
    CudaTexture<unsigned int> &w_labels = *w_labelsPtr;

    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(invokeIndex, w_labels.size()))
        return;

    w_labels[invokeIndex] = findClamped(w_labels, invokeIndex);
}

// Analysis phase.
__global__ void analyzeUnclamped(CudaTexture<unsigned int> *w_labelsPtr) {
    CudaTexture<unsigned int> &w_labels = *w_labelsPtr;

    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(invokeIndex, w_labels.size()))
        return;

    w_labels[invokeIndex] = findUnclamped(w_labels, invokeIndex);
}


__global__ void reduce(const CudaTexture<uint8_t> *r_inputPtr, CudaTexture<unsigned int> *w_labelsPtr) {

    const CudaTexture<uint8_t> &r_input = *r_inputPtr;
    CudaTexture<unsigned int> &w_labels = *w_labelsPtr;

    const unsigned int invokeIndex = getInvokeIndex();

    if (!isWithinBounds(invokeIndex, r_input.size()))
        return;

    const Vec2<int> textureIdx = getTextureIndex(invokeIndex, r_input.size());
    const uint8_t currentId = r_input[invokeIndex];

    const Vec2<int> textureIdx_l = textureIdx + Vec2(-1, 0);
    const Vec2<int> textureIdx_tl = textureIdx + Vec2(-1, -1);

    if (textureIdx.y == 0)
    {
        const Vec2<int> textureIdx_t = textureIdx + Vec2(0, -1);
        const Vec2<int> textureIdx_tr = textureIdx + Vec2(1, -1);
        if (currentId == r_input[textureIdx_t])
            unionStep(w_labels, invokeIndex, r_input.coordinateToIndex(textureIdx_t));

        if (currentId == r_input[textureIdx_tr])
            unionStep(w_labels, invokeIndex, r_input.coordinateToIndex(textureIdx_tr));
    }

    if (currentId == r_input[textureIdx_tl])
        unionStep(w_labels, invokeIndex, r_input.coordinateToIndex(textureIdx_tl));

    if (currentId == r_input[textureIdx_l])
        unionStep(w_labels, invokeIndex, r_input.coordinateToIndex(textureIdx_l));

}