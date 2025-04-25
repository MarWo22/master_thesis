#ifndef PLATE_TECTONICS_KERNEL_CUH
#define PLATE_TECTONICS_KERNEL_CUH
#include "cuda_texture.cuh"
#include "plate_data.h"
#include "vec2.cuh"


__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, const Vec2<float> *seeds, int numSeeds);

__global__ void initHeightmap(CudaTexture<float> * w_heightMapPtr, int seed, int octaves);


__global__ void plateMovement(CudaTexture<uint8_t> *idTexturePtr, CudaTexture<uint8_t> *writeIdTexturePtr,
                              const PlateData *plateLookup);

__global__ void updatePlateData(PlateData *plateLookup, Vec2<int> callSize);

__global__ void testingPlateMovement(const CudaTexture<uint8_t> *r_idTexturePtr,
                                     const PlateData *r_plateLookup,
                                     CudaTexture<unsigned int> *w_pixelIndicesPtr,
                                     CudaTexture<uint8_t> *w_plateIdsPtr);

__global__ void detectCollisions(const unsigned int *r_pixelIndices, uint32_t *collisionBitfield,
                                 unsigned int invokeSize);

__global__ void registerPlateCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr,
                                        const CudaTexture<unsigned int> *r_pixelIndicesPtr,
                                        const CudaTexture<uint8_t> *r_exclusivePrefixSumPtr,
                                        CudaTexture<uint32_t> *w_collisionsPtr);

__global__ void convertCollisionMapForGL(const CudaTexture<uint32_t> *r_texturePtr, CudaTexture<uint8_t> *w_texturePtr);

__global__ void createVelocityTexture(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateData,
                                      CudaTexture<float> *w_velocityPtr);

__global__ void createDirectionTexture(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateData,
                                       CudaTexture<float2> *w_velocityPtr);

__global__ void processCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, const PlateData *r_plateLookup,
                                  CudaTexture<uint8_t> *w_plateIdsPtr, CudaTexture<float> *w_heightMapPtr, CudaTexture<float>* w_upliftMapPtr);

__global__ void processUplift(CudaTexture<float>* w_upliftMapPtr, CudaTexture<float>* w_heightMapPtr, const int size, const float noiseFrequency, const float noiseIntensity, const int seed);

__device__ void processDivergence(const CudaTexture<uint8_t> *r_plateIdsPtr, CudaTexture<uint8_t> *w_plateIdsPtr,
                                  CudaTexture<float> *w_heightMapPtr, unsigned int invokeIndex);


__device__ void processConvergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                   const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                   CudaTexture<float> *w_heightMapPtr, CudaTexture<float>* w_upliftMapPtr, 
                                   uint8_t plateA, uint8_t plateB, uint8_t plateC,
                                   uint8_t plateD, unsigned int invokeIndex);

__device__ void processMovement(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                CudaTexture<float> *w_heightMapPtr, uint8_t originId, unsigned int invokeIndex);

#endif //PLATE_TECTONICS_KERNEL_CUH
