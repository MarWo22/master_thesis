#ifndef PLATE_TECTONICS_KERNEL_CUH
#define PLATE_TECTONICS_KERNEL_CUH
#include <curand_kernel.h>

#include "../generation_settings.h"
#include "types/cuda_texture.cuh"
#include "types/plate_data.h"
#include "types/iteration_statistics.h"
#include "types/blur_buffer.h"
#include "types/voronoi_seed.h"

#define MAX_PLATE_COUNT 255u


struct PlateTexturesWrite
{
    CudaTexture<uint8_t> *plateIdsPtr;
    CudaTexture<float> *heightMapPtr;
};

struct PlateTexturesRead
{
    const CudaTexture<uint8_t> *plateIdsPtr;
    const CudaTexture<float> *heightMapPtr;
    const PlateData *plateData;
};

struct NoiseParameters
{
    const int seed;
    const unsigned int simIndex;
};

struct DivergenceTextures
{
    uint32_t *divergenceBitmap;
    uint32_t *hasDivergedBitmap;
};

struct CollisionVelocityChanges
{
    Vec2<float> inelasticDirectionalChange;
    float frictionLoss{};
};

struct CollisionTypeCounts
{
    float weightedHeightA;
    float weightedHeightB;
    int subductionsAContinental;
    int subductionsAOceanic;
    int subductionsBContinental;
    int subductionsBOceanic;
    int continental;
};


__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, const VoronoiSeed *seeds, int numSeeds);

__global__ void initPixelDependantPlateData(const CudaTexture<uint8_t> *r_idTexturePtr,
                                            const CudaTexture<float> *r_heightTexturePtr, PlateData *w_plateData);


__global__ void finalPixelPass(CudaTexture<uint8_t> *rw_idTexturePtr,
                               const CudaTexture<float> *r_heightTexturePtr,
                               const uint8_t *r_plateMergeIds,
                               PlateData *w_plateData);

__global__ void statisticsPass(PlateData *w_plateData, IterationStatistics *w_stats);


__global__ void initHeightmap(CudaTexture<float> *w_heightMapPtr, int seed, int octaves);

__global__ void applyPlateMovementChanges(PlateData *rw_plateLookup, const CollisionVelocityChanges *r_velocityChanges);

__global__ void updatePlateData(PlateData *plateLookup, const Vec2<float> *r_velocityChanges, const float *r_plateMass,
                                const int *r_plateSize);

__global__ void getPixelMovements(const CudaTexture<uint8_t> *r_idTexturePtr,
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
                                       CudaTexture<float2> *w_directionPtr);

__global__ void findPlateCenter(const IterationStatistics *r_stats, PlateData *plateLookup,
                                const CudaTexture<uint8_t> *r_plateIdsPtr, float4 *samples);

__global__ void findPlausibleSplitLine(const IterationStatistics *r_stats, Vec2<float> r_pivot,
                                       const CudaTexture<uint8_t> *r_plateIdsPtr, Vec2<float> *output);

__global__ void splitPlate(const IterationStatistics *r_stats, const uint8_t *r_newPlateId, Vec2<float> r_pivot,
                           const Vec2<float> *r_dir, CudaTexture<uint8_t> *r_plateIdsPtr, PlateData *plateLookup);

__global__ void selectUnusedPlateId(PlateData *plateLookup, uint8_t *plateId);

__global__ void processCollisions(PlateTexturesRead r_plateTextures, const CudaTexture<uint32_t> *r_collisionsPtr,
                                  const uint32_t *r_divergenceBitmap, const uint8_t *r_collisionTypeBitmap,
                                  PlateTexturesWrite w_plateTextures, uint32_t *w_hasDivergedBitmap,
                                  CudaTexture<float> *w_convergenceMapPtr,
                                  CudaTexture<uint8_t> *w_platesHaveCollidedPtr,
                                  CudaTexture<uint8_t> *w_accretionTexturePtr,
                                  CollisionVelocityChanges *w_velocityChanges,
                                  NoiseParameters noiseParameters);

__global__ void applyAccretion(const CudaTexture<float> *r_heightMapPtr,
                               const CudaTexture<uint8_t> *r_accretionTexturePtr, CudaTexture<uint8_t> *w_plateIdsPtr);

__global__ void flipDivergedBitmap(uint32_t *divergenceBitmap, const uint32_t *hasDivergedBitmap);

__global__ void VerticalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                             const CudaTexture<float> *r_upliftMapPtr, const CudaTexture<bool> *r_gridMapPtr,
                             CudaTexture<BlurBuffer> *w_bufferPtr, int range);

__global__ void HorizontalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                               const CudaTexture<bool> *r_gridMapPtr, const CudaTexture<BlurBuffer> *r_bufferPtr,
                               CudaTexture<float> *w_heightMapPtr, int range);

__global__ void processUplift(const CudaTexture<float> *w_upliftMapPtr, const CudaTexture<bool> *r_gridMapPtr,
                              CudaTexture<float> *w_heightMapPtr, int size,
                              float noiseFrequency, float noiseIntensity, int seed);

__global__ void createUpliftGrid(const CudaTexture<float> *r_upliftMapPtr, CudaTexture<bool> *w_gridMapPtr);

__device__ bool collisionContains(uint32_t collision, uint8_t plateId);

__global__ void downscaleUplift(const CudaTexture<float> *r_inputMapPtr, CudaTexture<float> *w_outputMapPtr,
                                int inWidth, int inHeight, int outWidth, int outHeight);

__global__ void upscaleUplift(const CudaTexture<float> *r_inputMapPtr, CudaTexture<float> *w_outputMapPtr, int inWidth,
                              int inHeight, int outWidth, int outHeight);

__device__ void processDivergence(PlateTexturesRead r_plateTextures, PlateTexturesWrite w_plateTextures,
                                  const CudaTexture<uint32_t> *r_collisionsPtr,
                                  const uint32_t *r_divergenceBitmap,
                                  uint32_t *w_hasDivergedBitmap,
                                  unsigned int invokeIndex, NoiseParameters noiseParameters);

__device__ void processConvergence(PlateTexturesWrite w_plateTextures, CudaTexture<float> *w_convergenceMapPtr,
                                   CudaTexture<uint8_t> *w_platesHaveCollidedPtr, const uint8_t *plateIds,
                                   const unsigned int *collisionTypes, const float *heights,
                                   unsigned int invokeIndex);

__device__ void processMovement(PlateTexturesRead r_plateTextures, PlateTexturesWrite w_plateTextures,
                                uint8_t originId, unsigned int invokeIndex);

__global__ void determinePlateMerge(const CudaTexture<uint8_t> *r_platesHaveCollided, const PlateData *r_plateLookup,
                                    uint8_t *w_plateMergeIds);


__device__ int getNewPlateId(const unsigned int *labelsShared, const unsigned int *labelCountsShared,
                             unsigned int label, unsigned int numUniqueLabels);

__global__ void assignNewPlateIds(CudaTexture<unsigned int> *r_labelsPtr, const unsigned int *r_uniqueLabels,
                                  const unsigned int *r_labelCounts, uint8_t *w_originalPlateIds,
                                  CudaTexture<uint8_t> *plateIdsPtr, unsigned int *w_unassignedIndices,
                                  int *unassignedIndicesCount, int numUniqueLabels);

__global__ void assignUnassignedIdsToNeighbor(CudaTexture<uint8_t> *plateIdsPtr, const unsigned int *unassignedIndices,
                                              unsigned int unassignedIndicesLen, int *hasRemainingWorkFlag);

__global__ void copyNewPlateIdLookup(const PlateData *r_plateData, const uint8_t *r_originalPlateIds,
                                     PlateData *w_plateData);

__global__ void rain(CudaTexture<float> *hydration, float deltatime, unsigned int seed);

__global__ void flux(const CudaTexture<float> *r_materialPtr, const CudaTexture<float> *r_hydrationPtr,
                     const CudaTexture<float4> *r_fluxPtr, CudaTexture<float4> *w_fluxPtr, float deltatime);

__global__ void flow(CudaTexture<float> *w_hydrationPtr, const CudaTexture<float4> *r_fluxPtr,
                     CudaTexture<float4> *w_fluxPtr, CudaTexture<Vec2<float> > *w_velocityPtr, float deltatime);

__global__ void sediment(CudaTexture<float> *w_materialPtr, CudaTexture<float> *w_sedimentPtr,
                         CudaTexture<Vec2<float> > *r_velocityPtr, float deltatime);

__global__ void transport(CudaTexture<float> *r_sedimentPtr, CudaTexture<float> *w_sedimentPtr,
                          CudaTexture<Vec2<float> > *r_velocityPtr, float deltatime);

__global__ void evaporate(CudaTexture<float> *hydration, float deltatime);

__global__ void mergeAndCountSizeMass(CudaTexture<uint8_t> *rw_plateIdsPtr,
                                      const CudaTexture<float> *r_heightTexturePtr, const uint8_t *r_plateMergeIds,
                                      PlateData *w_plateLookup, int *w_plateSize);

__global__ void determineCollisionType(PlateTexturesRead r_plateTextures,
                                       const CudaTexture<uint32_t> *r_collisionsPtr,
                                       const uint8_t *r_collisionTypeBitmap,
                                       CollisionTypeCounts *w_collisionTypeCounts);

__global__ void createCollisionTypeMatrix(const CollisionTypeCounts *r_collisionTypeCounts,
                                          uint8_t *w_collisionTypeBitmap);

__global__ void copyPlateDataGuiKernel(const PlateData *r_plateData, const uint8_t *r_collisionTypeBitmap,
                                       GuiPlateData *w_plateDataGui);

#endif //PLATE_TECTONICS_KERNEL_CUH
