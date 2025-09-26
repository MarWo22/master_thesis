#ifndef PLATE_TECTONICS_KERNEL_CUH
#define PLATE_TECTONICS_KERNEL_CUH
#include <curand_kernel.h>

#include "types/cuda_texture.cuh"
#include "types/plate_data.h"
#include "types/vec2.cuh"
#include "types/iteration_statistics.h"
#include "types/distance_field_buffer.h"
#include "types/voronoi_seed.h"

#define MAX_PLATE_COUNT 255

__global__ void initPlateIDs(const CudaTexture<uint8_t> *idTexturePtr, const VoronoiSeed *seeds, int numSeeds);

__global__ void initPixelDependantPlateData(const CudaTexture<uint8_t> *r_idTexturePtr,
                                            const CudaTexture<float> *r_heightTexturePtr, PlateData *w_plateData);


__global__ void finalPixelPass(CudaTexture<uint8_t> *rw_idTexturePtr,
                               const CudaTexture<float> *r_heightTexturePtr,
                               const uint8_t *r_plateMergeIds,
                               PlateData *w_plateData);

__global__ void statisticsPass(PlateData *w_plateData, IterationStatistics *w_stats);


__global__ void initHeightmap(CudaTexture<float> *w_heightMapPtr, int seed, int octaves);


__global__ void plateMovement(CudaTexture<uint8_t> *idTexturePtr, CudaTexture<uint8_t> *writeIdTexturePtr,
                              const PlateData *plateLookup);

__global__ void updatePlateData(PlateData *plateLookup, Vec2<int> callSize);

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

__global__ void accumulatePlateAngularCoords(const CudaTexture<uint8_t> *r_plateIdsPtr,
                                           CudaTexture<float4> *w_plateAngularSumsPtr, CudaTexture<int> *w_plateCountsPtr);

__global__ void calculatePlateCenters(const CudaTexture<uint8_t> *r_plateIdsPtr,
                                    const CudaTexture<float4> *r_plateAngularSumsPtr, const CudaTexture<int> *r_plateCountsPtr,
                                    PlateData *w_plateData);


__global__ void findPlausibleSplitLine(const uint8_t highestStressPlateId, const Vec2<float> r_pivot,
                                       const CudaTexture<uint8_t> *r_plateIdsPtr, Vec2<float> *output);

__global__ void splitPlate(const uint8_t highestStressPlateId, const uint8_t *r_newPlateId, const Vec2<float> r_pivot,
                           const Vec2<float> *r_dir, CudaTexture<uint8_t> *r_plateIdsPtr, PlateData *plateLookup);

__global__ void selectUnusedPlateId(PlateData *plateLookup, uint8_t *plateId);

__global__ void processCollisions(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                  const CudaTexture<uint32_t> *r_collisionsPtr, PlateData *r_plateLookup,
                                  CudaTexture<uint8_t> *w_plateIdsPtr, CudaTexture<float> *w_heightMapPtr,
                                  CudaTexture<float> *w_convergenceMapPtr,
                                  CudaTexture<uint8_t> *w_platesHaveCollidedPtr);

__global__ void VerticalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                             const CudaTexture<float> *r_upliftMapPtr,
                             CudaTexture<DistanceFieldBuffer> *w_bufferPtr);

__global__ void HorizontalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<uint32_t> *r_collisionsPtr,
                               const CudaTexture<DistanceFieldBuffer> *r_bufferPtr,
                               CudaTexture<float> *w_heightMapPtr);

__global__ void createUpliftGrid(const CudaTexture<float> *r_upliftMapPtr, CudaTexture<bool> *w_gridMapPtr);

__device__ bool collisionContains(const uint32_t collision, const uint8_t plateId);

__device__ void processDivergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const PlateData *r_plateLookup,
                                  const CudaTexture<uint32_t> *r_collisionsPtr,
                                  const CudaTexture<float> *r_heightMapPtr, CudaTexture<uint8_t> *w_plateIdsPtr,
                                  CudaTexture<float> *w_heightMapPtr, const unsigned int invokeIndex);


__device__ void processConvergence(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                   const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                   CudaTexture<float> *w_heightMapPtr, CudaTexture<float> *w_convergenceMapPtr,
                                   CudaTexture<uint8_t> *w_platesHaveCollidedPtr,
                                   uint8_t plateA, uint8_t plateB, uint8_t plateC,
                                   uint8_t plateD, unsigned int invokeIndex);

__device__ void processMovement(const CudaTexture<uint8_t> *r_plateIdsPtr, const CudaTexture<float> *r_heightMapPtr,
                                const PlateData *r_plateLookup, CudaTexture<uint8_t> *w_plateIdsPtr,
                                CudaTexture<float> *w_heightMapPtr, uint8_t originId, unsigned int invokeIndex);

__global__ void determinePlateMerge(const CudaTexture<uint8_t> *r_platesHaveCollidedPtr, const PlateData *r_plateLookup,
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

__global__ void evaporate(CudaTexture<float>* w_hydrationPtr, float deltatime);

__global__ void pressureAccumulation(CudaTexture<float>* r_pressurePtr, CudaTexture<float>* w_pressurePtr, CudaTexture<uint8_t>* r_plateIdsPtr);

__global__ void pressureVerticalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, 
                                     const CudaTexture<float> *r_pressurePtr,
                                     CudaTexture<float> *w_bufferPtr);

__global__ void pressureHorizontalBlur(const CudaTexture<uint8_t> *r_plateIdsPtr, 
                                       const CudaTexture<float> *r_bufferPtr,
                                       CudaTexture<float> *w_pressurePtr);

__global__ void stress(const CudaTexture<float>* r_pressurePtr, const CudaTexture<float>* r_Material, CudaTexture<float>* w_stressPtr, const CudaTexture<uint8_t>* r_plateIdsPtr, const PlateData* r_plateData);

__global__ void computePerimeterAreaRatios(PlateData *w_plateData);

__global__ void thermalErosionKernel(CudaTexture<float> *w_materialPtr);

__global__ void calculatePressureVelocity(const CudaTexture<float> *r_pressurePtr,
                                          const CudaTexture<uint8_t> *r_plateIdsPtr);

#endif //PLATE_TECTONICS_KERNEL_CUH
