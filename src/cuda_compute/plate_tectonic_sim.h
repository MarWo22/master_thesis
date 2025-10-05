#ifndef PLATE_TECTONIC_SIM_H
#define PLATE_TECTONIC_SIM_H

#include <curand_kernel.h>
#include <random>
#include <vector>

#include "cuda_gl_interop_manager.h"
#include "types/cuda_texture.cuh"
#include "plate_tectonics_kernel.cuh"
#include "texture_manager.cuh"
#include "types/plate_data.h"
#include "types/vec2.cuh"

constexpr int THREADS_PER_BLOCK = 256;

// For flat full matrix (e.g., MAX_PLATE_COUNT^2 elements)
constexpr int NUM_BLOCKS_PLATES_MATRIX = (MAX_PLATE_COUNT * MAX_PLATE_COUNT + THREADS_PER_BLOCK - 1) /
                                         THREADS_PER_BLOCK;

//  For simple per-plate parallelism
constexpr int NUM_BLOCKS_PLATES = (MAX_PLATE_COUNT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// Total number of unique unordered pairs (i, j) where i ≠ j, i < j
constexpr int NUM_TRIANGLE_ENTRIES = MAX_PLATE_COUNT * (MAX_PLATE_COUNT - 1) / 2;
constexpr int NUM_BLOCKS_TRIANGLE_ENTRIES = (NUM_TRIANGLE_ENTRIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// Single-bit version (1 bit per entry)
constexpr int NUM_BITS_TRIANGLE_SINGLE = NUM_TRIANGLE_ENTRIES * 1;
constexpr int NUM_WORDS_TRIANGLE_SINGLE_BITS = (NUM_BITS_TRIANGLE_SINGLE + 31) / 32;
constexpr int NUM_BLOCKS_TRIANGLE_SINGLE_BITS = (NUM_WORDS_TRIANGLE_SINGLE_BITS + THREADS_PER_BLOCK - 1) /
                                                THREADS_PER_BLOCK;

// --- Double-bit version (2 bits per entry)
constexpr int NUM_BITS_TRIANGLE_DOUBLE = NUM_TRIANGLE_ENTRIES * 2;
constexpr int NUM_WORDS_TRIANGLE_DOUBLE_BITS = (NUM_BITS_TRIANGLE_DOUBLE + 31) / 32;
constexpr int NUM_BLOCKS_TRIANGLE_DOUBLE_BITS = (NUM_WORDS_TRIANGLE_DOUBLE_BITS + THREADS_PER_BLOCK - 1) /
                                                THREADS_PER_BLOCK;

#define CUDA_ERROR_CHECK() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s (%d) at %s:%d\n", \
        cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(err); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s (%d) at %s:%d\n", \
        cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(err); \
    } \
} while (0)

class PlateTectonicSim
{
    /*
     * Constant parameters data
     */

    const int m_width;
    const int m_height;
    TextureManager m_textureManager;
    CudaGlInteropManager *const m_interopManager;
    unsigned int m_seed; // Not const, can be changed by calling resetSim()
    int m_numBlocksPixels;

    /*
     *  Persistent textures
     */

    std::unique_ptr<CudaTextureHost<float> > m_heightMapTexture;
    std::unique_ptr<CudaTextureHost<uint8_t> > m_plateIdsTexture;

    std::unique_ptr<CudaTextureHost<uint32_t> > m_plateCollisions; // Temp for debugging

    std::unique_ptr<CudaTextureHost<float> > m_hydrationLevel;
    std::unique_ptr<CudaTextureHost<float4> > m_hydrationFlux;
    std::unique_ptr<CudaTextureHost<Vec2<float> > > m_hydrationVelocity;
    std::unique_ptr<CudaTextureHost<float> > m_sedimentLevel;

    std::unique_ptr<CudaTextureHost<float>> m_pressure;
    std::unique_ptr<CudaTextureHost<float>> m_stress;
    std::unique_ptr<CudaTextureHost<Vec2<float>>> m_pressureVelocity;

    /*
     *  Persistent CUDA data containers
     *  Still implemented as a texture for simplicity reasons
     */

    std::unique_ptr<CudaTextureHost<PlateData> > m_plateDataLookup;
    std::unique_ptr<CudaTextureHost<curandState> > m_randStatesPlates;
    std::unique_ptr<CudaTextureHost<IterationStatistics> > m_iterationStats;

    std::unique_ptr<CudaTextureHost<uint32_t> > m_divergenceBitmap;
    // 0 - No collisions; 1 - Plate A is subducting; 2 - Plate B is subducting; 3 - Continental collision
    std::unique_ptr<CudaTextureHost<uint8_t> > m_collisionTypeBitmap;

    // Iteration counter
    unsigned int m_iterations;

public:
    PlateTectonicSim(int width, int height, unsigned int seed, CudaGlInteropManager *interopManager);

    ~PlateTectonicSim();

    void initialize(int numStartingPlates, const std::vector<int> &numVoronoiSeeds);

    void executeIteration();

    void resetSim(unsigned int seed, int numStartingPlates, const std::vector<int> &numVoronoiSeeds);

    static std::vector<VoronoiSeed> generateVoronoiSeeds(std::default_random_engine &generator,
                                                         const std::vector<VoronoiSeed> &centerSeeds, int numSeeds,
                                                         int width, int height);

    static std::vector<VoronoiSeed> generatePlateCenters(std::default_random_engine &generator, int numPlates,
                                                         int width, int height);

    static std::vector<PlateData> generatePlateData(std::default_random_engine &generator,
                                                    const std::vector<VoronoiSeed> &plateCenters,
                                                    int numStartingPlates);

private:
    void initializeTextures();

    void initializeTectonics(int numStartingPlates, const std::vector<int> &numVoronoiSeeds);

    void getPlateCollisions(CudaTextureHost<uint32_t> *plateCollisions);

    void processCollisionUplift(CudaTextureHost<float> *heightMapTextureWrite,
                                CudaTextureHost<uint8_t> *plateIdsTextureWrite,
                                CudaTextureHost<uint8_t> *platesHaveCollided,
                                CudaTextureHost<uint32_t> *plateCollisions,
                                CudaTextureHost<CollisionVelocityChanges> *velocityChanges);

    void applyHydraulicErosion(CudaTextureHost<float> *heightMapTextureWrite);

    void applyCCL();

    void processPlateSplitting();

    void SplitPlateV2(Vec2<int> point, uint8_t oldPlateId);

    Vec2<float> getPlateCenter();

    void setupGUICallbacks();

    void copyConstantTexturesInterop() const;

    void copyDirectionGL() const;

    void copyVelocitiesGL() const;

    void HydrationSubSim() const;

    void UpliftSubSim() const;

    void saveTexture() const;

    void onRenderSettingChange();

    void copyPlateDataGui() const;
};


#endif //PLATE_TECTONIC_SIM_H
