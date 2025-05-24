#ifndef PLATE_TECTONIC_SIM_H
#define PLATE_TECTONIC_SIM_H
#include <curand_kernel.h>
#include <random>
#include <vector>

#include "cuda_gl_interop_manager.h"
#include "cuda_texture.cuh"
#include "plate_data.h"
#include "vec2.cuh"
#include "iteration_statistics.h"

class PlateTectonicSim {

    // Heightmap dimensions
    int m_width;
    int m_height;

    int m_seed;

    // Main heightmap on device
    CudaTextureHost<float> m_heightMapTexture;

    // Plate tectonic sim specific device arrays
    CudaTextureHost<uint8_t> m_plateIdsTexture;
    CudaTextureHost<float> m_overlapCrustTexture;

    PlateData *m_plateDataLookup;
    curandState *m_randStatesPlates;
    IterationStatistics *m_iterationStats;

    // General execution parameters
    int m_threadsPerBlock = 256;

    // Plate tectonic sim specific execution parameters
    int m_maxPlates = 255; // Higher values will require the plateIdsDevice to be increased to 16bit (Need to reserve 1 for the algorithm to work)
    int m_numStartingPlates;

    CudaGlInteropManager *m_interopManager;

public:

    PlateTectonicSim(int width, int height, int seed, int numStartingPlates, CudaGlInteropManager *interopManager);

    [[nodiscard]] CudaTextureHost<float> &heightMapCudaTexture() { return m_heightMapTexture; }
    [[nodiscard]] CudaTextureHost<uint8_t> &plateIdsCudaTexture() { return m_plateIdsTexture; }

    void initialize();

    void executeIteration();

    Vec2<float> getPlateCenter(int numBlocksPixels, int m_threadsPerBlock);

private:

    void initializeTectonics();

    std::vector<Vec2<float>> initializeVoronoiSeeds(
        std::default_random_engine &generator) const;

    std::vector<PlateData> initializePlateData(std::default_random_engine &generator,
        const std::vector<Vec2<float>> &voronoiSeeds) const;

    void setupToggleCallbacks() const;
    void copyPlateIdsGL() const;
    void copyDirectionGL() const;
    void copyVelocitiesGL() const;
};



#endif //PLATE_TECTONIC_SIM_H
