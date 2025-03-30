#ifndef PLATE_TECTONIC_SIM_H
#define PLATE_TECTONIC_SIM_H
#include <random>
#include <vector>

#include "cuda_texture.cuh"
#include "plate_data.h"
#include "vec2.cuh"

class PlateTectonicSim {

    // Heightmap dimensions
    int m_width;
    int m_height;

    // Main heightmap on device
    float *m_heightMapDevice;

    // Plate tectonic sim specific device arrays
    CudaTextureHost<uint8_t> m_plateIdsTexture;
    CudaTextureHost<float> m_overlapCrustTexture;
    PlateData *m_plateDataLookup;

    // General execution parameters
    int m_threadsPerBlock = 256;

    // Plate tectonic sim specific execution parameters
    int m_maxPlates = 256; // Higher values will require the plateIdsDevice to be increased to 16bit
    int m_numStartingPlates;

public:

    PlateTectonicSim(int width, int height, int numStartingPlates);

    [[nodiscard]] float *heightMapCudaTexture() const { return m_heightMapDevice; }
    [[nodiscard]] CudaTextureHost<uint8_t> &plateIdsCudaTexture() { return m_plateIdsTexture; }

    void initialize(int seed);

    void executeIteration();

private:

    void initializeTectonics(int seed);

    std::vector<Vec2<float>> initializeVoronoiSeeds(
        std::default_random_engine &generator) const;

    std::vector<PlateData> initializePlateData(std::default_random_engine &generator,
        const std::vector<Vec2<float>> &voronoiSeeds) const;

};



#endif //PLATE_TECTONIC_SIM_H
