//
// Created by marti on 15/07/2025.
//

#ifndef DEBUG_LOGGING_CUH
#define DEBUG_LOGGING_CUH
#include <vector>
#include "../texture_manager.cuh"

void SaveVoronoiProcesTextures(int width, int height, unsigned int seed, int startingPlates,
                               const std::vector<int> &voronoiSeeds);

void MallocTestNoManager();
void MallocTestManager(TextureManager &textureManager);

#endif //DEBUG_LOGGING_CUH
