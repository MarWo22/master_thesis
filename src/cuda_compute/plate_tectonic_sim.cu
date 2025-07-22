#include "plate_tectonic_sim.h"

#include <iostream>
#include <map>
#include <random>

#include "ccl.cuh"
#include "cuda_gl_interop_manager.h"
#include "types/plate_data.h"
#include "plate_tectonics_kernel.cuh"
#include "../generation_settings.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include "texture_save.cuh"

extern RenderSettings renderSettings;
extern SimulationSettings simulationSettings;
extern SaveTextureGui saveTextureGui;

PlateTectonicSim::PlateTectonicSim(const int width, const int height, const unsigned int seed,
                                   CudaGlInteropManager *interopManager)
    : m_width(width)
      , m_height(height)
      , m_interopManager(interopManager)
      , m_seed(seed)
      , m_numBlocksPixels((width * height + 1) / THREADS_PER_BLOCK)
      , m_iterations(0)
{}

void PlateTectonicSim::initialize(const int numStartingPlates, const std::vector<int> &numVoronoiSeeds)
{
    initializeTextures();
    std::cout << "Initialized textured\n";
    initializeTectonics(numStartingPlates, numVoronoiSeeds);
    std::cout << "Initialized tectonics\n";
    setupGUICallbacks();

    if (m_interopManager != nullptr)
    {
        setupToggleCallbacks();
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap", "waterTexture"});
        copyConstantTexturesInterop();
    }
}


void PlateTectonicSim::setupToggleCallbacks() const
{
    renderSettings.registerCallback("toggleDefaultMode", [this]
    {
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap", "waterTexture"});
        copyConstantTexturesInterop();
    });

    renderSettings.registerCallback("toggleCollisionMode", [this]
    {
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap", "waterTexture", "collisionMap"});
        copyConstantTexturesInterop();
    });

    renderSettings.registerCallback("toggleDirectionMode", [this]
    {
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap", "waterTexture", "directionTexture"});
        copyConstantTexturesInterop();
    });

    renderSettings.registerCallback("toggleVelocityMode", [this]
    {
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap", "waterTexture", "velocityTexture"});
        copyConstantTexturesInterop();
    });

    renderSettings.registerCallback("toggleUpliftMode", [this]
    {
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap", "waterTexture", "upliftTexture"});
        copyConstantTexturesInterop();
    });
}


void PlateTectonicSim::executeIteration()
{
    const auto start{std::chrono::steady_clock::now()};
    std::cout << "Executing iteration" << std::endl;

    auto plateCollisions = m_textureManager.generateTextureAndReset<uint32_t>(m_width, m_height, 0);
    getPlateCollisions(plateCollisions.get());

    // Process colliding plates and perform uplift
    auto heightMapTextureWrite = m_textureManager.generateTexture<float>(m_width, m_height);
    auto plateIdsTextureWrite = m_textureManager.generateTexture<uint8_t>(m_width, m_height);
    auto platesHaveCollided = m_textureManager.generateTextureAndReset<uint8_t>(MAX_PLATE_COUNT, MAX_PLATE_COUNT, 0);

    processCollisionUplift(heightMapTextureWrite.get(), plateIdsTextureWrite.get(), platesHaveCollided.get(),
                           plateCollisions.get());

    // applyHydraulicErosion(heightMapTextureWrite.get());

    copyAndReleaseTexture(*m_heightMapTexture, std::move(heightMapTextureWrite), m_height, m_width);
    copyAndReleaseTexture(*m_plateIdsTexture, std::move(plateIdsTextureWrite), m_height, m_width);


    /*
     * STEP SIX
     * Apply the movement to the plates, and update the velocities and directions according to collisions.
     */

    updatePlateData<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateDataLookup->getPointer(),
                                                              Vec2(MAX_PLATE_COUNT, 1));

    auto plateMergeIds = m_textureManager.generateTextureAndReset<uint8_t>(MAX_PLATE_COUNT, 1, MAX_PLATE_COUNT);

    /*
     * STEP SEVEN
     * Final maxplatematrix and full-pixel pass that allow for the merging and splitting of plates, and the updates of
     * mass and sizes
     */

    determinePlateMerge<<<NUM_BLOCKS_PLATES_MATRIX, THREADS_PER_BLOCK>>>(
        platesHaveCollided->deviceTexture(), m_plateDataLookup->getPointer(), plateMergeIds->getPointer());

    finalPixelPass<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                             m_heightMapTexture->deviceTexture(),
                                                             plateMergeIds->getPointer(),
                                                             m_plateDataLookup->getPointer());

    statisticsPass<<<1, 1>>>(m_plateDataLookup->getPointer(), m_iterationStats->getPointer());

    processPlateSplitting();

    applyCCL();

    cudaDeviceSynchronize();
    if (m_interopManager)
    {
        if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_COLLISION_AREAS)
        {
            CudaTextureHost<uint8_t> glTexture;
            glTexture.initialize(m_width, m_height);
            convertCollisionMapForGL<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(
                plateCollisions->deviceTexture(), glTexture.deviceTexture());

            m_interopManager->copyConnection("collisionMap", glTexture.getPointer());
        }
        copyConstantTexturesInterop();
    }

    m_iterations++;

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "Iteration duration: " << elapsed_seconds.count() << std::endl;
}

void PlateTectonicSim::copyConstantTexturesInterop() const
{
    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_PLATE_VELOCITIES)
        copyVelocitiesGL();

    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_PLATE_DIRECTIONS)
        copyDirectionGL();

    m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture->getPointer());
    m_interopManager->copyConnection("heightMap", m_heightMapTexture->getPointer());
    m_interopManager->copyConnection("waterTexture", m_hydrationLevel->getPointer());
}

void PlateTectonicSim::getPlateCollisions(CudaTextureHost<uint32_t> *plateCollisions)
{
    const auto pixelIndicesCollisions = m_textureManager.generateTexture<unsigned int>(m_width, m_height);
    const auto plateIdsCollisions = m_textureManager.generateTexture<uint8_t>(m_width, m_height);

    // Apply movement to pixels according to the velocity, direction, and sub-pixel location of each pixel.
    // Each pixel registers to which pixel (linear index) it moves, and writes this to the w_pixelIndicesCollisions texture
    // Also performing a copy of m_plateIdsTexture to plateIdsCollisions here to omit a separate CudaMemcpy call
    getPixelMovements<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                   m_plateDataLookup->getPointer(),
                                                                   pixelIndicesCollisions->deviceTexture(),
                                                                   plateIdsCollisions->deviceTexture());


    // Perform a sort by key on the pixelIndices and plateIds textures to align the pixelIndices in ascending order while
    // remaining the alignment. Position n in pixelIndices still corresponds to position n in plateIds
    // This is followed by an exclusive prefix sum by key. The prefix sum output is stored in the exclusivePrefixSum texture
    // This texture will now contain the occurrence index of the pixel, aligned with the pixelIndicesCollision and
    // plateIdsCollisions texture
    const auto exclusivePrefixSum = m_textureManager.generateTextureAndReset<uint8_t>(m_width, m_height, 1);
    const thrust::device_ptr<unsigned int> pixelIndicesThrust(pixelIndicesCollisions->getPointer());
    const thrust::device_ptr<uint8_t> plateIdsThrust(plateIdsCollisions->getPointer());
    const thrust::device_ptr<uint8_t> exclusivePrefixSumThrust(exclusivePrefixSum->getPointer());

    sort_by_key(pixelIndicesThrust, pixelIndicesThrust + m_height * m_width, plateIdsThrust,
                thrust::greater<unsigned int>());

    exclusive_scan_by_key(pixelIndicesThrust, pixelIndicesThrust + m_width * m_height, exclusivePrefixSumThrust,
                          exclusivePrefixSumThrust);


    // Go through each pair of entries of the pixelIndices, exclusivePrefixSum and plateCollisions textures and register
    // the collisions to the associated pixels. The collisions are stored in plateCollisions, a 32bit texture where each
    // entry has four packed 8bit values, indicating the presence of a plate in that pixel. A byte with value  0-255
    // indicates the presence of plate (255-n) in that position . A value of n=0 indicates there is no plate in that
    // specific byte. This allows for the detection of 0-4 plates in a pixel, any exceeds will be ignored.
    registerPlateCollisions<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(plateIdsCollisions->deviceTexture(),
                                                                      pixelIndicesCollisions->deviceTexture(),
                                                                      exclusivePrefixSum->deviceTexture(),
                                                                      plateCollisions->deviceTexture());

    // pixelIndicesCollisions, plateIdsCollisions, exclusivePrefixSum get released again
}


void PlateTectonicSim::processCollisionUplift(CudaTextureHost<float> *heightMapTextureWrite,
                                              CudaTextureHost<uint8_t> *plateIdsTextureWrite,
                                              CudaTextureHost<uint8_t> *platesHaveCollided,
                                              CudaTextureHost<uint32_t> *plateCollisions)
{
    constexpr int de = 6;

    const auto upliftBufferA = m_textureManager.generateTextureAndReset<float>(m_width, m_height, 0);
    const auto upliftBufferB = m_textureManager.generateTextureAndReset<float>(m_width, m_height, 0);
    const auto upliftGrid = m_textureManager.generateTexture<bool>(static_cast<int>(m_width * powf(0.5, de)),
                                                                   static_cast<int>(m_height * powf(0.5, de)));
    const auto blurBuffer = m_textureManager.generateTexture<BlurBuffer>(m_width, m_height);

    /*
     * STEP FOUR
     * Main bulk of work. Each pixel updates depending on the presence of plates. If there are no plates present in
     * a pixel, it indicates the divergence of plates, triggering the creation of new oceanic crust in that plate. This
     * new crust is assigned to the plate last seen in this location (CURRENTLY RANDOM CHOSEN, BUT SHOULD LIKELY CHANGE).
     * The presence of one plate indicates a simple movement, and more indicates a convergence. Generation of new crust
     * and movement of original crust is dealt with in this step.
     */

    processCollisions<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                m_heightMapTexture->deviceTexture(),
                                                                plateCollisions->deviceTexture(),
                                                                m_plateDataLookup->getPointer(),
                                                                plateIdsTextureWrite->deviceTexture(),
                                                                heightMapTextureWrite->deviceTexture(),
                                                                upliftBufferA->deviceTexture(),
                                                                platesHaveCollided->deviceTexture());

    /*
     * STEP FIVE
     * Perform uplift
     */

    createUpliftGrid<<<static_cast<int>(m_numBlocksPixels * 0.125), static_cast<int>(THREADS_PER_BLOCK * 0.125)>>>(
        upliftBufferA->deviceTexture(), upliftGrid->deviceTexture());


    VerticalBlur<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                           plateCollisions->deviceTexture(),
                                                           upliftBufferA->deviceTexture(),
                                                           upliftGrid->deviceTexture(),
                                                           blurBuffer->deviceTexture(), 50);

    HorizontalBlur<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                             plateCollisions->deviceTexture(),
                                                             upliftGrid->deviceTexture(),
                                                             blurBuffer->deviceTexture(),
                                                             heightMapTextureWrite->deviceTexture(), 50);
}

void PlateTectonicSim::applyHydraulicErosion(CudaTextureHost<float> *heightMapTextureWrite)
{
    const auto fluxBuffer = m_textureManager.generateTexture<float4>(m_width, m_height);
    const auto sedimentBuffer = m_textureManager.generateTexture<float>(m_width, m_height);

    for (size_t i = 0; i < 100; i++)
    {
        rain<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_hydrationLevel->deviceTexture(), 1.0, m_seed + m_iterations);
        flux<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(heightMapTextureWrite->deviceTexture(),
                                                       m_hydrationLevel->deviceTexture(),
                                                       m_hydrationFlux->deviceTexture(), fluxBuffer->deviceTexture(),
                                                       1.0);
        flow<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_hydrationLevel->deviceTexture(),
                                                       fluxBuffer->deviceTexture(),
                                                       m_hydrationFlux->deviceTexture(),
                                                       m_hydrationVelocity->deviceTexture(), 1.0);
        sediment<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(heightMapTextureWrite->deviceTexture(),
                                                           m_sedimentLevel->deviceTexture(),
                                                           m_hydrationVelocity->deviceTexture(), 1.0);
        transport<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_sedimentLevel->deviceTexture(),
                                                            sedimentBuffer->deviceTexture(),
                                                            m_hydrationVelocity->deviceTexture(), 1.0);
        evaporate<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_hydrationLevel->deviceTexture(), 1.0);

        cudaMemcpyAsync(m_sedimentLevel->getPointer(), sedimentBuffer->getPointer(), sizeof(float) * m_width * m_height,
                        cudaMemcpyDeviceToDevice);
    }
}

void PlateTectonicSim::applyCCL()
{
    const auto labels = m_textureManager.generateTexture<unsigned int>(m_width, m_height);


    // First, apply 8-way CCL to generate a texture of labels
    init<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(), labels->deviceTexture());
    analyzeClamped<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(labels->deviceTexture());
    reduce<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(), labels->deviceTexture());
    analyzeUnclamped<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(labels->deviceTexture());

    const int labelCountSize = static_cast<int>(m_width * m_height * 0.005);
    auto labelCounts = m_textureManager.generateTexture<unsigned int>(labelCountSize, 1);
    auto uniqueLabels = m_textureManager.generateTexture<unsigned int>(labelCountSize, 1);
    auto labelsCopy = m_textureManager.generateTexture<unsigned int>(m_width, m_height);

    copyTexture(*labelsCopy, *labels, m_width, m_height);

    // Create thrust device ptr wrappers
    const thrust::device_ptr<unsigned int> labelsCopyThrust(labelsCopy->getPointer());
    const thrust::device_ptr<unsigned int> labelCountsThrust(labelCounts->getPointer());
    const thrust::device_ptr<unsigned int> uniqueLabelsThrust(uniqueLabels->getPointer());

    // First, sort the labels
    sort(labelsCopyThrust, labelsCopyThrust + m_height * m_width,
         thrust::greater<unsigned int>());

    // Second, apply a reduce by key to create an array of key:count pairs. Capture the end iterator such that we are
    // aware of the size of the final array.
    const auto resultEnd = reduce_by_key(labelsCopyThrust, labelsCopyThrust + m_height * m_width,
                                         thrust::make_constant_iterator<int>(1), uniqueLabelsThrust,
                                         labelCountsThrust);
    labelsCopy.reset(); // Explicitly return the texture back to the manager

    // We can use the iterator to calculate the number of unique labels. This is clamped to max_plate_count, as we cannot
    // allocate more plates than the max count anyway.
    const int numUniqueLabels = min(static_cast<int>(resultEnd.first - uniqueLabelsThrust), MAX_PLATE_COUNT);

    // Third, sort the section of the labels:count pairs that has been initialized in the previous step. This gives us
    // the pairs sorted by counts, in descending order
    sort_by_key(labelCountsThrust, labelCountsThrust + numUniqueLabels, uniqueLabelsThrust,
                thrust::greater<unsigned int>());

    auto originalPlateIds = m_textureManager.generateTextureAndReset<uint8_t>(MAX_PLATE_COUNT, 1, MAX_PLATE_COUNT);
    auto unassignedIndices = m_textureManager.generateTexture<unsigned int>(labelCountSize, 1);
    auto unassignedIndicesCount = m_textureManager.generateTextureAndReset<int>(1, 1, 0);

    assignNewPlateIds<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(labels->deviceTexture(), uniqueLabels->getPointer(),
                                                                labelCounts->getPointer(),
                                                                originalPlateIds->getPointer(),
                                                                m_plateIdsTexture->deviceTexture(),
                                                                unassignedIndices->getPointer(),
                                                                unassignedIndicesCount->getPointer(),
                                                                numUniqueLabels);
    labelCounts.reset();
    uniqueLabels.reset();

    int unassignedIndicesCountHost;

    if (const cudaError_t err = cudaMemcpy(&unassignedIndicesCountHost, unassignedIndicesCount->getPointer(),
                                           sizeof(int),
                                           cudaMemcpyDeviceToHost); err != cudaSuccess)
        std::cerr << "Error memcpy unassignedIndicesCountHost: " << cudaGetErrorString(err) << "\n";
    unassignedIndicesCount.reset();
    if (unassignedIndicesCountHost != 0)
    {
        // Using int instead of boolean since cuda uses ints
        int hasWork = 1;
        const int gridSize = (unassignedIndicesCountHost + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        while (hasWork)
        {
            const auto hasRemainingWork = m_textureManager.generateTextureAndReset(1, 1, 0);

            assignUnassignedIdsToNeighbor<<<gridSize, THREADS_PER_BLOCK>>>(
                m_plateIdsTexture->deviceTexture(), unassignedIndices->getPointer(), unassignedIndicesCountHost,
                hasRemainingWork->getPointer());

            if (const cudaError_t err = cudaMemcpy(&hasWork, hasRemainingWork->getPointer(), sizeof(int),
                                                   cudaMemcpyDeviceToHost);
                err != cudaSuccess)
                std::cerr << "Error memcpy hasWork: " << cudaGetErrorString(err) << "\n";
        }
    }
    auto plateDataWrite = m_textureManager.generateTexture<PlateData>(MAX_PLATE_COUNT, 1);

    copyNewPlateIdLookup<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateDataLookup->getPointer(),
                                                                   originalPlateIds->getPointer(),
                                                                   plateDataWrite->getPointer());

    copyAndReleaseTexture(*m_plateDataLookup, std::move(plateDataWrite), MAX_PLATE_COUNT, 1);
}

void PlateTectonicSim::processPlateSplitting()
{
    int h_largest;
    const int *d_largest = &m_iterationStats->getPointer()->largestValue;
    cudaMemcpy(&h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);

    printf("largest: %.i\n", h_largest);
    if (h_largest > 220000)
    {
        const Vec2<float> h_pivot = getPlateCenter();
        auto d_dir = m_textureManager.generateTexture<Vec2<float> >(1, 1);

        findPlausibleSplitLine<<<1, 10, 10 * sizeof(float)>>>(m_iterationStats->getPointer(), h_pivot,
                                                              m_plateIdsTexture->deviceTexture(), d_dir->getPointer());

        auto newPlateId = m_textureManager.generateTexture<uint8_t>(1, 1);

        selectUnusedPlateId<<<1, 1>>>(m_plateDataLookup->getPointer(), newPlateId->getPointer());

        splitPlate << <m_numBlocksPixels, THREADS_PER_BLOCK >> >(m_iterationStats->getPointer(),
                                                                 newPlateId->getPointer(), h_pivot,
                                                                 d_dir->getPointer(),
                                                                 m_plateIdsTexture->deviceTexture(),
                                                                 m_plateDataLookup->getPointer());
    }
}

Vec2<float> PlateTectonicSim::getPlateCenter()
{
    auto d_samples = m_textureManager.generateTexture<float4>(m_numBlocksPixels, 1);
    findPlateCenter<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_iterationStats->getPointer(),
                                                              m_plateDataLookup->getPointer(),
                                                              m_plateIdsTexture->deviceTexture(), d_samples->getPointer());
    const auto h_samples = new float4[m_numBlocksPixels];
    cudaMemcpy(h_samples, d_samples->getPointer(), sizeof(float4) * m_numBlocksPixels, cudaMemcpyDeviceToHost);

    float sinX = 0, cosX = 0, sinY = 0, cosY = 0;
    for (int i = 0; i < m_numBlocksPixels; ++i)
    {
        sinX += h_samples[i].x;
        cosX += h_samples[i].y;
        sinY += h_samples[i].z;
        cosY += h_samples[i].w;
    }

    float angleX = atan2f(sinX, cosX);
    float angleY = atan2f(sinY, cosY);

    if (angleX < 0) angleX += CURAND_2PI;
    if (angleY < 0) angleY += CURAND_2PI;

    const float midX = m_width * angleX / CURAND_2PI;
    const float midY = m_height * angleY / CURAND_2PI;

    delete[] h_samples;

    return Vec2(midX, midY);
}

void PlateTectonicSim::copyDirectionGL() const
{
    CudaTextureHost<float2> glTexture;
    glTexture.initialize(m_width, m_height);
    int m_numBlocksPixels = (m_width * m_height + 1) / THREADS_PER_BLOCK;
    createDirectionTexture<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                     m_plateDataLookup->getPointer(),
                                                                     glTexture.deviceTexture());

    m_interopManager->copyConnection("directionTexture", glTexture.getPointer());
}

void PlateTectonicSim::copyVelocitiesGL() const
{
    CudaTextureHost<float> glTexture;
    glTexture.initialize(m_width, m_height);
    createVelocityTexture<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                    m_plateDataLookup->getPointer(),
                                                                    glTexture.deviceTexture());

    m_interopManager->copyConnection("velocityTexture", glTexture.getPointer());
}

void PlateTectonicSim::resetSim(const unsigned int seed, const int numStartingPlates,
                                const std::vector<int> &numVoronoiSeeds)
{
    m_seed = seed;
    m_heightMapTexture.reset();

    // Plate tectonic sim specific device arrays
    m_plateIdsTexture.reset();

    m_hydrationFlux.reset();
    m_hydrationLevel.reset();
    m_hydrationVelocity.reset();
    m_sedimentLevel.reset();
    m_plateDataLookup.reset();
    m_randStatesPlates.reset();
    m_iterationStats.reset();

    initializeTextures();
    initializeTectonics(numStartingPlates, numVoronoiSeeds);

    if (m_interopManager)
        copyConstantTexturesInterop();
}


void PlateTectonicSim::saveTexture() const
{
    switch (saveTextureGui.texture)
    {
        case SaveTextureGui::TextureType::HEIGHTMAP:
            saveCudaTextureToDiskGrayscale(saveTextureGui.path.c_str(), *m_heightMapTexture);
            break;
        case SaveTextureGui::TextureType::PLATE_IDS:
            saveGrayscale8BitCudaTextureToDiskAsRgb(saveTextureGui.path.c_str(), *m_plateIdsTexture);
            break;
        default:
            std::cerr << "Unknown texture type during saving\n";
    }
}

void PlateTectonicSim::initializeTectonics(const int numStartingPlates, const std::vector<int> &numVoronoiSeeds)
{
    std::default_random_engine generator(m_seed);

    // Init voronoi vector on host, copy to local device, and free at end of function
    const auto plateCenters = generatePlateCenters(generator, numStartingPlates, m_width, m_height);
    // Not using the memory manager, since these are one-time temporary allocations
    std::cout << "Generated plate centers\n";
    CudaTextureHost<uint8_t> plateCentersTexture;
    plateCentersTexture.initializeAndClear(m_width, m_height, 255);

    std::vector<VoronoiSeed> voronoiSeedsHost = plateCenters;

    // Apply all iterations of voronoi seed
    // If there are no seeds, or less than the original centers, the centers are used instead
    for (const int numSeeds: numVoronoiSeeds)
    {
        voronoiSeedsHost = generateVoronoiSeeds(generator, voronoiSeedsHost, numSeeds, m_width, m_height);
        std::cout << "Generated" << numSeeds << " seeds\n";

    }


    if (voronoiSeedsHost.size() < plateCenters.size())
        voronoiSeedsHost = plateCenters;

    CudaTextureHost<VoronoiSeed> voronoiSeedsDevice;
    voronoiSeedsDevice.initialize(static_cast<int>(voronoiSeedsHost.size()), 1);

    if (const cudaError_t err = cudaMemcpy(voronoiSeedsDevice.getPointer(), voronoiSeedsHost.data(),
                                           sizeof(VoronoiSeed) * voronoiSeedsHost.size(),
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy voronoiSeedsDevice: " << cudaGetErrorString(err) << std::endl;

    const std::vector<PlateData> plateDataHost = generatePlateData(generator, plateCenters, numStartingPlates);

    if (const cudaError_t err = cudaMemcpy(m_plateDataLookup->getPointer(), plateDataHost.data(), sizeof(PlateData) * MAX_PLATE_COUNT,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    // Each pixel is assigned the ID of the nearest voronoi seed
    // The plateIDs are written to the m_plateIdsTexture texture
    initPlateIDs<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                           voronoiSeedsDevice.getPointer(),
                                                           static_cast<int>(voronoiSeedsHost.size()));
    std::cout << "Initialized plate IDs\n";
    // Initialized the heightmap with simplex noise
    // The heightmap is written to the m_heightMapTexture texture
    initHeightmap<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_heightMapTexture->deviceTexture(), m_seed, 1);
    std::cout << "Initialized heightmap\n";

    // Extracts the size and mass of the plates
    // This is stored in the m_plateDataLookup lookup texture
    initPixelDependantPlateData<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                          m_heightMapTexture->deviceTexture(),
                                                                          m_plateDataLookup->getPointer());
    std::cout << "Initialized pixel data\n";

    cudaDeviceSynchronize();
}

void PlateTectonicSim::setupGUICallbacks()
{
    renderSettings.registerCallback("executeIterations", [this]
    {
        if (!renderSettings.isExecutingRealtime)
            for (int i = 0; i < renderSettings.executionIterations; ++i)
                executeIteration();
        else
        {
            if (renderSettings.executionIterations > 0)
            {
                executeIteration();
                renderSettings.executionIterations -= 1;
            }
            if (renderSettings.executionIterations <= 0)
                renderSettings.isExecutingRealtime = false;
        }
    });

    simulationSettings.resetCallback = [this](const unsigned int seed, const int numPlates,
                                              const std::vector<int> &numVoronoiSeeds)
    {
        resetSim(seed, numPlates, numVoronoiSeeds);
    };

    saveTextureGui.saveTextureCallback = [this]
    {
        saveTexture();
    };
}

std::vector<PlateData> PlateTectonicSim::generatePlateData(std::default_random_engine &generator,
                                                           const std::vector<VoronoiSeed> &plateCenters,
                                                           const int numStartingPlates)
{
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<PlateData> plateData(MAX_PLATE_COUNT);

    for (int i = 0; i != numStartingPlates; ++i)
    {
        plateData[i].pixelCenter = {
            plateCenters[i].position.x - floor(plateCenters[i].position.x),
            plateCenters[i].position.y - floor(plateCenters[i].position.y)
        };
        plateData[i].velocity = (dist(generator) + 1) / 2;
        const float x_dir = dist(generator);
        const float y_dir = dist(generator);
        const float magnitude = sqrt(x_dir * x_dir + y_dir * y_dir);


        plateData[i].direction = Vec2(x_dir / magnitude, y_dir / magnitude);
    }

    return plateData;
}

void PlateTectonicSim::initializeTextures()
{
    m_heightMapTexture = m_textureManager.generateTexture<float>(m_width, m_height);
    m_plateIdsTexture = m_textureManager.generateTexture<uint8_t>(m_width, m_height);

    m_hydrationLevel = m_textureManager.generateTexture<float>(m_width, m_height);
    m_hydrationFlux = m_textureManager.generateTexture<float4>(m_width, m_height);
    m_hydrationVelocity = m_textureManager.generateTexture<Vec2<float> >(m_width, m_height);
    m_sedimentLevel = m_textureManager.generateTexture<float>(m_width, m_height);

    m_plateDataLookup = m_textureManager.generateTexture<PlateData>(MAX_PLATE_COUNT, 1);
    m_randStatesPlates = m_textureManager.generateTexture<curandState>(MAX_PLATE_COUNT, 1);
    m_iterationStats = m_textureManager.generateTexture<IterationStatistics>(1, 1);

    cudaDeviceSynchronize();
}

std::vector<VoronoiSeed> PlateTectonicSim::generatePlateCenters(std::default_random_engine &generator,
                                                                const int numPlates,
                                                                const int width, const int height)
{
    std::vector<VoronoiSeed> plateCenters(numPlates);

    std::uniform_real_distribution<float> randomHeight(0, static_cast<float>(height));
    std::uniform_real_distribution<float> randomWidth(0, static_cast<float>(width));

    for (int i = 0; i != numPlates; ++i)
    {
        plateCenters[i].id = i;
        plateCenters[i].position = {randomHeight(generator), randomWidth(generator)};
    }

    return plateCenters;
}

std::vector<VoronoiSeed> PlateTectonicSim::generateVoronoiSeeds(std::default_random_engine &generator,
                                                                const std::vector<VoronoiSeed> &centerSeeds,
                                                                const int numSeeds,
                                                                const int width, const int height)
{
    std::vector<VoronoiSeed> seeds(numSeeds);

    std::uniform_real_distribution<float> randomHeight(0, static_cast<float>(height));
    std::uniform_real_distribution<float> randomWidth(0, static_cast<float>(width));

    for (int i = 0; i != numSeeds; ++i)
    {
        float minDistance = FLT_MAX;
        uint8_t closestID = 0;
        const Vec2 seed{randomHeight(generator), randomWidth(generator)};
        for (const auto &[position, id]: centerSeeds)
        {
            const float diffX = abs(seed.x - position.x);
            const float diffY = abs(seed.y - position.y);

            const float wrapAdjustedX = min(diffX, static_cast<float>(width) - diffX);
            const float wrapAdjustedY = min(diffY, static_cast<float>(height) - diffY);

            if (const float distanceSq = wrapAdjustedX * wrapAdjustedX + wrapAdjustedY * wrapAdjustedY;
                distanceSq < minDistance)
            {
                closestID = id;
                minDistance = distanceSq;
            }
        }

        seeds[i].position = seed;
        seeds[i].id = closestID;
    }

    return seeds;
}

void PlateTectonicSim::HydrationSubSim() const
{}
