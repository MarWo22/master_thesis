#include "plate_tectonic_sim.h"

#include <iostream>
#include <map>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "ccl.cuh"
#include "cuda_gl_interop_manager.h"
#include "types/plate_data.h"
#include "plate_tectonics_kernel.cuh"
#include "../generation_settings.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include "texture_save.cuh"
#include "kernel_settings.cuh"

extern RenderSettings renderSettings;
extern SimulationSettings simulationSettings;
extern SaveTextureGui saveTextureGui;
extern KernelSettings kernelSettingsHost;
extern std::array<GuiPlateData, MAX_PLATE_COUNT> guiPlateData;

PlateTectonicSim::PlateTectonicSim(const int width, const int height, const unsigned int seed,
                                   CudaGlInteropManager *interopManager)
    : m_width(width)
      , m_height(height)
      , m_interopManager(interopManager)
      , m_seed(seed)
      , m_numBlocksPixels((width * height + 1) / THREADS_PER_BLOCK)
      , m_iterations(0)
{}

PlateTectonicSim::~PlateTectonicSim() {}

void PlateTectonicSim::initialize(const int numStartingPlates, const std::vector<int> &numVoronoiSeeds)
{
    cudaFree(nullptr); // force context initialization
    initializeTextures();
    std::cout << "Initialized textured\n";
    initializeTectonics(numStartingPlates, numVoronoiSeeds);
    std::cout << "Initialized tectonics\n";
    setupGUICallbacks();

    if (m_interopManager != nullptr)
        onRenderSettingChange();
}


void PlateTectonicSim::executeIteration()
{
    const auto start{std::chrono::steady_clock::now()};
    std::cout << "Executing iteration" << std::endl;

    m_plateCollisions->memsetTexture(0, true);
    getPlateCollisions(m_plateCollisions.get());

    // Process colliding plates and perform uplift
    auto heightMapTextureWrite = m_textureManager.generateTexture<float>(m_width, m_height);
    auto plateIdsTextureWrite = m_textureManager.generateTexture<uint8_t>(m_width, m_height);

    const auto plateVelocityChanges = m_textureManager.generateTextureAndReset<CollisionVelocityChanges>(
        MAX_PLATE_COUNT, 1, 0);

    processCollisionUplift(heightMapTextureWrite.get(), plateIdsTextureWrite.get(), m_plateCollisions.get(),
                           plateVelocityChanges.get());


    thermalErosionKernel << <m_numBlocksPixels, THREADS_PER_BLOCK >> >(heightMapTextureWrite->deviceTexture());

    //applyHydraulicErosion(heightMapTextureWrite.get());

    copyAndReleaseTexture(*m_heightMapTexture, std::move(heightMapTextureWrite), m_height, m_width);
    copyAndReleaseTexture(*m_plateIdsTexture, std::move(plateIdsTextureWrite), m_height, m_width);


    /*
     * STEP SIX
     * Apply the movement to the plates, and update the velocities and directions according to collisions.
     */


    applyPlateMovementChanges << <NUM_BLOCKS_PLATES, THREADS_PER_BLOCK >> >(
        m_plateDataLookup->getPointer(), plateVelocityChanges->getPointer());

    applyAccretion<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_heightMapTexture->deviceTexture(),
                                                             m_accretionTexture->deviceTexture(),
                                                             m_plateIdsTexture->deviceTexture());

    if (m_iterations % simulationSettings.iterationsForCCL == 0)
        applyCCL();

    mergePlates();

    // Compute perimeter-area ratios after finalPixelPass updates the plate data
    computeBreakScore<<<MAX_PLATE_COUNT, 1>>>(m_plateDataLookup->getPointer());


    auto plateAngularSums = m_textureManager.generateTexture<float4>(MAX_PLATE_COUNT, 1);
    auto plateCounts = m_textureManager.generateTexture<int>(MAX_PLATE_COUNT, 1);

    accumulatePlateAngularCoords<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(
        m_plateIdsTexture->deviceTexture(), plateAngularSums->deviceTexture(), plateCounts->deviceTexture());
    cudaDeviceSynchronize();
    calculatePlateCenters<<<MAX_PLATE_COUNT, 1>>>(m_plateIdsTexture->deviceTexture(), plateAngularSums->deviceTexture(),
                                                  plateCounts->deviceTexture(), m_plateDataLookup->getPointer());

    statisticsPass<<<1, 1>>>(m_plateDataLookup->getPointer(), m_iterationStats->getPointer());

    processPlateSplitting();

    CUDA_ERROR_CHECK();

    cudaDeviceSynchronize();

    if (m_interopManager)
    {
        copyConstantTexturesInterop();
    }

    m_iterations++;

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "Iteration duration: " << elapsed_seconds.count() << std::endl;
}

void PlateTectonicSim::copyConstantTexturesInterop() const
{
    m_interopManager->copyConnection("heightMap", m_heightMapTexture->getPointer());

    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_PLATE_VELOCITIES)
        copyVelocitiesGL();

    if (renderSettings.renderDirections)
        copyDirectionGL();

    if (renderSettings.copyPlateData)
        copyPlateDataGui();


    if (renderSettings.borderRenderMode == RenderSettings::BorderRenderMode::RAW_BORDER ||
        renderSettings.borderRenderMode == RenderSettings::BorderRenderMode::SMOOTH_BORDER ||
        renderSettings.shadingMode == RenderSettings::ShadingMode::SHOW_PLATE_IDS)
    {
        std::cout << "Copying palte id\n";
        m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture->getPointer());
    }

    if (renderSettings.renderCollisionBorders)
    {
        m_interopManager->copyConnection("collisionMap", m_plateCollisions->getPointer());
    }

    if (renderSettings.renderAccretionPixels)
    {
        m_interopManager->copyConnection("accretionTexture", m_accretionTexture->getPointer());
    }

    if (renderSettings.renderWater)
        m_interopManager->copyConnection("waterTexture", m_hydrationLevel->getPointer());

    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_PRESSURE_AREAS)
        m_interopManager->copyConnection("pressureTexture", m_pressure->getPointer());

    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_STRESS_AREAS)
        m_interopManager->copyConnection("stressTexture", m_stress->getPointer());
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
    CUDA_ERROR_CHECK();

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
    CUDA_ERROR_CHECK();
    // pixelIndicesCollisions, plateIdsCollisions, exclusivePrefixSum get released again
}


void PlateTectonicSim::processCollisionUplift(CudaTextureHost<float> *heightMapTextureWrite,
                                              CudaTextureHost<uint8_t> *plateIdsTextureWrite,
                                              CudaTextureHost<uint32_t> *plateCollisions,
                                              CudaTextureHost<CollisionVelocityChanges> *velocityChanges)
{
    const auto upliftData = m_textureManager.generateTexture<UpliftData>(m_width, m_height);
    const auto blurBuffer = m_textureManager.generateTexture<DistanceFieldBuffer>(m_width, m_height);

    /*
     * STEP FOUR
     * Main bulk of work. Each pixel updates depending on the presence of plates. If there are no plates present in
     * a pixel, it indicates the divergence of plates, triggering the creation of new oceanic crust in that plate. This
     * new crust is assigned to the plate last seen in this location (CURRENTLY RANDOM CHOSEN, BUT SHOULD LIKELY CHANGE).
     * The presence of one plate indicates a simple movement, and more indicates a convergence. Generation of new crust
     * and movement of original crust is dealt with in this step.
     */

    const auto hasDivergedBitmap = m_textureManager.generateTextureAndReset<uint32_t>(
        NUM_WORDS_TRIANGLE_SINGLE_BITS, 1, 0);

    const PlateTexturesRead readTextures = {
        m_plateIdsTexture->deviceTexture(), m_heightMapTexture->deviceTexture(), m_plateDataLookup->getPointer()
    };

    const PlateTexturesWrite writeTextures = {
        plateIdsTextureWrite->deviceTexture(), heightMapTextureWrite->deviceTexture()
    };


    const auto collisionTypeCounts = m_textureManager.generateTextureAndReset<CollisionTypeCounts>(
        MAX_PLATE_COUNT, MAX_PLATE_COUNT, 0);

    determineCollisionType<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(readTextures, plateCollisions->deviceTexture(),
                                                                     m_collisionTypeBitmap->getPointer(),
                                                                     collisionTypeCounts->getPointer());
    CUDA_ERROR_CHECK();

    createCollisionTypeMatrix<<<NUM_BLOCKS_TRIANGLE_ENTRIES, THREADS_PER_BLOCK>>>(
        collisionTypeCounts->getPointer(), m_collisionTypeBitmap->getPointer());
    CUDA_ERROR_CHECK();


    m_accretionTexture->memsetTexture(0, true);

    processCollisions<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(readTextures,
                                                                plateCollisions->deviceTexture(),
                                                                m_divergenceBitmap->getPointer(),
                                                                m_collisionTypeBitmap->getPointer(),
                                                                writeTextures,
                                                                hasDivergedBitmap->getPointer(),
                                                                upliftData->deviceTexture(),
                                                                m_accretionTexture->deviceTexture(),
                                                                velocityChanges->getPointer(),
                                                                {static_cast<int>(m_seed), m_iterations});

    CUDA_ERROR_CHECK();
    flipDivergedBitmap<<<NUM_BLOCKS_TRIANGLE_SINGLE_BITS, THREADS_PER_BLOCK>>>(
        m_divergenceBitmap->getPointer(), hasDivergedBitmap->getPointer());
    CUDA_ERROR_CHECK();

    /*
     * STEP FIVE
     * Perform uplift
     */

    VerticalBlur<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(plateCollisions->deviceTexture(),
                                                           blurBuffer->deviceTexture());
    CUDA_ERROR_CHECK();

    HorizontalBlur<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(plateIdsTextureWrite->deviceTexture(),
                                                             plateCollisions->deviceTexture(),
                                                             blurBuffer->deviceTexture(), upliftData->deviceTexture(),
                                                             heightMapTextureWrite->deviceTexture());
    CUDA_ERROR_CHECK();
}

void PlateTectonicSim::applyHydraulicErosion(CudaTextureHost<float> *heightMapTextureWrite)
{
    const auto fluxBuffer = m_textureManager.generateTexture<float4>(m_width, m_height);
    const auto sedimentBuffer = m_textureManager.generateTexture<float>(m_width, m_height);

    for (size_t i = 0; i < 10; i++)
    {
        rain<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_hydrationLevel->deviceTexture(), 1.0, m_seed + m_iterations);
        CUDA_ERROR_CHECK();
        flux<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(heightMapTextureWrite->deviceTexture(),
                                                       m_hydrationLevel->deviceTexture(),
                                                       m_hydrationFlux->deviceTexture(), fluxBuffer->deviceTexture(),
                                                       1.0);
        CUDA_ERROR_CHECK();
        flow<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_hydrationLevel->deviceTexture(),
                                                       fluxBuffer->deviceTexture(),
                                                       m_hydrationFlux->deviceTexture(),
                                                       m_hydrationVelocity->deviceTexture(), 1.0);
        CUDA_ERROR_CHECK();
        sediment<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(heightMapTextureWrite->deviceTexture(),
                                                           m_sedimentLevel->deviceTexture(),
                                                           m_hydrationVelocity->deviceTexture(), 1.0);
        CUDA_ERROR_CHECK();
        transport<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_sedimentLevel->deviceTexture(),
                                                            sedimentBuffer->deviceTexture(),
                                                            m_hydrationVelocity->deviceTexture(), 1.0);
        CUDA_ERROR_CHECK();
        evaporate<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_hydrationLevel->deviceTexture(), 1.0);

        CUDA_ERROR_CHECK();

        cudaMemcpyAsync(m_sedimentLevel->getPointer(), sedimentBuffer->getPointer(), sizeof(float) * m_width * m_height,
                        cudaMemcpyDeviceToDevice);
        CUDA_ERROR_CHECK();
    }
}

void PlateTectonicSim::applyCCL()
{
    std::cout << "Executing CCL\n";
    const auto labels = m_textureManager.generateTexture<unsigned int>(m_width, m_height);


    // First, apply 8-way CCL to generate a texture of labels
    init<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(), labels->deviceTexture());
    CUDA_ERROR_CHECK();
    analyzeClamped<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(labels->deviceTexture());
    CUDA_ERROR_CHECK();
    reduce<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(), labels->deviceTexture());
    CUDA_ERROR_CHECK();
    analyzeUnclamped<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(labels->deviceTexture());
    CUDA_ERROR_CHECK();

    const auto labelIdsPacked = m_textureManager.generateTexture<uint64_t>(m_width, m_height);
    createPlateIdLabelMap<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                    labels->deviceTexture(),
                                                                    labelIdsPacked->deviceTexture());

    const int labelCountSize = static_cast<int>(m_width * m_height * 0.01);
    auto labelCounts = m_textureManager.generateTexture<unsigned int>(labelCountSize, 1);
    auto uniqueLabels = m_textureManager.generateTexture<uint64_t>(labelCountSize, 1);
    // Create thrust device ptr wrappers
    const thrust::device_ptr<uint64_t> labelsIdsPackedThrust(labelIdsPacked->getPointer());
    const thrust::device_ptr<unsigned int> labelCountsThrust(labelCounts->getPointer());
    const thrust::device_ptr<uint64_t> uniqueLabelsThrust(uniqueLabels->getPointer());

    // First, sort the labels
    sort(labelsIdsPackedThrust, labelsIdsPackedThrust + m_height * m_width,
         thrust::greater<unsigned int>());

    // Second, apply a reduce by key to create an array of key:count pairs. Capture the end iterator such that we are
    // aware of the size of the final array.

    const auto resultEnd = reduce_by_key(labelsIdsPackedThrust, labelsIdsPackedThrust + m_height * m_width,
                                         thrust::make_constant_iterator<int>(1), uniqueLabelsThrust,
                                         labelCountsThrust);

    const int size = static_cast<int>(resultEnd.first - uniqueLabelsThrust);

    int gridSize = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    sort_by_key(labelCountsThrust, labelCountsThrust + size, uniqueLabelsThrust,
                thrust::greater<unsigned int>());

    const auto idsMaxCounts = m_textureManager.generateTextureAndReset<int>(MAX_PLATE_COUNT, 1, 0);
    getMaxSizeLabels<<<gridSize, THREADS_PER_BLOCK>>>(uniqueLabels->deviceTexture(), labelCounts->deviceTexture(),
                                                      idsMaxCounts->deviceTexture(), size);

    const auto labelIdLookup = m_textureManager.generateTextureAndReset<uint8_t>(m_width, m_height, MAX_PLATE_COUNT);
    createLabelIdLookup<<<gridSize, THREADS_PER_BLOCK>>>(uniqueLabels->deviceTexture(), labelCounts->deviceTexture(),
                                                         idsMaxCounts->deviceTexture(), m_plateDataLookup->getPointer(),
                                                         labelIdLookup->deviceTexture(), size);

    const auto unassignedIndicesCount = m_textureManager.generateTextureAndReset<int>(1, 1, 0);
    const auto unassignedIndices = m_textureManager.generateTexture<unsigned int>(m_width, m_height);
    postCClIdReassign<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(labelIdLookup->deviceTexture(), labels->deviceTexture(),
                                                                m_plateIdsTexture->deviceTexture(),
                                                                unassignedIndices->deviceTexture(),
                                                                unassignedIndicesCount->getPointer());

    int remainingIndices;
    if (const cudaError_t err = cudaMemcpy(&remainingIndices, unassignedIndicesCount->getPointer(),
                                           sizeof(int), cudaMemcpyDeviceToHost); err != cudaSuccess)
        std::cerr << "Error memcpy remainingIndices: " << cudaGetErrorString(err) << "\n";


    CUDA_ERROR_CHECK();
    labelCounts.reset();
    uniqueLabels.reset();
    if (remainingIndices)
    {
        // Using int instead of boolean since cuda uses ints
        gridSize = (remainingIndices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bool hasWork = true;
        while (hasWork)
        {
            const auto hasRemainingWork = m_textureManager.generateTextureAndReset<bool>(1, 1, 0);

            assignUnassignedIdsToNeighbor<<<gridSize, THREADS_PER_BLOCK>>>(
                m_plateIdsTexture->deviceTexture(), unassignedIndices->getPointer(), remainingIndices,
                hasRemainingWork->getPointer());
            CUDA_ERROR_CHECK();
            if (const cudaError_t err = cudaMemcpy(&hasWork, hasRemainingWork->getPointer(), sizeof(bool),
                                                   cudaMemcpyDeviceToHost);
                err != cudaSuccess)
                std::cerr << "Error memcpy hasWork: " << cudaGetErrorString(err) << "\n";
        }
    }
}

void PlateTectonicSim::mergePlates()
{
    const auto neighborMatrix = m_textureManager.generateTextureAndReset<bool>(MAX_PLATE_COUNT, MAX_PLATE_COUNT, 0);
    getNeighboringPlates<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                   neighborMatrix->deviceTexture());
    CUDA_ERROR_CHECK();


    const auto plateMergeIds = m_textureManager.generateTextureAndReset<int>(MAX_PLATE_COUNT, 1, 0);
    getPlateMerges<<<NUM_BLOCKS_PLATES_MATRIX, THREADS_PER_BLOCK>>>(neighborMatrix->deviceTexture(),
                                                                    m_plateDataLookup->getPointer(),
                                                                    plateMergeIds->getPointer(),
                                                                    m_collisionTypeBitmap->deviceTexture());
    CUDA_ERROR_CHECK();


    resetPlateDataPreCount<<<NUM_BLOCKS_PLATES, THREADS_PER_BLOCK>>>(m_plateDataLookup->getPointer());

    mergeAndCountSizeMass<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                    m_heightMapTexture->deviceTexture(),
                                                                    plateMergeIds->getPointer(),
                                                                    m_plateDataLookup->getPointer(),
                                                                    m_pressureVelocity->deviceTexture());
    CUDA_ERROR_CHECK();
}

void PlateTectonicSim::processPlateSplitting()
{
    auto buffer = m_textureManager.generateTexture<float>(m_width, m_height);

    // Step 1: Accumulate pressure and reset at fault lines
    pressureAccumulation<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_pressure->deviceTexture(),
                                                                   m_pressure->deviceTexture(),
                                                                   m_plateIdsTexture->deviceTexture(),
                                                                   m_heightMapTexture->deviceTexture());

    // Step 2-3: Apply pressure blur simulation
    pressureVerticalBlur<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                   m_pressure->deviceTexture(),
                                                                   buffer->deviceTexture());

    pressureHorizontalBlur<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                     buffer->deviceTexture(),
                                                                     m_pressure->deviceTexture());

    // Calculate pressure velocity from pressure gradients
    calculatePressureVelocity<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_pressure->deviceTexture(),
                                                                        m_plateIdsTexture->deviceTexture(),
                                                                        m_pressureVelocity->deviceTexture());

    // Step 4: Calculate stress from pressure and terrain height
    stress<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_pressure->deviceTexture(),
                                                     m_heightMapTexture->deviceTexture(),
                                                     m_stress->deviceTexture(),
                                                     m_plateIdsTexture->deviceTexture(),
                                                     m_plateDataLookup->getPointer());

    const size_t numPixels = m_width * m_height;

    thrust::device_ptr<uint8_t> keys_ptr(m_plateIdsTexture->getPointer());
    thrust::device_ptr<float> values_ptr(m_stress->getPointer());

    // Find the pixel with maximum stress across all plates
    auto max_pixel_iter = thrust::max_element(values_ptr, values_ptr + numPixels);
    size_t max_pixel_index = max_pixel_iter - values_ptr;

    float h_highest_stress = *max_pixel_iter;
    uint8_t h_highest_stress_plate_id = keys_ptr[max_pixel_index];
    Vec2<float> h_stress_location = Vec2<float>(max_pixel_index % m_width, max_pixel_index / m_width);

    if (h_highest_stress > kernelSettingsHost.stressSplitThreshold)
    {
        SplitPlateV2(
            Vec2<int>(static_cast<int>(h_stress_location.x + 0.5), static_cast<int>(h_stress_location.y + 0.5)),
            h_highest_stress_plate_id);

        return;

        auto d_dir = m_textureManager.generateTexture<Vec2<float> >(1, 1);

        findPlausibleSplitLine<<<1, 10, 10 * sizeof(float)>>>(h_highest_stress_plate_id, h_stress_location,
                                                              m_plateIdsTexture->deviceTexture(), d_dir->getPointer());
        CUDA_ERROR_CHECK();
        auto newPlateId = m_textureManager.generateTexture<uint8_t>(1, 1);

        selectUnusedPlateId<<<1, 1>>>(m_plateDataLookup->getPointer(), newPlateId->getPointer());

        splitPlate << <m_numBlocksPixels, THREADS_PER_BLOCK >> >(h_highest_stress_plate_id,
                                                                 newPlateId->getPointer(), h_stress_location,
                                                                 d_dir->getPointer(),
                                                                 m_plateIdsTexture->deviceTexture(),
                                                                 m_plateDataLookup->getPointer());
        CUDA_ERROR_CHECK();
    }
}

Vec2<float> PlateTectonicSim::getPlateCenter()
{
    /*auto d_samples = m_textureManager.generateTexture<float4>(m_numBlocksPixels, 1);
    findPlateCenter<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_iterationStats->getPointer(),
                                                              m_plateDataLookup->getPointer(),
                                                              m_plateIdsTexture->deviceTexture(),
                                                              d_samples->getPointer());
    CUDA_ERROR_CHECK();
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

    delete[] h_samples;*/

    //return Vec2(midX, midY);

    return Vec2<float>(0, 0);
}

void PlateTectonicSim::copyDirectionGL() const
{
    CudaTextureHost<float2> glTexture;
    glTexture.initialize(m_width / 16, m_height / 16);
    int numBlocks = ((m_width / 16) * (m_height / 16) + 1) / THREADS_PER_BLOCK;
    createDirectionTexture<<<numBlocks, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                             m_plateDataLookup->getPointer(),
                                                             glTexture.deviceTexture());
    CUDA_ERROR_CHECK();
    m_interopManager->copyConnection("directionTexture", glTexture.getPointer());
}

void PlateTectonicSim::copyVelocitiesGL() const
{
    CudaTextureHost<float> glTexture;
    glTexture.initialize(m_width, m_height);
    createVelocityTexture<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                    m_plateDataLookup->getPointer(),
                                                                    glTexture.deviceTexture());
    CUDA_ERROR_CHECK();
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
    m_pressure.reset();
    m_stress.reset();
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

void PlateTectonicSim::onRenderSettingChange()
{
    std::vector<std::string> activeTextures;

    switch (renderSettings.renderMode)
    {
        case RenderSettings::RenderMode::NORMAL:
            activeTextures = {"heightMap"};
            break;
        case RenderSettings::RenderMode::SHOW_PRESSURE_AREAS:
            activeTextures = {"heightMap", "pressureTexture"};
            break;
        case RenderSettings::RenderMode::SHOW_STRESS_AREAS:
            activeTextures = {"heightMap", "stressTexture"};
            break;
        case RenderSettings::RenderMode::SHOW_PLATE_VELOCITIES:
            activeTextures = {"heightMap", "velocityTexture"};
            break;
        default:
            activeTextures = {"heightMap"};
            break;
    }
    if (renderSettings.borderRenderMode == RenderSettings::BorderRenderMode::RAW_BORDER ||
        renderSettings.borderRenderMode == RenderSettings::BorderRenderMode::SMOOTH_BORDER ||
        renderSettings.shadingMode == RenderSettings::ShadingMode::SHOW_PLATE_IDS)
    {
        activeTextures.emplace_back("cudaPlateTexture");
    }
    if (renderSettings.renderCollisionBorders)
    {
        activeTextures.emplace_back("collisionMap");
    }

    if (renderSettings.renderAccretionPixels)
        activeTextures.emplace_back("accretionTexture");

    if (renderSettings.renderWater)
        activeTextures.emplace_back("waterTexture");
    if (renderSettings.renderDirections)
        activeTextures.emplace_back("directionTexture");

    m_interopManager->toggleSubTextures(activeTextures);
    copyConstantTexturesInterop();
}

void PlateTectonicSim::copyPlateDataGui() const
{
    std::cout << "Copying plate data gui\n";
    GuiPlateData *gpuPlateDataDevice;

    cudaMalloc(&gpuPlateDataDevice, sizeof(GuiPlateData) * MAX_PLATE_COUNT);
    CUDA_ERROR_CHECK();
    copyPlateDataGuiKernel<<<NUM_BLOCKS_PLATES, THREADS_PER_BLOCK>>>(m_plateDataLookup->getPointer(),
                                                                     m_collisionTypeBitmap->getPointer(),
                                                                     gpuPlateDataDevice);
    CUDA_ERROR_CHECK();
    cudaMemcpy(guiPlateData.data(), gpuPlateDataDevice, sizeof(GuiPlateData) * MAX_PLATE_COUNT, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();
    cudaFree(gpuPlateDataDevice);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK();
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

    if (const cudaError_t err = cudaMemcpy(m_plateDataLookup->getPointer(), plateDataHost.data(),
                                           sizeof(PlateData) * MAX_PLATE_COUNT,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    // Each pixel is assigned the ID of the nearest voronoi seed
    // The plateIDs are written to the m_plateIdsTexture texture
    initPlateIDs<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                           voronoiSeedsDevice.getPointer(),
                                                           static_cast<int>(voronoiSeedsHost.size()));
    CUDA_ERROR_CHECK();
    std::cout << "Initialized plate IDs\n";
    // Initialized the heightmap with simplex noise
    // The heightmap is written to the m_heightMapTexture texture

    initHeightmap<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_heightMapTexture->deviceTexture(), m_seed, 4);

    CUDA_ERROR_CHECK();
    std::cout << "Initialized heightmap\n";

    // Extracts the size and mass of the plates
    // This is stored in the m_plateDataLookup lookup texture
    initPixelDependantPlateData<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                          m_heightMapTexture->deviceTexture(),
                                                                          m_plateDataLookup->getPointer());
    CUDA_ERROR_CHECK();

    applyCCL();

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

    renderSettings.registerCallback("renderSettingsChanged", [this]
    {
        onRenderSettingChange();
    });

    renderSettings.registerCallback("copyPlateInfo", [this]
    {
        copyPlateDataGui();
    });
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
    m_plateCollisions = m_textureManager.generateTextureAndReset<uint32_t>(m_width, m_height, 0);
    m_hydrationLevel = m_textureManager.generateTexture<float>(m_width, m_height);
    m_hydrationFlux = m_textureManager.generateTexture<float4>(m_width, m_height);
    m_hydrationVelocity = m_textureManager.generateTexture<Vec2<float> >(m_width, m_height);
    m_sedimentLevel = m_textureManager.generateTexture<float>(m_width, m_height);

    m_pressure = m_textureManager.generateTexture<float>(m_width, m_height);
    m_stress = m_textureManager.generateTexture<float>(m_width, m_height);
    m_pressureVelocity = m_textureManager.generateTexture<Vec2<float> >(m_width, m_height);

    m_plateDataLookup = m_textureManager.generateTexture<PlateData>(MAX_PLATE_COUNT, 1);
    m_randStatesPlates = m_textureManager.generateTexture<curandState>(MAX_PLATE_COUNT, 1);
    m_iterationStats = m_textureManager.generateTexture<IterationStatistics>(1, 1);

    m_divergenceBitmap = m_textureManager.generateTextureAndReset<uint32_t>(NUM_WORDS_TRIANGLE_SINGLE_BITS, 1, 0);
    m_collisionTypeBitmap = m_textureManager.generateTextureAndReset<uint8_t>(NUM_TRIANGLE_ENTRIES, 1, 0);

    m_accretionTexture = m_textureManager.generateTexture<uint8_t>(m_width, m_height);

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

void PlateTectonicSim::SplitPlateV2(Vec2<int> point, uint8_t oldPlateId)
{
    std::cout << static_cast<int>(oldPlateId) << " is splitting\n";
    auto l_effortToBoundaryMap = m_textureManager.generateTextureAndReset<float>(m_width, m_height, 0);
    auto l_effortToBoundaryMapNext = m_textureManager.generateTexture<float>(m_width, m_height);
    auto hasChangedDevice = m_textureManager.generateTextureAndReset<int>(1, 1, 0);

    initEffortToBoundary<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                   l_effortToBoundaryMap->deviceTexture(), oldPlateId);

    // Ping-pong buffer iterations with early exit
    constexpr int maxIterations = 10000;
    int hasChangedHost = 1;
    int iteration = 0;

    while (hasChangedHost && iteration < maxIterations)
    {
        // Reset flag
        cudaMemset(hasChangedDevice->getPointer(), 0, sizeof(int));

        propagateEffortToBoundary<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(
            l_effortToBoundaryMap->deviceTexture(),
            l_effortToBoundaryMapNext->deviceTexture(),
            m_heightMapTexture->deviceTexture(),
            hasChangedDevice->getPointer()
        );

        // Check if any pixel changed
        cudaMemcpy(&hasChangedHost, hasChangedDevice->getPointer(), sizeof(int), cudaMemcpyDeviceToHost);

        l_effortToBoundaryMap.swap(l_effortToBoundaryMapNext);

        iteration++;
    }

    std::cout << "Converged after " << iteration << " iterations" << std::endl;

    auto newPlateId = m_textureManager.generateTexture<uint8_t>(1, 1);

    selectUnusedPlateId << <1, 1 >> >(m_plateDataLookup->getPointer(), newPlateId->getPointer());

    uint8_t newPlateIdHost;
    cudaMemcpy(&newPlateIdHost, newPlateId->getPointer(), sizeof(uint8_t), cudaMemcpyDeviceToHost);


    int *deadEndDevice;
    cudaMalloc(&deadEndDevice, sizeof(int));
    int deadEndHost = 0;
    cudaMemcpy(deadEndDevice, &deadEndHost, sizeof(int), cudaMemcpyHostToDevice);

    BacktrackPath<<<1, 1>>>(l_effortToBoundaryMap->deviceTexture(), m_plateIdsTexture->deviceTexture(), point,
                            newPlateIdHost, deadEndDevice);

    cudaMemcpy(&deadEndHost, deadEndDevice, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(deadEndDevice);

    if (deadEndHost)
    {
        std::cout << "Backtrack path led to a dead end, aborting split" << std::endl;

        finalizePlateSplit << <m_numBlocksPixels, THREADS_PER_BLOCK >> >(
            m_plateIdsTexture->deviceTexture(),
            oldPlateId,
            oldPlateId,
            m_plateDataLookup->getPointer()
        );

        return;
    }

    // Flood fill one side of the split with the new plate ID

    int floodFillIterations = 0;
    hasChangedHost = 1;
    constexpr int maxFloodFillIterations = 10000;

    while (hasChangedHost && floodFillIterations < maxFloodFillIterations)
    {
        cudaMemset(hasChangedDevice->getPointer(), 0, sizeof(int));

        floodFillPlate<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(
            m_plateIdsTexture->deviceTexture(),
            oldPlateId,
            newPlateIdHost,
            hasChangedDevice->getPointer()
        );

        cudaMemcpy(&hasChangedHost, hasChangedDevice->getPointer(), sizeof(int), cudaMemcpyDeviceToHost);
        floodFillIterations++;
    }

    std::cout << "Flood fill completed after " << floodFillIterations << " iterations" << std::endl;

    // Reset the dividing line (marked with MAX_PLATE_COUNT) to the new plate ID
    finalizePlateSplit <<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(
        m_plateIdsTexture->deviceTexture(),
        oldPlateId,
        newPlateIdHost,
        m_plateDataLookup->getPointer()
    );

    resetPlateDataPreCount<<<NUM_BLOCKS_PLATES, THREADS_PER_BLOCK>>>(m_plateDataLookup->getPointer());
    countSizePostSplit<<<m_numBlocksPixels, THREADS_PER_BLOCK>>>(m_plateIdsTexture->deviceTexture(),
                                                                 m_heightMapTexture->deviceTexture(),
                                                                 m_plateDataLookup->getPointer());

    std::cout << "Split plate!" << std::endl;
}
