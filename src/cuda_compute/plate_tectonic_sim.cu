#include "plate_tectonic_sim.h"

#include <iostream>
#include <map>
#include <random>

#include "ccl.cuh"
#include "random_texture.h"
#include "cuda_gl_interop_manager.h"
#include "plate_data.h"
#include "plate_tectonics_kernel.cuh"
#include "../generation_settings.h"
#include "flux_erosion.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>

extern RenderSettings renderSettings;
extern SimulationSettings simulationSettings;

PlateTectonicSim::PlateTectonicSim(const int width, const int height, const unsigned int seed, const int numStartingPlates,
                                   CudaGlInteropManager *interopManager)
    : m_width(width)
    , m_height(height)
    , m_seed(seed)
    , m_plateDataLookup(nullptr)
    , m_randStatesPlates(nullptr)
    , m_iterationStats(nullptr)
    , m_numStartingPlates(numStartingPlates)
    , m_interopManager(interopManager)
{
    // FluxVelocityErosion erosion = FluxVelocityErosion(height, width);
    //
    // erosion.mCapacityConstant = 4;
    // erosion.mDissolvingConstant = 0.1f;
    // erosion.mPipeLengthConstant = 0.5f;
    // erosion.mPipeCrossSectionConstant = 0.5f;
    // erosion.mGravityConstant = 9.81f;
    // erosion.mEvaporationConstant = 0.99;
    //
    // erosion.simulate(10);
    //
    // m_heightMapDevice = erosion.m_materialDevice.getPointer();
}

void PlateTectonicSim::initialize()
{
    initializeTectonics();
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

    simulationSettings.resetCallback = [this](unsigned int seed, int numPlates) {
        resetSim(seed, numPlates);
    };

    if (m_interopManager != nullptr)
    {
        m_interopManager->toggleSubTextures({"heightMap", "cudaPlateTexture"});
        copyConstantTexturesInterop();
        setupToggleCallbacks();
    }
}

void PlateTectonicSim::executeIteration()
{
    /*
     * Initialization of textures and such
     */

    const auto start{std::chrono::steady_clock::now()};
    std::cout << "Executing iteration" << std::endl;

    // Determine block numbers
    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    int numBlocksPlates = (m_maxPlates + m_threadsPerBlock - 1) / m_threadsPerBlock;
    int numBlocksMaxPlatesMatrix = (MAX_PLATE_COUNT * MAX_PLATE_COUNT +1) / m_threadsPerBlock;


    // Allocate the pixel plate pairs as separate arrays
    CudaTextureHost<unsigned int> pixelIndicesCollisions;
    CudaTextureHost<uint8_t> plateIdsCollisions;
    pixelIndicesCollisions.initialize(m_width, m_height);
    plateIdsCollisions.initialize(m_width, m_height);

    /*
     * STEP ONE
     * Move the pixels of plates according to their velocities and directions. Register any collisions into the
     * pixelIndicesCollisions and plateIdsCollisions textures. pixelIndices holds the unsigned int index of the pixel in
     * the original texture, and plateIds holds the plate moving to that pixel. The two textures are aligned, meaning that
     * the pixelIndex at position n corresponds to the plateId ate position n.
     */

    // Execute plate movement kernel
    testingPlateMovement<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup,
                                                                 pixelIndicesCollisions.deviceTexture(),
                                                                 plateIdsCollisions.deviceTexture());

    /*
     * STEP TWO
     * Perform a sort by key on the pixelIndices and plateIds textures to align the pixelIndices in ascending order while
     * remaining the alignment. Position n in pixelIndices still corresponds to position n in plateIds
     * This is followed by an exclusive prefix sum by key. The prefix sum output is stored in the exclusivePrefixSum texture
     * This texture will now contain the occurrence index of the pixel, aligned with the pixelIndicesCollision and
     * plateIdsCollisions texture
     */

    const thrust::device_ptr<unsigned int> pixelIndicesThrust(pixelIndicesCollisions.getPointer());
    const thrust::device_ptr<uint8_t> plateIdsThrust(plateIdsCollisions.getPointer());

    sort_by_key(pixelIndicesThrust, pixelIndicesThrust + m_height * m_width, plateIdsThrust,
                        thrust::greater<unsigned int>());

    // Allocate array for the exclusive prefix sum values
    CudaTextureHost<uint8_t> exclusivePrefixSum;
    exclusivePrefixSum.initialize(m_width, m_height, 1);

    // Perform exclusive scan
    const thrust::device_ptr<uint8_t> exclusivePrefixSumThrust(exclusivePrefixSum.getPointer());
    exclusive_scan_by_key(pixelIndicesThrust, pixelIndicesThrust + m_width * m_height, exclusivePrefixSumThrust,
                                  exclusivePrefixSumThrust);


    /*
     * STEP THREE
     * Go through each pair of entries of the pixelIndices, exclusivePrefixSum and plateCollisions textures and register
     * the collisions to the associated pixels. The collisions are stored in plateCollisions, a 32bit texture where each
     * entry has four packed 8bit values, indicating the presence of a plate in that pixel. A byte with value  0-255
     * indicates the presence of plate (255-n) in that position . A value of n=0 indicates there is no plate in that
     * specific byte. This allows for the detection of 0-4 plates in a pixel, any exceeds will be ignored.
     */

    CudaTextureHost<uint32_t> plateCollisions;
    plateCollisions.initialize(m_width, m_height, 0);

    registerPlateCollisions<<<numBlocksPixels, m_threadsPerBlock>>>(plateIdsCollisions.deviceTexture(),
                                                                    pixelIndicesCollisions.deviceTexture(),
                                                                    exclusivePrefixSum.deviceTexture(),
                                                                    plateCollisions.deviceTexture());

    // Free memory since we no longer need them
    plateIdsCollisions.free();
    pixelIndicesCollisions.free();
    exclusivePrefixSum.free();

    CudaTextureHost<float> heightMapTextureWrite;
    CudaTextureHost<uint8_t> plateIdsTextureWrite;
    CudaTextureHost<float> uplift;
    CudaTextureHost<uint8_t> platesHaveCollided;

    heightMapTextureWrite.initialize(m_width, m_height);
    plateIdsTextureWrite.initialize(m_width, m_height);
    uplift.initialize(m_width, m_height, 0);
    platesHaveCollided.initialize(m_maxPlates, m_maxPlates, 0);

    /*
     * STEP FOUR
     * Main bulk of work. Each pixel updates depending on the presence of plates. If there are no plates present in
     * a pixel, it indicates the divergence of plates, triggering the creation of new oceanic crust in that plate. This
     * new crust is assigned to the plate last seen in this location (CURRENTLY RANDOM CHOSEN, BUT SHOULD LIKELY CHANGE).
     * The presence of one plate indicates a simple movement, and more indicates a convergence. Generation of new crust
     * and movement of original crust is dealt with in this step.
     */

    processCollisions<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(),
                                                              m_heightMapTexture.deviceTexture(),
                                                              plateCollisions.deviceTexture(), m_plateDataLookup,
                                                              plateIdsTextureWrite.deviceTexture(),
                                                              heightMapTextureWrite.deviceTexture(),
                                                              uplift.deviceTexture(), platesHaveCollided.deviceTexture());

    /*
     * STEP FIVE
     * Perform uplift
     */

    processUplift << <numBlocksPixels, m_threadsPerBlock >> > (uplift.deviceTexture(), heightMapTextureWrite.deviceTexture(), 10, 0.1f, 20.f, m_seed);


    


    // TODO: Add when heightmap stuff is implemented
    if (const cudaError_t err = cudaMemcpy(m_heightMapTexture.getPointer(), heightMapTextureWrite.getPointer(), sizeof(float) * m_height * m_width, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy heightmap: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(m_plateIdsTexture.getPointer(), plateIdsTextureWrite.getPointer(), sizeof(uint8_t) * m_height * m_width, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy plateIds: " << cudaGetErrorString(err) << std::endl;

    heightMapTextureWrite.free();
    plateIdsTextureWrite.free();
    uplift.free();

    /*
     * STEP SIX
     * Apply the movement to the plates, and update the velocities and directions according to collisions.
     */

    updatePlateData<<<numBlocksPlates, m_threadsPerBlock>>>(m_plateDataLookup, m_randStatesPlates, Vec2(m_maxPlates, 1));

    uint8_t *plateMergeIds;
    if (const cudaError_t err = cudaMalloc(&plateMergeIds, sizeof(uint8_t) * MAX_PLATE_COUNT); err != cudaSuccess)
        std::cerr << "Error malloc plateMergeIds: " << cudaGetErrorString(err) << std::endl;
    if (const cudaError_t err = cudaMemset(plateMergeIds, MAX_PLATE_COUNT, MAX_PLATE_COUNT * sizeof(uint8_t)); err != cudaSuccess)
        std::cerr << "Error memset cuda texture: " << cudaGetErrorString(err) << std::endl;

    /*
     * STEP SEVEN
     * Final maxplatematrix and full-pixel pass that allow for the merging and splitting of plates, and the updates of
     * mass and sizes
     */

    determinePlateMerge<<<numBlocksMaxPlatesMatrix, m_threadsPerBlock>>>(platesHaveCollided.deviceTexture(), m_plateDataLookup, plateMergeIds);

    finalPixelPass<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_heightMapTexture.deviceTexture(), plateMergeIds, m_plateDataLookup);
    
    statisticsPass << <1, 1 >> > (m_plateDataLookup, m_iterationStats);

    int h_largest;
    int* d_largest = &(m_iterationStats->largestValue);
    cudaMemcpy(&h_largest, d_largest, sizeof(int), cudaMemcpyDeviceToHost);

    printf("largest: %.i", h_largest);
    if (h_largest > 120000) {
        Vec2<float> h_pivot = getPlateCenter(numBlocksPixels, m_threadsPerBlock);
        Vec2<float>* d_dir;

        if (const cudaError_t err = cudaMalloc(&d_dir, sizeof(Vec2<float>)); err != cudaSuccess)
            std::cerr << "Error malloc plateMergeIds: " << cudaGetErrorString(err) << std::endl;

        findPlausibleSplitLine << <1, 10, 10 * sizeof(float) >> > (m_iterationStats, h_pivot, m_plateIdsTexture.deviceTexture(), d_dir);

        uint8_t* newPlateId;
        if (const cudaError_t err = cudaMalloc(&newPlateId, sizeof(uint8_t)); err != cudaSuccess)
            std::cerr << "Error malloc plateMergeIds: " << cudaGetErrorString(err) << std::endl;

        selectUnusedPlateId << <1, 1 >> > (m_plateDataLookup, newPlateId);

        splitPlate << <numBlocksPixels, m_threadsPerBlock >> > (m_iterationStats, newPlateId, h_pivot, d_dir, m_plateIdsTexture.deviceTexture(), m_plateDataLookup);

        cudaFree(newPlateId);
        cudaFree(d_dir);
    }

    cudaDeviceSynchronize();

    /*
     * CCL SECTION
     */


    CudaTextureHost<unsigned int> labels;
    labels.initialize(m_width , m_height);

    // First, apply 8-way CCL to generate a texture of labels
    init<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), labels.deviceTexture());
    analyzeClamped<<<numBlocksPixels, m_threadsPerBlock>>>(labels.deviceTexture());
    reduce<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), labels.deviceTexture());
    analyzeUnclamped<<<numBlocksPixels, m_threadsPerBlock>>>(labels.deviceTexture());

    unsigned int *labelCounts;
    unsigned int *uniqueLabels;
    unsigned int *labelsCopy;

    if (const cudaError_t err = cudaMalloc(&labelCounts, sizeof(unsigned int) * m_width * m_height / 100); err != cudaSuccess)
        std::cerr << "Error malloc labelCounts: " << cudaGetErrorString(err) << "\n";

    if (const cudaError_t err = cudaMalloc(&uniqueLabels, sizeof(unsigned int) * m_width * m_height / 100); err != cudaSuccess)
        std::cerr << "Error malloc uniqueLabels: " << cudaGetErrorString(err) << "\n";

    if (const cudaError_t err = cudaMalloc(&labelsCopy, sizeof(unsigned int) * m_width * m_height); err != cudaSuccess)
        std::cerr << "Error malloc labelsCopy: " << cudaGetErrorString(err) << "\n";

    if (const cudaError_t err = cudaMemcpy(labelsCopy, labels.getPointer(), sizeof(unsigned int) * m_width * m_height, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy labelsCopy: " << cudaGetErrorString(err) << "\n";

    // Create thrust device ptr wrappers
    const thrust::device_ptr<unsigned int> labelsCopyThrust(labelsCopy);
    const thrust::device_ptr<unsigned int> labelCountsThrust(labelCounts);
    const thrust::device_ptr<unsigned int> uniqueLabelsThrust(uniqueLabels);

    std::vector<unsigned int> labelsHost(m_width * m_height);

    if (const cudaError_t err = cudaMemcpy(labelsHost.data(), labelsCopy, sizeof(unsigned int) * m_width * m_height, cudaMemcpyDeviceToHost); err != cudaSuccess)
        std::cerr << "Error memcpy labelsHost: " << cudaGetErrorString(err) << "\n";

    // First, sort the labels
    sort(labelsCopyThrust, labelsCopyThrust + m_height * m_width,
                        thrust::greater<unsigned int>());


    // Second, apply a reduce by key to create an array of key:count pairs. Capture the end iterator such that we are
    // aware of the size of the final array.
    const auto resultEnd = reduce_by_key(labelsCopyThrust, labelsCopyThrust + m_height * m_width, thrust::make_constant_iterator<int>(1), uniqueLabelsThrust, labelCountsThrust);
    cudaFree(labelsCopy);
    // We can use the iterator to calculate the number of unique labels. This is clamped to max_plate_count, as we cannot
    // allocate more plates than the max count anyway.
    const int numUniqueLabels = min(static_cast<int>(resultEnd.first - uniqueLabelsThrust), MAX_PLATE_COUNT);

    // Third, sort the section of the labels:count pairs that has been initialized in the previous step. This gives us
    // the pairs sorted by counts, in descending order
    sort_by_key(labelCountsThrust, labelCountsThrust + numUniqueLabels, uniqueLabelsThrust, thrust::greater<unsigned int>());

    uint8_t *originalPlateIds;

    if (const cudaError_t err = cudaMalloc(&originalPlateIds, sizeof(uint8_t) * MAX_PLATE_COUNT); err != cudaSuccess)
        std::cerr << "Error malloc originalPlateIds: " << cudaGetErrorString(err) << "\n";

    if (const cudaError_t err = cudaMemset(originalPlateIds, MAX_PLATE_COUNT, sizeof(uint8_t)); err != cudaSuccess)
        std::cerr << "Error memset originalPlateIds: " << cudaGetErrorString(err) << "\n";

    unsigned int *unassignedIndices;

    if (const cudaError_t err = cudaMalloc(&unassignedIndices, sizeof(unsigned int) * m_width * m_height / 100); err != cudaSuccess)
        std::cerr << "Error malloc unassignedIndices: " << cudaGetErrorString(err) << "\n";

    int *unassignedIndicesCount;

    if (const cudaError_t err = cudaMalloc(&unassignedIndicesCount, sizeof(int)); err != cudaSuccess)
        std::cerr << "Error malloc unassignedIndicesCount: " << cudaGetErrorString(err) << "\n";

    if (const cudaError_t err = cudaMemset(unassignedIndicesCount, 0, sizeof(int)); err != cudaSuccess)
        std::cerr << "Error memset unassignedIndicesCount: " << cudaGetErrorString(err) << "\n";


    assignNewPlateIds<<<numBlocksPixels, m_threadsPerBlock>>>(labels.deviceTexture(), uniqueLabels, labelCounts, originalPlateIds, m_plateIdsTexture.deviceTexture(), unassignedIndices, unassignedIndicesCount, numUniqueLabels);

    copyToUint8Texture<<<numBlocksPixels, m_threadsPerBlock>>>(labels.deviceTexture(), m_cllPlateIds.deviceTexture(), uniqueLabels, numUniqueLabels);


    cudaFree(labelCounts);
    cudaFree(uniqueLabels);


    int unassignedIndicesCountHost;

    if (const cudaError_t err = cudaMemcpy(&unassignedIndicesCountHost, unassignedIndicesCount, sizeof(int), cudaMemcpyDeviceToHost); err != cudaSuccess)
        std::cerr << "Error memcpy unassignedIndicesCountHost: " << cudaGetErrorString(err) << "\n";
    cudaFree(unassignedIndicesCount);
    if (unassignedIndicesCountHost != 0)
    {
        int hasWork = 1;
        const int gridSize = (unassignedIndicesCountHost + m_threadsPerBlock - 1) / m_threadsPerBlock;

        int *hasRemainingWork;
        if (const cudaError_t err = cudaMalloc(&hasRemainingWork, sizeof(int)); err != cudaSuccess)
            std::cerr << "Error malloc unassignedIndicesCount: " << cudaGetErrorString(err) << "\n";

        while (hasWork)
        {
            if (const cudaError_t err = cudaMemset(hasRemainingWork, 0, sizeof(int)); err != cudaSuccess)
                std::cerr << "Error memset unassignedIndicesCount: " << cudaGetErrorString(err) << "\n";

            assignUnassignedIdsToNeighbor<<<gridSize, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), unassignedIndices, unassignedIndicesCountHost, hasRemainingWork);

            if (const cudaError_t err = cudaMemcpy(&hasWork, hasRemainingWork, sizeof(int), cudaMemcpyDeviceToHost); err != cudaSuccess)
                std::cerr << "Error memcpy hasWork: " << cudaGetErrorString(err) << "\n";
        }
        cudaFree(hasRemainingWork);
    }
    cudaFree(unassignedIndices);


    PlateData *plateDataWrite;

    if (const cudaError_t err = cudaMalloc(&plateDataWrite, sizeof(PlateData) * MAX_PLATE_COUNT); err != cudaSuccess)
        std::cerr << "Error malloc plateDataWrite: " << cudaGetErrorString(err) << "\n";

    copyNewPlateIdLookup<<<numBlocksPlates, m_threadsPerBlock>>>(m_plateDataLookup, originalPlateIds, plateDataWrite);

    if (const cudaError_t err = cudaMemcpy(m_plateDataLookup, plateDataWrite, sizeof(PlateData) * MAX_PLATE_COUNT, cudaMemcpyDeviceToDevice); err != cudaSuccess)
        std::cerr << "Error memcpy plateData: " << cudaGetErrorString(err) << "\n";

    cudaFree(plateDataWrite);
    cudaFree(originalPlateIds);


    if (m_interopManager)
    {
        if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_COLLISION_AREAS)
        {
            CudaTextureHost<uint8_t> glTexture;
            glTexture.initialize(m_width, m_height);
            convertCollisionMapForGL<<<numBlocksPixels, m_threadsPerBlock>>>(
                plateCollisions.deviceTexture(), glTexture.deviceTexture());

            m_interopManager->copyConnection("collisionMap", glTexture.getPointer());
        }
        copyConstantTexturesInterop();
    }

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "Iteration duration: " << elapsed_seconds.count() << std::endl;
}

void PlateTectonicSim::copyConstantTexturesInterop() const
{
    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_CCL_AREAS)
    {
        m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture.getPointer());
    }
    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_CCL_AREAS) // TODO: DEBUG
    {
        m_interopManager->copyConnection("cclTexture", m_cllPlateIds.getPointer());
    }
    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_PLATE_VELOCITIES)
    {
        copyVelocitiesGL();
    }
    if (renderSettings.renderMode == RenderSettings::RenderMode::SHOW_PLATE_DIRECTIONS)
    {
        copyDirectionGL();
    }
    m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture.getPointer());
    m_interopManager->copyConnection("heightMap", m_heightMapTexture.getPointer());
}

void PlateTectonicSim::copyCCL() const // TODO: DEBUGGING ONLY
{
    m_interopManager->copyConnection("cclTexture", m_cllPlateIds.getPointer());
}

Vec2<float> PlateTectonicSim::getPlateCenter(int numBlocksPixels, int m_threadsPerBlock) {
    float4* d_samples;
    cudaMalloc(&d_samples, sizeof(float4) * numBlocksPixels);
    findPlateCenter<<<numBlocksPixels, m_threadsPerBlock>>>(m_iterationStats, m_plateDataLookup, m_plateIdsTexture.deviceTexture(), d_samples);
    float4* h_samples = new float4[numBlocksPixels];
    cudaMemcpy(h_samples, d_samples, sizeof(float4) * numBlocksPixels, cudaMemcpyDeviceToHost);

    float sinX = 0, cosX = 0, sinY = 0, cosY = 0;
    for (int i = 0; i < numBlocksPixels; ++i) {
        sinX += h_samples[i].x;
        cosX += h_samples[i].y;
        sinY += h_samples[i].z;
        cosY += h_samples[i].w;
    }

    float angleX = atan2f(sinX, cosX);
    float angleY = atan2f(sinY, cosY);

    if (angleX < 0) angleX += CURAND_2PI;
    if (angleY < 0) angleY += CURAND_2PI;

    float midX = m_width * angleX / CURAND_2PI;
    float midY = m_height * angleY / CURAND_2PI;

    cudaFree(d_samples);
    delete[] h_samples;

    return Vec2<float>(midX, midY);
}

void PlateTectonicSim::copyPlateIdsGL() const
{
    m_interopManager->copyConnection("cudaPlateTexture", m_plateIdsTexture.getPointer());
}

void PlateTectonicSim::copyDirectionGL() const
{
    CudaTextureHost<float2> glTexture;
    glTexture.initialize(m_width, m_height);
    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    createDirectionTexture<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup,
                                                                   glTexture.deviceTexture());

    m_interopManager->copyConnection("directionTexture", glTexture.getPointer());
}

void PlateTectonicSim::copyVelocitiesGL() const
{
    CudaTextureHost<float> glTexture;
    glTexture.initialize(m_width, m_height);
    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    createVelocityTexture<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup,
                                                                  glTexture.deviceTexture());

    m_interopManager->copyConnection("velocityTexture", glTexture.getPointer());
}

void PlateTectonicSim::resetSim(const unsigned int seed, const int numStartingPlates)
{
    m_seed = seed;
    m_numStartingPlates = numStartingPlates;
    m_heightMapTexture.free();

    // Plate tectonic sim specific device arrays
    m_plateIdsTexture.free();
    m_overlapCrustTexture.free();

    m_cllPlateIds.free(); // TODO: ONLY FOR DEDUG

    if (m_plateDataLookup)
        cudaFree(m_plateDataLookup);

    if (m_randStatesPlates)
        cudaFree(m_randStatesPlates);

    if (m_iterationStats)
        cudaFree(m_iterationStats);

    initializeTectonics();
    if (m_interopManager)
        copyConstantTexturesInterop();
}


void PlateTectonicSim::initializeTectonics()
{
    std::default_random_engine generator(m_seed);
    if (const cudaError_t err = cudaMalloc(&m_randStatesPlates,  m_maxPlates*sizeof(curandState)); err != cudaSuccess)
        std::cerr << "Error Malloc m_randStatePlates:" << cudaGetErrorString(err) << std::endl;

    // Init plate data vector on host, copy to device

    // Init voronoi vector on host, copy to local device, and free at end of function
    const auto voronoiSeedsHost = initializeVoronoiSeeds(generator);

    Vec2<float> *voronoiSeedsDevice;
    if (const cudaError_t err = cudaMalloc(&voronoiSeedsDevice, sizeof(Vec2<float>) * m_numStartingPlates);
        err != cudaSuccess)
        std::cerr << "Error Malloc voronoiSeedsDevice: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(voronoiSeedsDevice, voronoiSeedsHost.data(),
                                           sizeof(Vec2<float>) * m_numStartingPlates,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy voronoiSeedsDevice: " << cudaGetErrorString(err) << std::endl;

    std::vector<PlateData> plateDataHost = initializePlateData(generator, voronoiSeedsHost);
    if (const cudaError_t err = cudaMalloc(&m_plateDataLookup, sizeof(PlateData) * m_maxPlates); err != cudaSuccess)
        std::cerr << "Error Malloc m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaMemcpy(m_plateDataLookup, plateDataHost.data(), sizeof(PlateData) * m_maxPlates,
                                           cudaMemcpyHostToDevice); err != cudaSuccess)
        std::cerr << "Error copy m_plateDataLookup: " << cudaGetErrorString(err) << std::endl;

    m_plateIdsTexture.initialize(m_width, m_height);
    m_heightMapTexture.initialize(m_width, m_height);

    int numBlocksPixels = (m_width * m_height + 1) / m_threadsPerBlock;
    initPlateIDs<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_plateDataLookup, voronoiSeedsDevice,
                                                   static_cast<int>(voronoiSeedsHost.size()));
    initHeightmap << <numBlocksPixels, m_threadsPerBlock >> > (m_heightMapTexture.deviceTexture(), m_seed, 5);

    initPixelDependantPlateData<<<numBlocksPixels, m_threadsPerBlock>>>(m_plateIdsTexture.deviceTexture(), m_heightMapTexture.deviceTexture(), m_plateDataLookup);

    int numBlocksPlates = (m_maxPlates + m_threadsPerBlock - 1) / m_threadsPerBlock;
    initPlatesRngGen<<<numBlocksPlates, m_threadsPerBlock>>>(m_randStatesPlates, m_seed, Vec2<int>(m_maxPlates, 1));

    cudaFree(voronoiSeedsDevice);
    cudaDeviceSynchronize();

    m_cllPlateIds.initialize(m_width, m_height); // TODO: ONLY FOR DEBUG
    
    if (const cudaError_t err = cudaMalloc(&m_iterationStats, sizeof(IterationStatistics)); err != cudaSuccess)
        std::cerr << "Error malloc Simulation Stats: " << cudaGetErrorString(err) << std::endl;

}

std::vector<PlateData> PlateTectonicSim::initializePlateData(std::default_random_engine &generator,
                                                             const std::vector<Vec2<float> > &voronoiSeeds) const
{
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<PlateData> plateData(m_maxPlates);

    for (int i = 0; i != m_numStartingPlates; ++i)
    {
        plateData[i].pixelCenter = {
            voronoiSeeds[i].x - floor(voronoiSeeds[i].x), voronoiSeeds[i].y - floor(voronoiSeeds[i].y)
        };
        plateData[i].velocity = (dist(generator) + 1) / 2;
        const float x_dir = dist(generator);
        const float y_dir = dist(generator);
        const float magnitude = sqrt(x_dir * x_dir + y_dir * y_dir);


        plateData[i].direction = Vec2(x_dir / magnitude, y_dir / magnitude);
        plateData[i].divergenceRandomPlate = dist(generator) > 0 ? 0 : 1;
    }

    return plateData;
}

void PlateTectonicSim::setupToggleCallbacks() const
{
    renderSettings.registerCallback("toggleDefaultMode", [this]
    {
        m_interopManager->toggleSubTextures({"cudaPlateTexture", "heightMap"});
    });

    renderSettings.registerCallback("toggleCollisionMode", [this]
    {
        m_interopManager->toggleSubTextures({"collisionMap"});
    });

    renderSettings.registerCallback("toggleDirectionMode", [this]
    {
        m_interopManager->toggleSubTextures({"directionTexture"});
        copyDirectionGL();
    });

    renderSettings.registerCallback("toggleVelocityMode", [this]
    {
        m_interopManager->toggleSubTextures({"velocityTexture"});
        copyVelocitiesGL();
    });

    renderSettings.registerCallback("toggleUpliftMode", [this]
    {
        m_interopManager->toggleSubTextures({"upliftTexture"});
    });
    renderSettings.registerCallback("toggleCCLMode", [this]// TODO: DEBUGGING ONLY
    {
        m_interopManager->toggleSubTextures({"cclTexture"});
        copyCCL();
    });
}

std::vector<Vec2<float> > PlateTectonicSim::initializeVoronoiSeeds(std::default_random_engine &generator) const
{
    // Init vector to hold the seeds
    std::vector<Vec2<float> > seeds;
    // Init random distribution for height and width
    std::uniform_real_distribution<float> randomHeight(0, static_cast<float>(m_height));
    std::uniform_real_distribution<float> randomWidth(0, static_cast<float>(m_width));

    // Populate vector with random seeds
    for (int i = 0; i != m_numStartingPlates; ++i)
        seeds.emplace_back(randomHeight(generator), randomWidth(generator));

    return seeds;
}
