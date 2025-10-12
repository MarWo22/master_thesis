#ifndef KERNEL_SETTINGS_CUH
#define KERNEL_SETTINGS_CUH

struct KernelSettings
{
    float inelasticCollisionMultiplierContinental = 1.f;
    float inelasticCollisionMultiplierSubduction = 0.25f;
    float inelasticCollisionMultiplierAccretion = 0.5f;

    float frictionCoefficientContinental = 1.f;
    float frictionCoefficientSubduction = 0.2f;
    float frictionCoefficientAccretion = 0.5f;

    float environmentalDragCoefficient = 0.25f;


    float mergeDotDirectionThreshold = .995f;
    float mergeVelocityDiffThreshold = 0.025f;

    float mergeMinVelocity = 0.05f;

    int minPlateSize = 10;

    float continentalCrustThreshold = 10000.f;

    float divergence_height_target_min = 1000.f;
    float divergence_height_target_max = 2000.f;

    float divergence_interpolation_factor = 0.025f;
    int collisionTypeUpdateCooldown = 4;
    int minCollisionSize = 50;

    float upliftMultiplier = 1.f;
    int upliftRange = 50;

    float hydrationPipeCrossSection = 1000; // 1M m² cross-section for 1km² cell
    float hydrationPipeLength = 10; // 1km pipe length matching cell size
    float hydrationRainfall = 0.002; // ~2mm per iteration (realistic for geological time)
    float hydrationEvaporation = 0.001; // ~1mm per iteration
    float minimumWaterLevel = 10.0f; // 10m minimum water level (more realistic)
    float gravity = 9.81;
    float sedimentCapacity = 0.01; // Reduced for 1km scale - less sediment per unit
    float sedimentDissolving = 0.001; // Reduced dissolution rate for larger scale

    float pressureAccumulation = 0.02f;      // Rate of pressure buildup per iteration
    float pressureDecayRate = 0.8f;          // How fast pressure decays at fault lines
    int pressureBlurRange = 25;              // Range for pressure distance field propagation
    float pressureMultiplier = 1.0f;         // Final pressure application multiplier
    float stressSplitThreshold = 30.0f;      // Stress threshold for plate splitting
    int targetMinimumPlateArea = 100;
    int targetMaximumPlateArea = 100000;

    // Thermal erosion parameters
    float thermalErosionStrength = 1.0f; // Strength of thermal erosion smoothing
    float thermalErosionAmplitude = 0.5f; // Increased for 1km scale - more material movement
    float thermalCellSize = 1000.0f; // 1000m (1km) cell size
    float thermalThresholdAngle = 0.5f; // Tangent of the threshold angle for erosion

    // Precomputed Gaussian weights will be defined as constants

    bool operator==(const KernelSettings &other) const = default;

    bool operator!=(const KernelSettings &other) const
    {
        return !(*this == other);
    }
};

void copyHostKernelSettingsToDevice();


#ifdef __CUDACC__

extern __constant__ KernelSettings kernelSettings;
#endif

extern KernelSettings kernelSettingsHost;

#endif // KERNEL_SETTINGS_CUH
