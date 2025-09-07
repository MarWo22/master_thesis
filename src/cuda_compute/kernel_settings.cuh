#ifndef KERNEL_SETTINGS_CUH
#define KERNEL_SETTINGS_CUH

struct KernelSettings
{
    float inelasticCollisionMultiplier = 10.f;
    float mergeDotDirectionThreshold = .995f;
    float mergeVelocityDiffThreshold = 0.025f;
    int minPlateSize = 10;

    float divergence_height_target = 10.f;
    float divergence_interpolation_factor = 0.05f;

    float upliftMultiplier = 1.f;
    int upliftRange = 50;

    float hydrationPipeCrossSection = 1000;  // 1M m² cross-section for 1km² cell
    float hydrationPipeLength = 10;          // 1km pipe length matching cell size
    float hydrationRainfall = 0.002;           // ~2mm per iteration (realistic for geological time)
    float hydrationEvaporation = 0.001;        // ~1mm per iteration
    float minimumWaterLevel = 10.0f;               // 10m minimum water level (more realistic)
    float gravity = 9.81;
    float sedimentCapacity = 0.01;             // Reduced for 1km scale - less sediment per unit
    float sedimentDissolving = 0.001;          // Reduced dissolution rate for larger scale

    // Pressure system parameters
    float pressureAccumulation = 0.02f;      // Rate of pressure buildup per iteration
    float pressureDecayRate = 0.8f;          // How fast pressure decays at fault lines
    int pressureBlurRange = 25;              // Range for pressure distance field propagation
    float pressureMultiplier = 1.0f;         // Final pressure application multiplier

    // Thermal erosion parameters
    float thermalErosionStrength = 1.0f;    // Strength of thermal erosion smoothing
    float thermalErosionAmplitude = 0.5f;     // Increased for 1km scale - more material movement
    float thermalCellSize = 1000.0f;          // 1000m (1km) cell size
    float thermalThresholdAngle = 0.5f;       // Tangent of the threshold angle for erosion

    bool operator==(const KernelSettings &other) const
    {
        return inelasticCollisionMultiplier == other.inelasticCollisionMultiplier &&
               mergeDotDirectionThreshold == other.mergeDotDirectionThreshold &&
               mergeVelocityDiffThreshold == other.mergeVelocityDiffThreshold &&
               minPlateSize == other.minPlateSize &&
               upliftMultiplier == other.upliftMultiplier &&
               upliftRange == other.upliftRange &&
               hydrationPipeCrossSection == other.hydrationPipeCrossSection &&
               hydrationPipeLength == other.hydrationPipeLength &&
               hydrationRainfall == other.hydrationRainfall &&
               hydrationEvaporation == other.hydrationEvaporation &&
               minimumWaterLevel == other.minimumWaterLevel &&
               gravity == other.gravity &&
               sedimentCapacity == other.sedimentCapacity &&
               sedimentDissolving == other.sedimentDissolving &&
               divergence_height_target == other.divergence_height_target &&
               divergence_interpolation_factor == other.divergence_interpolation_factor &&
               pressureAccumulation == other.pressureAccumulation &&
               pressureDecayRate == other.pressureDecayRate &&
               pressureBlurRange == other.pressureBlurRange &&
               pressureMultiplier == other.pressureMultiplier &&
               thermalErosionStrength == other.thermalErosionStrength &&
               thermalErosionAmplitude == other.thermalErosionAmplitude &&
               thermalCellSize == other.thermalCellSize &&
               thermalThresholdAngle == other.thermalThresholdAngle;
    }

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
