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

    float hydrationPipeCrossSection = 1;
    float hydrationPipeLength = 10;
    float hydrationRainfall = 3.0;
    float hydrationEvaporation = 0.03;
    float gravity = 9.81;
    float sedimentCapacity = 0.3;
    float sedimentDissolving = 0.01;

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
               gravity == other.gravity &&
               sedimentCapacity == other.sedimentCapacity &&
               sedimentDissolving == other.sedimentDissolving &&
               divergence_height_target == other.divergence_height_target &&
               divergence_interpolation_factor == other.divergence_interpolation_factor;
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
