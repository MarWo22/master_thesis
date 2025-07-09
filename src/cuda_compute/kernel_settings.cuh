#ifndef KERNEL_SETTINGS_CUH
#define KERNEL_SETTINGS_CUH

struct KernelSettings
{
    float inelasticCollisionMultiplier = 10.f;
    float mergeDotDirectionThreshold = .995f;
    float mergeVelocityDiffThreshold = 0.025f;
    int minPlateSize = 10;

    float hydrationPipeCrossSection = 1;
    float hydrationPipeLength = 10;
    float hydrationRainfall = 1.0;
    float hydrationEvaporation = 0.02;
    float gravity = 9.81;
    float sedimentCapacity = 0.3;
    float sedimentDissolving = 0.001;

    bool operator==(const KernelSettings &other) const
    {
        return inelasticCollisionMultiplier == other.inelasticCollisionMultiplier &&
            mergeDotDirectionThreshold == other.mergeDotDirectionThreshold &&
            mergeVelocityDiffThreshold == other.mergeVelocityDiffThreshold &&
            minPlateSize == other.minPlateSize &&
            hydrationPipeCrossSection == other.hydrationPipeCrossSection &&
            hydrationPipeLength == other.hydrationPipeLength &&
            hydrationRainfall == other.hydrationRainfall &&
            hydrationEvaporation == other.hydrationEvaporation &&
            gravity == other.gravity &&
            sedimentCapacity == other.sedimentCapacity &&
            sedimentDissolving == other.sedimentDissolving;
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
