#ifndef KERNEL_SETTINGS_CUH
#define KERNEL_SETTINGS_CUH

struct KernelSettings
{
    float inelasticCollisionMultiplierContinental = 1.f;
    float inelasticCollisionMultiplierSubduction = 0.25f;
    float environmentalDragCoefficient = 0.005f;
    float frictionCoefficientContinental = 1.f;
    float frictionCoefficientSubduction = 0.2f;

    float mergeDotDirectionThreshold = .995f;
    float mergeVelocityDiffThreshold = 0.025f;
    int minPlateSize = 10;

    float continentalCrustThreshold = 10000.f;

    float divergence_height_target = 20.f;
    float divergence_interpolation_factor = 0.04f;

    float hydrationPipeCrossSection = 1;
    float hydrationPipeLength = 10;
    float hydrationRainfall = 1.0;
    float hydrationEvaporation = 0.02;
    float gravity = 9.81;
    float sedimentCapacity = 0.3;
    float sedimentDissolving = 0.001;

    bool operator==(const KernelSettings &other) const
    {
        return inelasticCollisionMultiplierContinental == other.inelasticCollisionMultiplierContinental &&
               inelasticCollisionMultiplierSubduction == other.inelasticCollisionMultiplierSubduction &&
               environmentalDragCoefficient == other.environmentalDragCoefficient &&
               mergeDotDirectionThreshold == other.mergeDotDirectionThreshold &&
               mergeVelocityDiffThreshold == other.mergeVelocityDiffThreshold &&
               minPlateSize == other.minPlateSize &&
               hydrationPipeCrossSection == other.hydrationPipeCrossSection &&
               hydrationPipeLength == other.hydrationPipeLength &&
               hydrationRainfall == other.hydrationRainfall &&
               hydrationEvaporation == other.hydrationEvaporation &&
               gravity == other.gravity &&
               sedimentCapacity == other.sedimentCapacity &&
               sedimentDissolving == other.sedimentDissolving &&
               divergence_height_target == other.divergence_height_target &&
               divergence_interpolation_factor == other.divergence_interpolation_factor &&
               continentalCrustThreshold == other.continentalCrustThreshold;
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
