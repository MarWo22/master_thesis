#ifndef KERNEL_SETTINGS_CUH
#define KERNEL_SETTINGS_CUH

struct KernelSettings
{
    float inelasticCollisionMultiplier = 10.f;
    float mergeDotDirectionThreshold = .995f;
    float mergeVelocityDiffThreshold = 0.025f;
    int minPlateSize = 10;

    bool operator==(const KernelSettings &other) const
    {
        return inelasticCollisionMultiplier == other.inelasticCollisionMultiplier &&
               mergeDotDirectionThreshold == other.mergeDotDirectionThreshold &&
               mergeVelocityDiffThreshold == other.mergeVelocityDiffThreshold &&
               minPlateSize == other.minPlateSize;
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
