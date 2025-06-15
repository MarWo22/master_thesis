#include "kernel_settings.cuh"

__constant__ KernelSettings kernelSettings;

KernelSettings kernelSettingsHost = {};

__host__ void copyHostKernelSettingsToDevice()
{
    cudaMemcpyToSymbol(kernelSettings, &kernelSettingsHost, sizeof(KernelSettings));
}