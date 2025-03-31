#include "flux_erosion.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_save.h"
#include "erosion_kernel.cuh"



FluxVelocityErosion::FluxVelocityErosion(int width, int height)
    : mWidth(width)
    , mHeight(height)
    , completed(0)
{
    m_materialDevice.initialize(width, height);
    m_hydrationDevice.initialize(width, height);
    m_sedimentDevice.initialize(width, height);
    m_sedimentBufferDevice.initialize(width, height);
    m_fluxDevice.initialize(width, height);
    m_velocityDevice.initialize(width, height);
    
    initMaterial << <mWidth, mHeight >> > (m_materialDevice.deviceTexture(), 212312);
}

FluxVelocityErosion::~FluxVelocityErosion() {
    
}

void FluxVelocityErosion::simulate(int iterations) {
    for (unsigned int i = 0; i < iterations; i++)
    {
        rainComputation << <mWidth, mHeight >> > (m_hydrationDevice.deviceTexture(), 0.02, 212312);

        fluxComputation << <mWidth, mHeight >> > (m_materialDevice.deviceTexture(), m_hydrationDevice.deviceTexture(), m_fluxDevice.deviceTexture(), 0.02, mGravityConstant, mPipeCrossSectionConstant, mPipeLengthConstant);

        flowComputation << <mWidth, mHeight >> > (m_hydrationDevice.deviceTexture(), m_fluxDevice.deviceTexture(), m_velocityDevice.deviceTexture(), 0.02, mPipeLengthConstant);

        sedimentComputation << <mWidth, mHeight >> > (m_materialDevice.deviceTexture(), m_sedimentDevice.deviceTexture(), m_velocityDevice.deviceTexture(), 0.02, mCapacityConstant, mDissolvingConstant);

        transportComputation << <mWidth, mHeight >> > (m_sedimentDevice.deviceTexture(), m_sedimentBufferDevice.deviceTexture(), m_velocityDevice.deviceTexture(), 0.02);
        cudaMemcpy(m_sedimentDevice.deviceTexture(), m_sedimentBufferDevice.deviceTexture(), sizeof(float) * mWidth * mHeight, cudaMemcpyDeviceToDevice);

        evaporateComputation << <mWidth, mHeight >> > (m_hydrationDevice.deviceTexture(), 0.02, mEvaporationConstant);

        completed++;
        printf("Batch: %4i/%4i Total: %.i \n", i + 1, iterations, completed);
    }
    cudaThreadSynchronize();

}


