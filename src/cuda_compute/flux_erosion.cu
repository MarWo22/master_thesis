#include "flux_erosion.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_save.h"
#include "erosion_kernel.cuh"
#include "cuda_helper.cuh"


FluxVelocityErosion::FluxVelocityErosion(int width, int height)
    : mWidth(width)
    , mHeight(height)
    , completed(0)
    , m_data(nullptr)
{
    m_materialHost.initialize(width, height);
    m_hydrationHost.initialize(width, height);
    m_sedimentHost.initialize(width, height);
    m_sedimentBufferHost.initialize(width, height);
    m_fluxHost.initialize(width, height);
    m_velocityHost.initialize(width, height);
    
    

    CudaErosionData h_data;

    h_data.m_material = m_materialHost.deviceTexture();
    h_data.m_hydration = m_hydrationHost.deviceTexture();
    h_data.m_sediment = m_sedimentHost.deviceTexture();
    h_data.m_sedimentBuffer = m_sedimentBufferHost.deviceTexture();
    h_data.m_flux = m_fluxHost.deviceTexture();
    h_data.m_velocity = m_velocityHost.deviceTexture();

    if (cudaMalloc(&m_data, sizeof(CudaErosionData)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for data object!" << std::endl;
        cudaFree(m_data);
        return;
    }

    cudaMemcpy(m_data, &h_data, sizeof(CudaErosionData), cudaMemcpyHostToDevice);


    initMaterial << <mWidth, mHeight >> > (*m_data, 212312);
}

FluxVelocityErosion::~FluxVelocityErosion() {
    
}

void FluxVelocityErosion::simulate(int iterations) {

    int h_result = 0;
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err;

    for (unsigned int i = 0; i < iterations; i++)
    {
        rainComputation << <mWidth, mHeight >> > (*m_data, 0.02f, 212312);

        fluxComputation << <mWidth, mHeight >> > (*m_data, 0.02f, mGravityConstant, mPipeCrossSectionConstant, mPipeLengthConstant);

        flowComputation << <mWidth, mHeight >> > (*m_data, 0.02f, mPipeLengthConstant);

        sedimentComputation << <mWidth, mHeight >> > (*m_data, 0.02f, mCapacityConstant, mDissolvingConstant, mDispositionConstant);
        
        transportComputation << <mWidth, mHeight >> > (*m_data, 0.02f);
        
        cudaDeviceSynchronize();
        auto l_temp = m_sedimentHost;
        m_sedimentHost = m_sedimentBufferHost;
        m_sedimentBufferHost = l_temp;
      
        evaporateComputation << <mWidth, mHeight >> > (*m_data, 0.02f, mEvaporationConstant);
        
        completed++;
        
        printf("Batch: %4i/%4i Total: %.i \n", i + 1, iterations, completed);
    
        
    }
    cudaThreadSynchronize();
    cudaFree(d_result);

}


