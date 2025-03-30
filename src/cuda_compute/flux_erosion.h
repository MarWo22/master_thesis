#ifndef FLUX_EROSION_H
#define FLUX_EROSION_H

//float* run_erosion(float* input, int height, int width);
#include "cuda_texture.cuh"


class FluxVelocityErosion
{
    int mWidth;
    int mHeight;
    int completed;

public:
    float mPipeLengthConstant;
    float mPipeCrossSectionConstant;
    float mCapacityConstant;
    float mDissolvingConstant;
    float mEvaporationConstant;
    float mGravityConstant;
    
    CudaTextureHost<float> m_materialDevice;
    CudaTextureHost<float> m_hydrationDevice;
    CudaTextureHost<float> m_sedimentDevice;
    CudaTextureHost<float> m_sedimentBufferDevice;
    CudaTextureHost<float4> m_fluxDevice;
    CudaTextureHost<Vec2<float>> m_velocityDevice;

    FluxVelocityErosion(float* input, int width, int height);
    ~FluxVelocityErosion();

    void simulate(int iterations);
    
    void getMaterialHost(float* output);
    void getHydrationHost(float* output);
    void getSedimentHost(float* output);
    
    float* getMaterialDevice();
    float* getHydrationDevice();
    float* getSedimentDevice();

};

#endif //FLUX_EROSION_H