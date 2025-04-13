#ifndef FLUX_EROSION_H
#define FLUX_EROSION_H

#include "cuda_texture.cuh"

struct CudaErosionData {
public:
    CudaTexture<float>* m_material;
    CudaTexture<float>* m_hydration;
    CudaTexture<float>* m_sediment;
    CudaTexture<float>* m_sedimentBuffer;
    CudaTexture<float4>* m_flux;
    CudaTexture<Vec2<float>>* m_velocity;
};

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
    float mDispositionConstant;
    float mEvaporationConstant;
    float mGravityConstant;
    
    CudaTextureHost<float> m_materialHost;
    CudaTextureHost<float> m_hydrationHost;
    CudaTextureHost<float> m_sedimentHost;
    CudaTextureHost<float> m_sedimentBufferHost;
    CudaTextureHost<float4> m_fluxHost;
    CudaTextureHost<Vec2<float>> m_velocityHost;

    CudaErosionData *m_data;

    FluxVelocityErosion(int width, int height);
    ~FluxVelocityErosion();

    void simulate(int iterations);
};

#endif //FLUX_EROSION_H