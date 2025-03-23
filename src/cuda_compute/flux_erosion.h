#ifndef FLUX_EROSION_H
#define FLUX_EROSION_H

//float* run_erosion(float* input, int height, int width);
#include <curand_kernel.h>

class FluxVelocityErosion
{
    int mWidth;
    int mHeight;
    int completed;
    void simulate(int iterations);

public:
    float mPipeLengthConstant;
    float mPipeCrossSectionConstant;
    float mCapacityConstant;
    float mDissolvingConstant;
    float mEvaporationConstant;
    float mGravityConstant;
    
    curandState* mRandStatesDevice;
    float* mMaterialDevice;
    float* mHydrationDevice;
    float* mSedimentDevice;
    float4* mFluxDevice;
    float2* mVelocityDevice;

    FluxVelocityErosion(int width, int height);
    ~FluxVelocityErosion();

    void start(float* input, int iterations, bool fromDevice);
    void resume(int iterations);
    void getMaterialHost(float* output);
    void getHydrationHost(float* output);
    void getSedimentHost(float* output);
    float* getMaterialDevice();
    float* getHydrationDevice();
    float* getSedimentDevice();

};

#endif //FLUX_EROSION_H