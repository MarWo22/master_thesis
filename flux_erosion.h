#ifndef FLUX_EROSION_H
#define FLUX_EROSION_H

//float* run_erosion(float* input, int height, int width);
#include <curand_kernel.h>

class FluxVelocityErosion
{
    int mWidth;
    int mHeight;
    curandState* mRandStates;
    float* mMaterial;
    float* mHydration;
    float* mSediment;
    float4* mFlux;
    float2* mVelocity;

public:
    float mPipeLengthConstant;
    float mPipeCrossSectionConstant;
    float mCapacityConstant;
    float mDissolvingConstant;
    float mEvaporationConstant;
    float mGravityConstant;

    FluxVelocityErosion(int width, int height);
    ~FluxVelocityErosion();

    void simulate(float* input, int iterations);
    void getMaterial(float* output);
    void getHydration(float* output);
    void getSediment(float* output);

};

#endif //FLUX_EROSION_H