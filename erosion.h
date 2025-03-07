#ifndef EROSION_H
#define EROSION_H
#include <curand_kernel.h>
#include <vector>

struct ErosionGridPoint
{
    struct Flux
    {
        float fL;
        float fR;
        float fT;
        float fB;
    };

    float terrainHeight;
    float waterHeight;
    float sedimentAmount;
    Flux outflowFlux;
    std::pair<float, float> velocityVector;

};

class ErosionGrid
{
    int mWidth;
    int mHeight;
    ErosionGridPoint *mGridDevice;
    curandState *mRandStates;


public:
    ErosionGrid(int width, int height);
    ~ErosionGrid();

    void executeIteration();

    void initRandomGen(unsigned long long seed);

};




#endif //EROSION_H
