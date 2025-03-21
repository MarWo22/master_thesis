#ifndef PLATE_TECTONIC_SIM_H
#define PLATE_TECTONIC_SIM_H
#include "GL/glew.h"
#include "cuda_gl_interop.h"

class PlateTectonicSim {
    int m_width;
    int m_height;

    // Variables for the CUDA->OpenGL interop pipeline
    cudaGraphicsResource *m_cudaGR;
    cudaArray *m_textureArr;

    // Heightmap on device
    float *m_heightMapDevice;


public:
    PlateTectonicSim(int width, int height);

    void connectToOpengl2DTexture(GLuint texturePointer);
    void copyToOpenGl();


private:
    // void copyToOpenGl();
};



#endif //PLATE_TECTONIC_SIM_H
