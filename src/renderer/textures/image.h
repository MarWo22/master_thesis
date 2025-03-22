#ifndef IMAGE_H_
#define IMAGE_H_


struct Image {

    int width = 0;
    int height = 0;
    int components = 0;
    float *data = nullptr;
    // Constructor
    explicit Image(const std::string& filename, int channels);
    // Destructor
    ~Image();

};


#endif //IMAGE_H_
