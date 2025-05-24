#include <cstdint>
#ifndef ITERATION_STATISTICS_H
#define ITERATION_STATISTICS_H

struct IterationStatistics
{
    int largestValue;
    uint8_t largestPlateId;

    float heaviestValue;
    uint8_t heaviestPlateId;

    int numberOfPlates;


    IterationStatistics()
        : largestValue(0)
        , largestPlateId(0)
        , heaviestValue(0)
        , heaviestPlateId(0)
        , numberOfPlates(0)
    {}
};

#endif //ITERATION_STATISTICS_H