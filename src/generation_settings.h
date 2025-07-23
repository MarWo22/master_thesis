#ifndef GENERATION_SETTINGS_H
#define GENERATION_SETTINGS_H

#include <functional>
#include <unordered_map>
#include <string>



class  RenderSettings {
      std::unordered_map<std::string, std::vector<std::function<void()>>> m_callbacks;
public:
      enum RenderMode
      {
            NORMAL,
            SHOW_COLLISION_AREAS,
            SHOW_PLATE_DIRECTIONS,
            SHOW_PLATE_VELOCITIES,
            SHOW_UPLIFT_AREAS,
      };

      bool renderHeight = true;
      bool renderBorders = false;
      bool renderWater = true;
      bool renderDirections = false;
      float heightMultiplier = 0.2;

      // The actual settings. Could make them private with getter/setters, but I think this is sufficient
      int executionIterations = 1;
      int iterationsPerSecond = 5;
      bool isExecutingRealtime = false;
      int renderMode = NORMAL;

      void registerCallback(const std::string &ident, const std::function<void()> &);
      void callCallback(const std::string &ident);

};

struct SimulationSettings
{
      int numStartingPlates = 16;
      unsigned int seed = 1000;
      std::vector<int> numVoronoiSeeds = {200, 500};


      std::function<void(unsigned int, int, const std::vector<int> &)> resetCallback;
};

struct SaveTextureGui
{
      enum TextureType
      {
            HEIGHTMAP,
            PLATE_IDS,
      };

      int texture = HEIGHTMAP;
      std::string path;

      std::function<void()> saveTextureCallback;
};


#endif //GENERATION_SETTINGS_H
