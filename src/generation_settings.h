#ifndef GENERATION_SETTINGS_H
#define GENERATION_SETTINGS_H

#include <functional>
#include <unordered_map>
#include <string>



class GenerationSettings {
      std::unordered_map<std::string, std::vector<std::function<void()>>> m_callbacks;
public:
      enum RenderMode
      {
            NORMAL,
            SHOW_PLATES
      };

      // The actual settings. Could make them private with getter/setters, but I think this is sufficient
      int executionIterations = 1;
      int renderMode = NORMAL;


      void registerCallback(const std::string &ident, const std::function<void()> &);
      void callCallback(const std::string &ident);

};



#endif //GENERATION_SETTINGS_H
