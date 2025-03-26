#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <glm/vec2.hpp>

using json = nlohmann::json;


class Config {
public:
    glm::ivec2 HEIGHTMAP_DIMENSIONS;
    bool USE_RENDERER;

    static Config& getInstance() {
        static Config instance;
        return instance;
    }

    void loadConfig(const std::string& filename) {
        if (std::ifstream configFile(filename); configFile.is_open()) {
            json configJson;
            configFile >> configJson;
            // Assign values from the JSON object to config variables
            HEIGHTMAP_DIMENSIONS =  glm::ivec2(configJson["heightmap_dimensions"]["x"], configJson["heightmap_dimensions"]["y"]);
            USE_RENDERER = configJson["use_renderer"];
        } else {
            std::cerr << "Error: Couldn't open config file!" << std::endl;
        }
    }

    // Delete copy/move constructors
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

private:
    // Private constructor to prevent instantiation
    Config() = default;

};



#endif //CONFIG_H
