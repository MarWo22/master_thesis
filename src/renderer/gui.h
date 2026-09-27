#ifndef GUI_H
#define GUI_H
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

class Gui {

    float m_frameCount;
    float m_avgFramerate;

    float m_timeSum;

public:
    Gui();
    ~Gui();

    void render();

private:
    void gui();
    static void preRenderGUI();

    static void renderSettingsSection();
    static void simulationSettingsSection();
    static void saveTextureSection();
    void executionSettingsSection();

    static void resetSimulationSection();
    static unsigned int generateRandomSeed();

    static glm::vec3 getColorFromID(int id);
    static glm::vec3 hsvToRgb(float h, float s, float v);

};




#endif //GUI_H
