#ifndef MAIN_CAMERA_H
#define MAIN_CAMERA_H

#include<glm/glm.hpp>

class Camera
{
    struct KeyStatus
    {
        enum Status{KEY_UP, KEY_DOWN};

        Status w;
        Status a;
        Status s;
        Status d;
        Status q;
        Status e;
        Status shift;
    };

    float m_Fov;
    float m_LastFov;
    glm::vec3 m_Pos;
    glm::vec3 m_Dir;
    float m_BaseSpeed;
    float m_ShiftMultiplier;
    float m_RotationSpeed;
    glm::mat4 m_viewMatrix;

    glm::vec2 m_mouseCoords;
    KeyStatus m_keyStatus;

    float m_AspectRatio;
    glm::mat4 m_ProjMatrix;

    float m_ZNear;
    float m_ZFar;

    constexpr static glm::vec3 m_WorldUp = {0.0f, 1.0f, 0.0f};

public:
    Camera() = default;
    Camera(glm::vec3 position, float fov, float baseSpeed, float shiftMultiplier, float rotationSpeed);
    [[nodiscard]] float Fov() const { return m_Fov; }
    [[nodiscard]] const glm::vec3 &Pos() const { return m_Pos; }
    [[nodiscard]] const glm::vec3 &Dir() const { return m_Dir; }
    [[nodiscard]] float BaseSpeed() const { return m_BaseSpeed; }
    [[nodiscard]] float ShiftMultiplier() const {return m_ShiftMultiplier; }
    [[nodiscard]] const glm::mat4 &ViewMatrix() const {return m_viewMatrix; }
    [[nodiscard]] const glm::mat4 &ProjectionMatrix() const { return m_ProjMatrix; }
    [[nodiscard]] float AspectRatio() const { return m_AspectRatio; }
    [[nodiscard]] float ZNear() const { return m_ZNear; }
    [[nodiscard]] float ZFar() const { return m_ZFar; }


    void Fov(const float fov) { m_Fov = fov; }
    void BaseSpeed(const float baseSpeed) { m_BaseSpeed = baseSpeed; }
    void ShiftMultiplier(const float shiftMultiplier) { m_ShiftMultiplier = shiftMultiplier; }

    void HandleKeyInput(int key, int action);
    void Update(float deltaTime, float aspectRatio);
    void HandleMouseInput(float xPos, float yPos, bool allowMovement);

private:
    void UpdateViewMatrix();

    void UpdateProjectionMatrix();

};


#endif //MAIN_CAMERA_H