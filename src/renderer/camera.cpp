#include "camera.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#include <iostream>

Camera::Camera(const glm::vec3 position,
    const float fov,
    const float baseSpeed,
    const float shiftMultiplier,
    const float rotationSpeed)
        : m_Fov(fov)
        , m_LastFov(fov)
        , m_Pos(position)
        , m_Dir(normalize(glm::vec3(0.0f) - position))
        , m_BaseSpeed(baseSpeed)
        , m_ShiftMultiplier(shiftMultiplier)
        , m_RotationSpeed(rotationSpeed)
        , m_viewMatrix(lookAt(m_Pos, m_Pos + m_Dir, m_WorldUp))
        , m_mouseCoords(0, 0)
        , m_keyStatus()
        , m_AspectRatio(16.f/9.f)
        , m_ProjMatrix(glm::perspective(glm::radians(m_Fov), m_AspectRatio, 5.0f, 2000.0f))
        , m_ZNear(5.f)
        , m_ZFar(2000.f)
{}

void Camera::HandleKeyInput(const int key, const int action)
{
    KeyStatus::Status status;
    if (action == GLFW_PRESS)
        status = KeyStatus::KEY_DOWN;
    else if (action == GLFW_RELEASE)
        status = KeyStatus::KEY_UP;
    else
        return;


    switch(key)
    {
        case GLFW_KEY_LEFT_SHIFT:
            m_keyStatus.shift = status;
            break;
        case GLFW_KEY_W:
            m_keyStatus.w = status;
            break;
        case GLFW_KEY_S:
            m_keyStatus.s = status;
            break;
        case GLFW_KEY_D:
            m_keyStatus.d = status;
            break;
        case GLFW_KEY_A:
            m_keyStatus.a = status;
            break;
        case GLFW_KEY_Q:
            m_keyStatus.q = status;
            break;
        case GLFW_KEY_E:
            m_keyStatus.e = status;
            break;
        default:
            return;
    }
    UpdateViewMatrix();
}

void Camera::Update(const float deltaTime, const float aspectRatio)
{
    const float speed = m_keyStatus.shift ? m_BaseSpeed * m_ShiftMultiplier : m_BaseSpeed;
    bool hasMoved = false;
    if (m_keyStatus.w == KeyStatus::KEY_DOWN)
    {
        hasMoved = true;
        m_Pos += speed * m_Dir * deltaTime;
    }
    if (m_keyStatus.a == KeyStatus::KEY_DOWN)
    {
        hasMoved = true;
        m_Pos -= speed * cross(m_Dir, m_WorldUp) * deltaTime;
    }
    if (m_keyStatus.s == KeyStatus::KEY_DOWN)
    {
        hasMoved = true;
        m_Pos -= speed * m_Dir * deltaTime;
    }
    if (m_keyStatus.d == KeyStatus::KEY_DOWN)
    {
        hasMoved = true;
        m_Pos += speed * cross(m_Dir, m_WorldUp) * deltaTime;
    }
    if (m_keyStatus.q == KeyStatus::KEY_DOWN)
    {
        hasMoved = true;
        m_Pos -= speed * m_WorldUp * deltaTime;
    }
    if (m_keyStatus.e == KeyStatus::KEY_DOWN)
    {
        hasMoved = true;
        m_Pos += speed * m_WorldUp * deltaTime;
    }

    if (hasMoved)
    {
        UpdateViewMatrix();
    }

    if (m_Fov != m_LastFov || m_AspectRatio != aspectRatio)
    {
        m_AspectRatio = aspectRatio;
        UpdateProjectionMatrix();
    }
    m_LastFov = m_Fov;
}

void Camera::HandleMouseInput(const float xPos, const float yPos, const bool allowMovement)
{
    const float delta_x = xPos - m_mouseCoords.x;
    const float delta_y = yPos - m_mouseCoords.y;

    // Update previous mouse coordinates
    m_mouseCoords.x = xPos;
    m_mouseCoords.y = yPos;

    if (!allowMovement)
        return;

    const float pitchAngle = glm::degrees(glm::asin(m_Dir.y)); // Calculate current pitch angle
    constexpr float maxPitch = 89.5f;
    const float newPitch = glm::clamp(pitchAngle - m_RotationSpeed * delta_y, -maxPitch, maxPitch);
    const float deltaPitch = newPitch - pitchAngle;

    // Calculate yaw and pitch rotations
    const glm::mat4 yaw   = rotate(glm::mat4(1.0f), glm::radians(m_RotationSpeed * -delta_x), m_WorldUp);
    const glm::mat4 pitch = rotate(glm::mat4(1.0f), glm::radians(deltaPitch), normalize(cross(m_Dir, m_WorldUp)));

    // Apply rotations to camera direction
    m_Dir = glm::vec3(pitch * yaw * glm::vec4(m_Dir, 0.0f));
    m_Dir = normalize(m_Dir);
    UpdateViewMatrix();
}

void Camera::UpdateViewMatrix()
{
    m_viewMatrix = lookAt(m_Pos, m_Pos + m_Dir, m_WorldUp);
}

void Camera::UpdateProjectionMatrix()
{
    m_ProjMatrix = glm::perspective(glm::radians(m_Fov), m_AspectRatio, 5.0f, 2000.0f);
}