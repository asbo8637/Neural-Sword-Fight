#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/dense>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef unsigned int uint;


////////////////////////////////////////////////////////////
// Bot class
////////////////////////////////////////////////////////////
class Bot
{
public:
    // Allows specifying speed, body length, and sword length at creation
    Bot(float footX, float footY, float sword, float spd, float bodyLen, bool flipped)
        : m_footPos(footX, footY),
          m_speed(spd),
          m_bodyLength(bodyLen),
          m_shoulderOffset(0.4f),
          m_bodyAngle(3.1415f),
          m_armAngle(0.f),
          m_elbowAngle(0.f),
          m_wristAngle(0.f),
          m_armLength(30.f),
          m_forearmLength(30.f),
          m_handLength(sword),
          m_isAlive(true),
          m_flipped(flipped),
          m_momentum(0.f),
          score(0)
    {
        // Circles for joints
        m_jointCircle.setRadius(2.f);
        m_jointCircle.setOrigin(2.f, 2.f);
        m_jointCircle.setFillColor(sf::Color::Red);

        m_head.setRadius(10.f);
        m_head.setOrigin(10.f, 10.f);
        m_head.setFillColor(sf::Color::Blue);
    }

    // New update method: we pass in 5 values (0..1)
    // [0] => foot X, [1] => armAngle, [2] => elbowAngle, [3] => wristAngle, [4] => ???
    void updateFromNN(const std::array<float, 5> &controls)
    {
        if (!m_isAlive)
            return;
        
        sf::Vector2f last_sword_tip = getSwordTip(getWristPos(getElbowPos(getShoulderPos())));
        // Define boundaries for the foot's x position.
        const float minX = 100.f;
        const float maxX = 700.f;

        // Determine foot movement direction.
        float direction = m_flipped ? 1.0f : -1.0f;

        // (1) Move footX
        m_footPos.x += 2.f * direction * (controls[0]) - 0.5f*direction;

        float Pi = 3.14159f;

        // (2) For arm angle:
        m_armAngle = m_flipped ? (-m_armAngle) : m_armAngle;
        m_armAngle = std::max(-2.5f * Pi, std::min(1.5f * Pi, m_armAngle + 0.4f*controls[1] * m_speed));
        m_armAngle = m_flipped ? (-m_armAngle) : m_armAngle;

        // (3) For elbow angle:
        m_elbowAngle = m_flipped ? (-m_elbowAngle) : m_elbowAngle;
        m_elbowAngle = std::max(-2.f * Pi, std::min(2.f * Pi, m_elbowAngle + 0.9f * controls[2] * m_speed));
        m_elbowAngle = m_flipped ? (-m_elbowAngle) : m_elbowAngle;

        // (4) For wrist angle:
        m_wristAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;
        m_wristAngle = std::max(-0.2f * Pi, std::min(0.2f * Pi, m_wristAngle + controls[3] * m_speed));
        m_wristAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;

        // (5) For body angle:
        m_bodyAngle = m_flipped ? (-m_bodyAngle) : m_bodyAngle;
        m_bodyAngle = std::max(0.8f * Pi, std::min(1.2f * Pi, m_bodyAngle + 0.4f*controls[4] * m_speed));
        m_bodyAngle = m_flipped ? (-m_bodyAngle) : m_bodyAngle;

        float last_momentum=m_momentum;
        sf::Vector2f current_sword_tip = getSwordTip(getWristPos(getElbowPos(getShoulderPos())));
        m_momentum += (current_sword_tip.y - last_sword_tip.y);
        if(std::abs(last_momentum)>std::abs(m_momentum)){
            m_momentum=0;
        }
        m_momentum*=0.9f;
    }

    void kill() { m_isAlive = false; }
    bool isAlive() const { return m_isAlive; }
    sf::Vector2f getFootPos() const { return m_footPos; }

    int getScore() const {return score;}
    void incrementScore() {score++;}
    float get_m_momentum() const {return std::abs(m_momentum);}

    // Knockback
    void applyKnockback(float disp)
    {
        if (m_isAlive)
            m_footPos.x -= disp;
            m_momentum=0;
    }

    // For collisions: line foot->head
    void getBodyLine(sf::Vector2f &outStart, sf::Vector2f &outEnd) const
    {
        outStart = m_footPos;
        outEnd = getHeadPos();
    }

    // For collisions: line wrist->swordTip
    void getSwordLine(sf::Vector2f &outStart, sf::Vector2f &outEnd) const
    {
        sf::Vector2f elbowPos = getElbowPos(getShoulderPos());
        sf::Vector2f wristPos = getWristPos(elbowPos);
        sf::Vector2f swordTip = getSwordTip(wristPos);
        outStart = wristPos;
        outEnd = swordTip;
    }

    float getSwordBodyAngle() const
    {
        sf::Vector2f swordStart, swordEnd;
        getSwordLine(swordStart, swordEnd);
        return angleBetween(swordEnd, swordStart);
    }

    std::array<float, 6> getAllyValues() const
    {
        float footX = m_footPos.x;
        float temp_bodyAngle = m_bodyAngle;
        float temp_armAngle = m_armAngle;
        float temp_elbowAngle = m_elbowAngle;
        float temp_wristAngle = m_wristAngle;
        if (m_flipped)
        {
            footX = 800 - footX;
            float pi = 3.14159f;
            temp_bodyAngle = -m_bodyAngle;
            temp_armAngle = -m_armAngle;
            temp_elbowAngle = -m_elbowAngle;
            temp_wristAngle = -m_wristAngle;
        }
        // Normalize footX from [100, 700] to [0, 1].
        float normFootX = (footX - 100.0f) / 600.0f;

        auto normalizeAngle = [](float angle) -> float
        {
            if (angle < 0)
                angle += 2.0f * 3.14159f;
            return angle / (2.0f * 3.14159f);
        };

        float normBodyAngle = normalizeAngle(temp_bodyAngle);
        float normArmAngle = normalizeAngle(temp_armAngle);
        float normElbowAngle = normalizeAngle(temp_elbowAngle);
        float normWristAngle = normalizeAngle(temp_wristAngle);

        return {normFootX, normBodyAngle, normArmAngle, normElbowAngle, normWristAngle, m_momentum};
    }

    void drawSwordRectangle(sf::RenderWindow &window, const sf::Vector2f &wristPos, const sf::Vector2f &swordTip, sf::Color color)
    {
        // Compute the difference vector and its length.
        sf::Vector2f diff = swordTip - wristPos;
        float length = std::sqrt(diff.x * diff.x + diff.y * diff.y);

        // Determine the angle (in degrees) from the horizontal.
        float angle = std::atan2(diff.y, diff.x) * 180.f / 3.14159f;

        // Create a rectangle with the calculated length and a fixed thickness (say, 5 pixels).
        sf::RectangleShape rect(sf::Vector2f(length, 5.0f));
        rect.setFillColor(color);

        // Set the origin to the left-center so that the rectangle starts at wristPos.
        rect.setOrigin(0.f, rect.getSize().y / 2.f);

        // Position and rotate the rectangle.
        rect.setPosition(wristPos);
        rect.setRotation(angle);

        window.draw(rect);
    }

    void drawFace(sf::RenderWindow &window, const sf::Vector2f &faceCenter) {
        // Face circle.
        float faceRadius = 15.f;
        sf::CircleShape face(faceRadius);
        face.setFillColor(sf::Color::Yellow);
        face.setOutlineThickness(2.f);
        face.setOutlineColor(sf::Color::Black);
        // Center the face by setting its origin to its center.
        face.setOrigin(faceRadius, faceRadius);
        face.setPosition(faceCenter);
        window.draw(face);
    
        // Eyes: two small circles.
        float eyeRadius = 2.f;
        sf::CircleShape leftEye(eyeRadius);
        sf::CircleShape rightEye(eyeRadius);
        leftEye.setFillColor(sf::Color::Black);
        rightEye.setFillColor(sf::Color::Black);
        // Position eyes relative to the faceCenter.
        leftEye.setOrigin(eyeRadius, eyeRadius);
        rightEye.setOrigin(eyeRadius, eyeRadius);
        leftEye.setPosition(faceCenter.x - faceRadius/2, faceCenter.y - faceRadius/3);
        rightEye.setPosition(faceCenter.x + faceRadius/2, faceCenter.y - faceRadius/3);
        window.draw(leftEye);
        window.draw(rightEye);
    
        // Mouth: a simple line for a smile.
        sf::VertexArray mouth(sf::Lines, 2);
        mouth[0].position = sf::Vector2f(faceCenter.x - faceRadius/2, faceCenter.y + faceRadius/4);
        mouth[0].color = sf::Color::Black;
        mouth[1].position = sf::Vector2f(faceCenter.x + faceRadius/2, faceCenter.y + faceRadius/4);
        mouth[1].color = sf::Color::Black;
        window.draw(mouth);
    }
    

    void draw(sf::RenderWindow &window)
    {
        if (!m_isAlive)
            return;

        sf::Vector2f headPos = getHeadPos();
        sf::Vector2f shoulderPos = getShoulderPos();
        sf::Vector2f elbowPos = getElbowPos(shoulderPos);
        sf::Vector2f wristPos = getWristPos(elbowPos);
        sf::Vector2f swordTip = getSwordTip(wristPos);

        // Body
        drawLine(window, m_footPos, headPos, sf::Color::Green);

        // Arm
        drawLine(window, shoulderPos, elbowPos, sf::Color::White);
        drawLine(window, elbowPos, wristPos, sf::Color::White);
        drawSwordRectangle(window, wristPos, swordTip, sf::Color::Red);
        

        // Circles
        m_jointCircle.setPosition(m_footPos);
        window.draw(m_jointCircle);

        drawFace(window, headPos);


        m_jointCircle.setPosition(shoulderPos);
        window.draw(m_jointCircle);

        m_jointCircle.setPosition(elbowPos);
        window.draw(m_jointCircle);

        m_jointCircle.setPosition(wristPos);
        window.draw(m_jointCircle);
    }

private:
    sf::Vector2f getHeadPos() const
    {
        float dx = m_bodyLength * std::sin(m_bodyAngle);
        float dy = m_bodyLength * std::cos(m_bodyAngle);
        return sf::Vector2f(m_footPos.x + dx, m_footPos.y + dy);
    }


    // Returns angle in radians between two vectors
    float angleBetween(const sf::Vector2f &v1, const sf::Vector2f &v2) const
    {
        float dot = v1.x * v2.x + v1.y * v2.y;
        float len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
        float len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
        if (len1 < 1e-6f || len2 < 1e-6f)
            return 0.f;
        float cosVal = dot / (len1 * len2);
        cosVal = std::max(-1.f, std::min(1.f, cosVal)); // clamp
        return std::acos(cosVal);
    }

    sf::Vector2f getShoulderPos() const
    {
        sf::Vector2f headPos = getHeadPos();
        sf::Vector2f footDir = m_footPos - headPos;
        float len = std::sqrt(footDir.x * footDir.x + footDir.y * footDir.y);
        if (len < 1e-6f)
            return headPos;
        sf::Vector2f unitDir(footDir.x / len, footDir.y / len);
        return headPos + unitDir * m_shoulderOffset * m_bodyLength;
    }

    sf::Vector2f getElbowPos(const sf::Vector2f &shoulderPos) const
    {
        return sf::Vector2f(
            shoulderPos.x + m_armLength * std::sin(m_armAngle),
            shoulderPos.y + m_armLength * std::cos(m_armAngle));
    }

    sf::Vector2f getWristPos(const sf::Vector2f &elbowPos) const
    {
        float elbowGlobalAngle = m_armAngle + m_elbowAngle;
        return sf::Vector2f(
            elbowPos.x + m_forearmLength * std::sin(elbowGlobalAngle),
            elbowPos.y + m_forearmLength * std::cos(elbowGlobalAngle));
    }

    sf::Vector2f getSwordTip(const sf::Vector2f &wristPos) const
    {
        float wristGlobalAngle = m_armAngle + m_elbowAngle + m_wristAngle;
        return sf::Vector2f(
            wristPos.x + m_handLength * std::sin(wristGlobalAngle),
            wristPos.y + m_handLength * std::cos(wristGlobalAngle));
    }

    void drawLine(sf::RenderWindow &window, const sf::Vector2f &start, const sf::Vector2f &end, sf::Color color)
    {
        sf::VertexArray line(sf::Lines, 2);
        line[0].position = start;
        line[0].color = color;
        line[1].position = end;
        line[1].color = color;
        window.draw(line);
    }

private:
    // Bot properties
    sf::Vector2f m_footPos;
    float m_speed;
    float m_bodyLength;
    float m_shoulderOffset;
    float m_bodyAngle;
    float m_armAngle;
    float m_elbowAngle;
    float m_wristAngle;
    float m_armLength;
    float m_forearmLength;
    float m_handLength;
    int score; 
    float m_momentum;

    bool m_isAlive;
    bool m_flipped;

    // Visuals
    sf::CircleShape m_jointCircle;
    sf::CircleShape m_head;
};