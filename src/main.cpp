#include <SFML/Graphics.hpp>
#include <random>
#include <cmath>

// Utility random number generator for angles.
float randomAngleDelta(float minDelta, float maxDelta)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minDelta, maxDelta);
    return dist(gen);
}

class Bot
{
public:
    Bot(float groundX, float groundY)
        : m_groundPos(groundX, groundY),
          m_armAngle(0.f), m_elbowAngle(0.f), m_wristAngle(0.f),
          m_armLength(100.f), m_forearmLength(80.f), m_handLength(50.f)
    {
        // Set up joint circle shapes (just for visual representation).
        m_jointCircle.setRadius(5.f);
        m_jointCircle.setOrigin(5.f, 5.f); // so it is centered on its (x,y)
        m_jointCircle.setFillColor(sf::Color::Red);
    }

    // Randomly adjust the angles a tiny bit each frame.
    void update()
    {
        // For demonstration: jitter each joint angle slightly.
        m_armAngle += randomAngleDelta(-0.99f, 0.99f);
        m_elbowAngle += randomAngleDelta(-0.99f, 0.99f);
        m_wristAngle += randomAngleDelta(-0.3f, 0.3f);

        // If you want to clamp angles, you can do so here, e.g.:
        //   if (m_armAngle < -1.0f) m_armAngle = -1.0f;
        //   etc...
    }

    void draw(sf::RenderWindow &window)
    {
        // 1) Compute joint positions using basic geometry (relative angles).
        //
        //    We'll treat m_groundPos as the base (on the x-axis).
        //    The 'arm' rotates relative to the vertical or horizontal—your choice.
        //    Then the 'forearm' (elbow) rotates relative to the arm,
        //    and the 'hand' (wrist) rotates relative to the forearm.
        //
        //    For simplicity, assume the arm rotates around the base in a
        //    “standard math” orientation (0 angle means pointing right).
        //    You can tweak as needed.

        // Shoulder joint (top joint) – anchored at groundPos, but we offset upward:
        // If you prefer the entire bot standing on x-axis, you might shift groundPos
        // upward by the leg length. For demonstration, just do a “body segment” going up:
        sf::Vector2f shoulderPos = m_groundPos;

        // Arm end (elbow) relative to shoulder:
        //   x = shoulder.x + armLength * cos(armAngle)
        //   y = shoulder.y + armLength * sin(armAngle)
        // You can choose sin/cos usage depending on how you define “zero angle”.
        // Here, let’s define 0 radians = horizontal to the right.
        // If you want vertical alignment, just offset by 90° (M_PI/2).
        sf::Vector2f elbowPos;
        elbowPos.x = shoulderPos.x + m_armLength * std::cos(m_armAngle);
        elbowPos.y = shoulderPos.y + m_armLength * std::sin(m_armAngle);

        // Wrist joint relative to elbow:
        sf::Vector2f wristPos;
        float elbowGlobalAngle = m_armAngle + m_elbowAngle; // “global” elbow angle
        wristPos.x = elbowPos.x + m_forearmLength * std::cos(elbowGlobalAngle);
        wristPos.y = elbowPos.y + m_forearmLength * std::sin(elbowGlobalAngle);

        // “Hand” (or sword tip) if desired. For now just treat it as a short line:
        float wristGlobalAngle = elbowGlobalAngle + m_wristAngle;
        sf::Vector2f handEnd;
        handEnd.x = wristPos.x + m_handLength * std::cos(wristGlobalAngle);
        handEnd.y = wristPos.y + m_handLength * std::sin(wristGlobalAngle);

        // 2) Draw the lines (limbs).
        drawLine(window, shoulderPos, elbowPos, sf::Color::White);
        drawLine(window, elbowPos, wristPos, sf::Color::White);
        drawLine(window, wristPos, handEnd, sf::Color::Yellow); // represent sword

        // 3) Draw the joints as small circles at each position.
        m_jointCircle.setPosition(shoulderPos);
        window.draw(m_jointCircle);

        m_jointCircle.setPosition(elbowPos);
        window.draw(m_jointCircle);

        m_jointCircle.setPosition(wristPos);
        window.draw(m_jointCircle);

        // If you want a small circle at the sword tip:
        m_jointCircle.setPosition(handEnd);
        window.draw(m_jointCircle);
    }

private:
    // A helper to draw a line using SFML’s VertexArray
    void drawLine(sf::RenderWindow &window,
                  const sf::Vector2f &start,
                  const sf::Vector2f &end,
                  sf::Color color)
    {
        sf::VertexArray line(sf::Lines, 2);
        line[0].position = start;
        line[0].color = color;
        line[1].position = end;
        line[1].color = color;
        window.draw(line);
    }

private:
    sf::Vector2f m_groundPos;

    // Angles in radians. Adjust them in update() to animate.
    float m_armAngle;
    float m_elbowAngle;
    float m_wristAngle;

    // Limb lengths
    float m_armLength;
    float m_forearmLength;
    float m_handLength;

    // Visual representation of each joint
    sf::CircleShape m_jointCircle;
};

int main()
{
    // Create a window.
    sf::RenderWindow window(sf::VideoMode(800, 600), "2D Bot Example");
    window.setFramerateLimit(60);

    // Create one bot at some position near the bottom.
    // If you want the “body” to stand on the x-axis, you might place the ground
    // at y = 500 or so, then the arm angles can be adjusted as needed.
    Bot bot(400.f, 300.f);

    // Main loop
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Update bot logic (randomly tweak angles).
        bot.update();

        // Draw everything
        window.clear(sf::Color(50, 50, 50));
        bot.draw(window);
        window.display();
    }

    return 0;
}