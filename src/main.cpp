#include <SFML/Graphics.hpp>
#include <random>
#include <cmath>

// Utility random number generator for small “jitter” (or angle deltas).
float randomDelta(float minDelta, float maxDelta)
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
        : m_footPos(groundX, groundY), // "foot" on the ground
          m_bodyLength(120.f),         // total height from foot to "head"
          m_shoulderOffset(40.f),      // how far below the top of the body the shoulder is
          m_armAngle(0.f), m_elbowAngle(0.f), m_wristAngle(0.f),
          m_armLength(100.f), m_forearmLength(80.f), m_handLength(50.f)
    {
        // Set up joint circle shapes (just for visual representation).
        m_jointCircle.setRadius(5.f);
        m_jointCircle.setOrigin(5.f, 5.f); // center the circle on its coordinate
        m_jointCircle.setFillColor(sf::Color::Red);

        m_head.setRadius(10.f);
        m_head.setOrigin(10.f, 10.f); // center the circle on its coordinate
        m_head.setFillColor(sf::Color::Blue);
    }

    void update()
    {
        // Randomly “jitter” the foot horizontally
        m_footPos.x += randomDelta(-8.f, 8.f);
        //m_footPos.y = 400.f + 100.f * std::sin(m_footPos.x / 50.f);
        // Jitter each joint angle slightly
        m_armAngle += randomDelta(-0.01f, 0.01f);
        m_elbowAngle += randomDelta(-0.01f, 0.01f);
        m_wristAngle += randomDelta(-0.01f, 0.01f);
    }

    void draw(sf::RenderWindow &window)
    {
        // ----------------------------------------------------------
        // 1) Compute the vertical body: Foot -> Head
        // ----------------------------------------------------------
        // The top of the body ("head") is bodyLength above the foot
        sf::Vector2f headPos = m_footPos - sf::Vector2f(0.f, m_bodyLength);

        // Draw the full body as a green line from foot to head
        drawLine(window, m_footPos, headPos, sf::Color::Green);

        // ----------------------------------------------------------
        // 2) Compute the shoulder, slightly below the top (“head”)
        // ----------------------------------------------------------
        // The shoulder is “shoulderOffset” below the head
        sf::Vector2f shoulderPos = headPos + sf::Vector2f(0.f, m_shoulderOffset);

        // ----------------------------------------------------------
        // 3) Compute & draw the arm from the shoulder
        // ----------------------------------------------------------
        sf::Vector2f elbowPos;
        elbowPos.x = shoulderPos.x + m_armLength * std::cos(m_armAngle);
        elbowPos.y = shoulderPos.y + m_armLength * std::sin(m_armAngle);

        float elbowGlobalAngle = m_armAngle + m_elbowAngle;
        sf::Vector2f wristPos;
        wristPos.x = elbowPos.x + m_forearmLength * std::cos(elbowGlobalAngle);
        wristPos.y = elbowPos.y + m_forearmLength * std::sin(elbowGlobalAngle);

        float wristGlobalAngle = elbowGlobalAngle + m_wristAngle;
        sf::Vector2f handEnd;
        handEnd.x = wristPos.x + m_handLength * std::cos(wristGlobalAngle);
        handEnd.y = wristPos.y + m_handLength * std::sin(wristGlobalAngle);

        // Lines for the arm & “sword”
        drawLine(window, shoulderPos, elbowPos, sf::Color::White);
        drawLine(window, elbowPos, wristPos, sf::Color::White);
        drawLine(window, wristPos, handEnd, sf::Color::Yellow);

        // ----------------------------------------------------------
        // 4) Draw the circles at important joints
        // ----------------------------------------------------------
        // Foot
        m_jointCircle.setPosition(m_footPos);
        window.draw(m_jointCircle);

        // Head
        m_head.setPosition(headPos);
        window.draw(m_head);

        // Shoulder
        m_jointCircle.setPosition(shoulderPos);
        window.draw(m_jointCircle);

        // Elbow
        m_jointCircle.setPosition(elbowPos);
        window.draw(m_jointCircle);

        // Wrist
        m_jointCircle.setPosition(wristPos);
        window.draw(m_jointCircle);

        // Optional: tip of the “sword”
        // m_jointCircle.setPosition(handEnd);
        // window.draw(m_jointCircle);
    }

private:
    // Helper to draw a single line segment in SFML
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
    // “Foot” position on the ground (jittered each frame)
    sf::Vector2f m_footPos;

    // Total vertical height from foot to head
    float m_bodyLength;
    // Shoulder is some pixels below the very top (“head”)
    float m_shoulderOffset;

    // Arm joint angles
    float m_armAngle;
    float m_elbowAngle;
    float m_wristAngle;

    // Limb lengths
    float m_armLength;
    float m_forearmLength;
    float m_handLength;

    // Circles for drawing joints
    sf::CircleShape m_jointCircle;
    sf::CircleShape m_head;
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "2D Bot with Body & Offset Shoulder");
    window.setFramerateLimit(60);

    // Start the bot around the middle of the screen
    Bot bot(400.f, 400.f);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Update bot logic
        bot.update();

        // Draw everything
        window.clear(sf::Color(50, 50, 50));
        bot.draw(window);
        window.display();
    }

    return 0;
}