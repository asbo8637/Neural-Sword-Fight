#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>

////////////////////////////////////////////////////////////
// Quick geometry & collision helpers
////////////////////////////////////////////////////////////
sf::Vector2f normalize(const sf::Vector2f &v)
{
    float len = std::sqrt(v.x * v.x + v.y * v.y);
    if (len < 1e-6f)
        return sf::Vector2f(0.f, 0.f);
    return sf::Vector2f(v.x / len, v.y / len);
}

// Returns angle in radians between two vectors
float angleBetween(const sf::Vector2f &v1, const sf::Vector2f &v2)
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

// Standard line-segment intersection
bool linesIntersect(const sf::Vector2f &p1, const sf::Vector2f &p2,
                    const sf::Vector2f &p3, const sf::Vector2f &p4)
{
    float denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if (std::fabs(denom) < 1e-9)
    {
        // Lines are (nearly) parallel
        return false;
    }

    float t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    float u = ((p1.x - p3.x) * (p1.y - p2.y) - (p1.y - p3.y) * (p1.x - p2.x)) / denom;

    return (t >= 0.f && t <= 1.f && u >= 0.f && u <= 1.f);
}

////////////////////////////////////////////////////////////
// Bot class
////////////////////////////////////////////////////////////
class Bot
{
public:
    // Constructor
    Bot(float footX, float footY, bool flipped = false)
        : m_footPos(footX, footY),
          m_speed(0.3f),
          m_bodyLength(120.f),
          m_shoulderOffset(40.f),
          m_bodyAngle(0.f),
          m_armAngle(0.f),
          m_elbowAngle(0.f),
          m_wristAngle(0.f),
          m_armLength(60.f),
          m_forearmLength(70.f),
          m_handLength(160.f),
          m_isAlive(true),
          m_flipped(flipped)
    {
        // Circles for joints
        m_jointCircle.setRadius(5.f);
        m_jointCircle.setOrigin(5.f, 5.f);
        m_jointCircle.setFillColor(sf::Color::Red);

        m_head.setRadius(10.f);
        m_head.setOrigin(10.f, 10.f);
        m_head.setFillColor(sf::Color::Blue);

        // If flipped, rotate the entire arm 180 degrees
        if (m_flipped)
            m_armAngle = 3.14159f; // ~180 deg
    }

    // New update method: we pass in 5 values (0..1)
    // [0] => foot X, [1] => bodyAngle, [2] => armAngle, [3] => elbowAngle, [4] => wristAngle
    void updateFromNN(const std::array<float, 5> &controls)
    {
        if (!m_isAlive)
            return;

        // 1) Map foot X from [0..1] to some desired range (e.g. [100..700])
        float minX = 100.f;
        float maxX = 700.f;
        float isflipped = 0;

        if(m_flipped)
            isflipped = -m_speed;
        else
            isflipped = m_speed;
        // We can keep footPos.y fixed or also map it if desired:
        // m_footPos.y = <some constant> or from controls as well
        m_footPos.x += controls[0] * 5*isflipped;
        // 2) Map bodyAngle from [0..1] to e.g. [-π/2..+π/2]
        float halfPi = 3.14159f * 0.5f;

        m_bodyAngle = std::max(-halfPi, std::min(halfPi, m_bodyAngle+controls[1]*isflipped)); // clamp

        m_armAngle += controls[2] * isflipped;

        // 4) elbowAngle => similarly
        m_elbowAngle += controls[3] * isflipped;

        // 5) wristAngle => similarly
        m_wristAngle += controls[4] * isflipped;
    }

    void kill() { m_isAlive = false; }
    bool isAlive() const { return m_isAlive; }
    sf::Vector2f getFootPos() const { return m_footPos; }

    // Knockback
    void applyKnockback(const sf::Vector2f &disp)
    {
        if (m_isAlive)
            m_footPos += disp;
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
        sf::Vector2f bodyVec = getHeadPos() - m_footPos;
        sf::Vector2f swordStart, swordEnd;
        getSwordLine(swordStart, swordEnd);
        sf::Vector2f swordVec = swordEnd - swordStart;
        return angleBetween(bodyVec, swordVec);
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
        drawLine(window, wristPos, swordTip, sf::Color::Yellow);

        // Circles
        m_jointCircle.setPosition(m_footPos);
        window.draw(m_jointCircle);

        m_head.setPosition(headPos);
        window.draw(m_head);

        m_jointCircle.setPosition(shoulderPos);
        window.draw(m_jointCircle);

        m_jointCircle.setPosition(elbowPos);
        window.draw(m_jointCircle);

        m_jointCircle.setPosition(wristPos);
        window.draw(m_jointCircle);
    }

private:
    // Compute geometry with body angle
    sf::Vector2f getHeadPos() const
    {
        float dx = m_bodyLength * std::sin(m_bodyAngle);
        float dy = -m_bodyLength * std::cos(m_bodyAngle);
        return sf::Vector2f(m_footPos.x + dx, m_footPos.y + dy);
    }

    sf::Vector2f getShoulderPos() const
    {
        sf::Vector2f headPos = getHeadPos();
        sf::Vector2f footDir = m_footPos - headPos;
        float len = std::sqrt(footDir.x * footDir.x + footDir.y * footDir.y);
        if (len < 1e-6f)
            return headPos;
        sf::Vector2f unitDir = sf::Vector2f(footDir.x / len, footDir.y / len);
        return headPos + unitDir * m_shoulderOffset;
    }

    sf::Vector2f getElbowPos(const sf::Vector2f &shoulderPos) const
    {
        return sf::Vector2f(
            shoulderPos.x + m_armLength * std::cos(m_armAngle),
            shoulderPos.y + m_armLength * std::sin(m_armAngle));
    }

    sf::Vector2f getWristPos(const sf::Vector2f &elbowPos) const
    {
        float elbowGlobalAngle = m_armAngle + m_elbowAngle;
        return sf::Vector2f(
            elbowPos.x + m_forearmLength * std::cos(elbowGlobalAngle),
            elbowPos.y + m_forearmLength * std::sin(elbowGlobalAngle));
    }

    sf::Vector2f getSwordTip(const sf::Vector2f &wristPos) const
    {
        float wristGlobalAngle = m_armAngle + m_elbowAngle + m_wristAngle;
        return sf::Vector2f(
            wristPos.x + m_handLength * std::cos(wristGlobalAngle),
            wristPos.y + m_handLength * std::sin(wristGlobalAngle));
    }

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
    // Bot properties
    sf::Vector2f m_footPos; // foot location
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

    bool m_isAlive;
    bool m_flipped;

    // Visuals
    sf::CircleShape m_jointCircle;
    sf::CircleShape m_head;
};

////////////////////////////////////////////////////////////
// Collision logic
////////////////////////////////////////////////////////////
void checkSwordSwordCollision(Bot &A, Bot &B)
{
    if (!A.isAlive() || !B.isAlive())
        return;

    sf::Vector2f aSwordStart, aSwordEnd;
    A.getSwordLine(aSwordStart, aSwordEnd);
    sf::Vector2f bSwordStart, bSwordEnd;
    B.getSwordLine(bSwordStart, bSwordEnd);

    if (linesIntersect(aSwordStart, aSwordEnd, bSwordStart, bSwordEnd))
    {
        float angleA = A.getSwordBodyAngle();
        float angleB = B.getSwordBodyAngle();

        float forceA = std::fabs(std::sin(angleA));
        float forceB = std::fabs(std::sin(angleB));

        float knockbackScale = 20.f;
        sf::Vector2f dirAB = B.getFootPos() - A.getFootPos();
        sf::Vector2f normAB = normalize(dirAB);
        sf::Vector2f normBA = -normAB;

        B.applyKnockback(normAB * (forceA * knockbackScale));
        A.applyKnockback(normBA * (forceB * knockbackScale));
    }
}

void checkSwordHitsBody(Bot &attacker, Bot &victim)
{
    if (!attacker.isAlive() || !victim.isAlive())
        return;

    sf::Vector2f swordStart, swordEnd;
    attacker.getSwordLine(swordStart, swordEnd);

    sf::Vector2f bodyStart, bodyEnd;
    victim.getBodyLine(bodyStart, bodyEnd);

    if (linesIntersect(swordStart, swordEnd, bodyStart, bodyEnd))
    {
        victim.kill();
    }
}

void handleCollisions(Bot &A, Bot &B)
{
    checkSwordSwordCollision(A, B);
    checkSwordHitsBody(A, B);
    checkSwordHitsBody(B, A);
}

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////
int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "NN Control Example");
    window.setFramerateLimit(60);

    // Create two bots
    Bot botA(150.f, 400.f, false);
    Bot botB(650.f, 400.f, true);

    // For demonstration, a simple array of 5 control values [0..1]
    // that we will randomize each frame to show them moving.
    // In a real neural-net scenario, you'd get these from your forward pass.
    std::array<float, 5> controlsA = {{0.f, 0.f, 0.f, 0.f, 0.f}};
    std::array<float, 5> controlsB = {{1.f, 1.f, 1.f, 1.f, 1.f}};

    // Example "walls"
    float leftWallX = 50.f;
    float rightWallX = 750.f;

    while (window.isOpen())
    {
        // Poll events
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Randomly set the 5 controls for each bot
        // In your real code, you'd get these from a neural net forward pass:

        // Update each bot with its 5 new control values
        if (botA.isAlive())
            botA.updateFromNN(controlsA);
        if (botB.isAlive())
            botB.updateFromNN(controlsB);

        // Check collisions
        handleCollisions(botA, botB);

        // Kill if they cross walls
        if (botA.isAlive())
        {
            float xA = botA.getFootPos().x;
            if (xA < leftWallX || xA > rightWallX)
                botA.kill();
        }
        if (botB.isAlive())
        {
            float xB = botB.getFootPos().x;
            if (xB < leftWallX || xB > rightWallX)
                botB.kill();
        }

        // Draw
        window.clear(sf::Color(50, 50, 50));
        botA.draw(window);
        botB.draw(window);

        // Optional: draw the walls
        sf::VertexArray lw(sf::Lines, 2);
        lw[0].position = sf::Vector2f(leftWallX, 0.f);
        lw[0].color = sf::Color::Magenta;
        lw[1].position = sf::Vector2f(leftWallX, 600.f);
        lw[1].color = sf::Color::Magenta;
        window.draw(lw);

        sf::VertexArray rw(sf::Lines, 2);
        rw[0].position = sf::Vector2f(rightWallX, 0.f);
        rw[0].color = sf::Color::Magenta;
        rw[1].position = sf::Vector2f(rightWallX, 600.f);
        rw[1].color = sf::Color::Magenta;
        window.draw(rw);

        window.display();
    }

    return 0;
}