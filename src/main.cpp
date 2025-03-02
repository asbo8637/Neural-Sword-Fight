#include <SFML/Graphics.hpp>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/dense>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef unsigned int uint;

////////////////////////////////////////////////////////////
// Utility random number generator
////////////////////////////////////////////////////////////
float randomDelta(float minDelta, float maxDelta)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minDelta, maxDelta);
    return dist(gen);
}

////////////////////////////////////////////////////////////
// Quick geometry helper: Vector angle, line intersection, etc.
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
    // dot(a,b) = |a||b| cos(theta)
    // => theta = arccos( dot(a,b) / (|a||b|) )
    float dot = v1.x * v2.x + v1.y * v2.y;
    float len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    if (len1 < 1e-6f || len2 < 1e-6f)
        return 0.f;
    float cosVal = dot / (len1 * len2);
    cosVal = std::max(-1.f, std::min(1.f, cosVal)); // clamp
    return std::acos(cosVal);
}

// Check line segment intersection using standard parametric approach
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

    // Intersection if t and u are both within [0,1]
    return (t >= 0.f && t <= 1.f && u >= 0.f && u <= 1.f);
}

////////////////////////////////////////////////////////////
// Bot class
////////////////////////////////////////////////////////////
class Bot
{
public:
    // If flipped=true => the arm initially faces left instead of right
    Bot(float footX, float footY, bool flipped = false)
        : m_footPos(footX, footY),
          m_bodyLength(120.f),
          m_shoulderOffset(40.f),
          m_bodyAngle(0.f), // new angle for "poking" or leaning
          m_armAngle(0.f),
          m_elbowAngle(0.f),
          m_wristAngle(0.f),
          m_armLength(60.f),
          m_forearmLength(70.f),
          m_handLength(100.f),
          m_isAlive(true),
          m_flipped(flipped)
    {
        // Circle for joints
        m_jointCircle.setRadius(5.f);
        m_jointCircle.setOrigin(5.f, 5.f);
        m_jointCircle.setFillColor(sf::Color::Red);

        // "Head" circle
        m_head.setRadius(10.f);
        m_head.setOrigin(10.f, 10.f);
        m_head.setFillColor(sf::Color::Blue);

        // If flipped, rotate the entire arm 180 degrees
        if (m_flipped)
        {
            m_armAngle = 3.14159f; // ~180 degrees
        }
    }

    // Update with random foot jitter, body tilt, arm angles, etc.
    void update(const sf::Vector2f &opponentFootPos)
    {
        if (!m_isAlive)
            return;

        // Random foot movement
        if (!m_flipped)
            m_footPos.x += randomDelta(0.f, 2.f);
        else
            m_footPos.x += randomDelta(-2.f, 0.f);

        // --- 1) Body "poking" angle ---
        // Let bodyAngle vary a little to simulate leaning in/out
        m_bodyAngle += randomDelta(-0.02f, 0.02f);
        // If you want to clamp angle so they don't fold in half, e.g. [-1, +1] rad:
        // m_bodyAngle = std::max(-1.f, std::min(m_bodyAngle, 1.f));

        // --- 2) Arm angles ---
        m_armAngle += randomDelta(-0.07f, 0.07f);
        m_elbowAngle += randomDelta(-0.04f, 0.04f);

        // Aiming logic for the wrist
        float wAngleToOpponent = computeWristAimAngle(opponentFootPos);
        float currentWristAngle = (m_armAngle + m_elbowAngle + m_wristAngle);
        float diff = wAngleToOpponent - currentWristAngle;
        // wrap diff into [-π, π]
        if (diff > 3.14159f)
            diff -= 6.28318f;
        if (diff < -3.14159f)
            diff += 6.28318f;
        m_wristAngle += 0.1f * diff + randomDelta(-0.01f, 0.01f);
    }

    void kill() { m_isAlive = false; }
    bool isAlive() const { return m_isAlive; }

    // Move the entire bot's foot (knockback)
    void applyKnockback(const sf::Vector2f &displacement)
    {
        if (!m_isAlive)
            return;
        m_footPos += displacement;
    }

    // For collisions, we still treat the body line as foot->head
    void getBodyLine(sf::Vector2f &outStart, sf::Vector2f &outEnd) const
    {
        sf::Vector2f headPos = getHeadPos();
        outStart = m_footPos;
        outEnd = headPos;
    }

    // The sword line is wrist->swordTip
    void getSwordLine(sf::Vector2f &outStart, sf::Vector2f &outEnd) const
    {
        sf::Vector2f elbowPos = getElbowPos(getShoulderPos());
        sf::Vector2f wristPos = getWristPos(elbowPos);
        sf::Vector2f swordTip = getSwordTip(wristPos);
        outStart = wristPos;
        outEnd = swordTip;
    }

    // Return angle between sword direction and the body direction
    float getSwordBodyAngle() const
    {
        sf::Vector2f bodyVec = getHeadPos() - m_footPos;
        sf::Vector2f swordStart, swordEnd;
        getSwordLine(swordStart, swordEnd);
        sf::Vector2f swordVec = swordEnd - swordStart;
        return angleBetween(bodyVec, swordVec);
    }

    // Draw the bot
    void draw(sf::RenderWindow &window)
    {
        if (!m_isAlive)
            return;

        // Get the key points
        sf::Vector2f headPos = getHeadPos();
        sf::Vector2f shoulderPos = getShoulderPos();
        sf::Vector2f elbowPos = getElbowPos(shoulderPos);
        sf::Vector2f wristPos = getWristPos(elbowPos);
        sf::Vector2f swordTip = getSwordTip(wristPos);

        // Draw the body line foot->head
        drawLine(window, m_footPos, headPos, sf::Color::Green);

        // Arm lines
        drawLine(window, shoulderPos, elbowPos, sf::Color::White);
        drawLine(window, elbowPos, wristPos, sf::Color::White);
        drawLine(window, wristPos, swordTip, sf::Color::Yellow);

        // Circles for foot, head, shoulder, elbow, wrist
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

    sf::Vector2f getFootPos() const { return m_footPos; }

private:
    // Compute where the head is, given that the body can tilt by m_bodyAngle.
    // We define angle=0 => straight up. Positive angle => leaning one way, negative => the other.
    sf::Vector2f getHeadPos() const
    {
        // If angle=0 => head is exactly 120.f above foot on y-axis
        // We'll interpret m_bodyAngle so that it rotates around the foot:
        //   x offset = bodyLength * sin(angle)
        //   y offset = -bodyLength * cos(angle)  (minus because angle=0 means upward)
        float dx = m_bodyLength * std::sin(m_bodyAngle);
        float dy = -m_bodyLength * std::cos(m_bodyAngle);
        return sf::Vector2f(m_footPos.x + dx, m_footPos.y + dy);
    }

    // The shoulder is "shoulderOffset" below the top (the head), along the line from head->foot
    sf::Vector2f getShoulderPos() const
    {
        sf::Vector2f headPos = getHeadPos();
        // direction from head -> foot
        sf::Vector2f footDir = m_footPos - headPos;
        float len = std::sqrt(footDir.x * footDir.x + footDir.y * footDir.y);
        if (len < 1e-6f)
            return headPos; // degenerate

        sf::Vector2f unitDir = sf::Vector2f(footDir.x / len, footDir.y / len);
        // offset from head downward
        return headPos + unitDir * m_shoulderOffset;
    }

    sf::Vector2f getElbowPos(const sf::Vector2f &shoulderPos) const
    {
        return sf::Vector2f(shoulderPos.x + m_armLength * std::cos(m_armAngle),
                            shoulderPos.y + m_armLength * std::sin(m_armAngle));
    }

    sf::Vector2f getWristPos(const sf::Vector2f &elbowPos) const
    {
        float elbowGlobalAngle = m_armAngle + m_elbowAngle;
        return sf::Vector2f(elbowPos.x + m_forearmLength * std::cos(elbowGlobalAngle),
                            elbowPos.y + m_forearmLength * std::sin(elbowGlobalAngle));
    }

    sf::Vector2f getSwordTip(const sf::Vector2f &wristPos) const
    {
        float wristGlobalAngle = m_armAngle + m_elbowAngle + m_wristAngle;
        return sf::Vector2f(wristPos.x + m_handLength * std::cos(wristGlobalAngle),
                            wristPos.y + m_handLength * std::sin(wristGlobalAngle));
    }

    // A helper to guess the angle from the wrist to the opponent's foot
    float computeWristAimAngle(const sf::Vector2f &opponentFootPos)
    {
        sf::Vector2f shoulderPos = getShoulderPos();
        sf::Vector2f elbowPos = getElbowPos(shoulderPos);
        sf::Vector2f wristPos = getWristPos(elbowPos);

        sf::Vector2f dir = opponentFootPos - wristPos;
        return std::atan2(dir.y, dir.x);
    }

    // Just draws a line using a VertexArray
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
    // Basic properties
    sf::Vector2f m_footPos; // foot location
    float m_bodyLength;     // total length from foot to head
    float m_shoulderOffset;
    float m_bodyAngle; // new angle for "poking" (0=vertical, +/- tilt)
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

class neural {
public:
    // Constructor with default parameters
    neural(std::vector<uint> topology, Scalar evolutionRate = 0.005f, Scalar mutationRate = 0.1f) {
        this->topology = topology;
        this->mutationRate = mutationRate;
        this->evolutionRate = evolutionRate;
        for (uint i = 0; i < topology.size(); i++) {
            // For non-output layers, add one extra neuron for bias.
            if (i == topology.size() - 1)
                neuronLayers.push_back(new RowVector(topology[i]));
            else
                neuronLayers.push_back(new RowVector(topology[i] + 1));

            // Set the bias neuron to 1.0 for non-output layers.
            if (i != topology.size() - 1)
                neuronLayers.back()->coeffRef(topology[i]) = 1.0f;

            // Initialize weights matrix (starting from the second layer)
            if (i > 0) {
                if (i != topology.size() - 1) {
                    // Hidden layers: include bias for both previous and current layer.
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                    *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i] + 1);
                } else {
                    // Output layer: previous layer includes bias; output layer does not.
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                    *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i]);
                }
            }
        }
    }

    // Forward propagation: calculates activations through the network.
    void propagateForward(RowVector& input) {
        // Set the input layer (excluding the bias element)
        neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
    
        // Forward propagation: multiply by weights and apply activation function.
        for (uint i = 1; i < topology.size(); i++) {
            (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
            neuronLayers[i]->block(0, 0, 1, topology[i])
                = neuronLayers[i]->block(0, 0, 1, topology[i])
                    .unaryExpr([this](Scalar x) { return activationFunction(x); });
        }
    }

    // Mutate weights based on a mutation probability.
    void updateWeights() {
        // Iterate over each weight matrix.
        for (uint i = 0; i < weights.size(); i++) {
            for (uint r = 0; r < weights[i]->rows(); r++) {
                for (uint c = 0; c < weights[i]->cols(); c++) {
                    Scalar randVal = static_cast<Scalar>(rand()) / RAND_MAX;
                    if (randVal < mutationRate) {  // use the member mutationRate
                        Scalar mutation = evolutionRate * (2.0f * static_cast<Scalar>(rand()) / RAND_MAX - 1.0f);
                        weights[i]->coeffRef(r, c) += mutation;
                    }
                }
            }
        }
    }

    // Activation function: sigmoid in this case.
    Scalar activationFunction(Scalar x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Members
    std::vector<Matrix*> weights;
    std::vector<RowVector*> neuronLayers;
    std::vector<uint> topology;
    Scalar evolutionRate;
    Scalar mutationRate;
};


////////////////////////////////////////////////////////////
// Collision/knockback logic
////////////////////////////////////////////////////////////

// Sword vs. Sword => knockback
void checkSwordSwordCollision(Bot &A, Bot &B)
{
    if (!A.isAlive() || !B.isAlive())
        return;

    // Get each sword's line
    sf::Vector2f aSwordStart, aSwordEnd;
    A.getSwordLine(aSwordStart, aSwordEnd);
    sf::Vector2f bSwordStart, bSwordEnd;
    B.getSwordLine(bSwordStart, bSwordEnd);

    if (linesIntersect(aSwordStart, aSwordEnd, bSwordStart, bSwordEnd))
    {
        // Swords have clashed. Compute angles:
        float angleA = A.getSwordBodyAngle(); // 0=parallel, ~1.57=perp
        float angleB = B.getSwordBodyAngle();

        float forceA = std::fabs(std::sin(angleA));
        float forceB = std::fabs(std::sin(angleB));

        float knockbackScale = 20.f; // tweak as desired

        // Push each bot away from the other
        sf::Vector2f dirAB = B.getFootPos() - A.getFootPos();
        sf::Vector2f normAB = normalize(dirAB);
        sf::Vector2f normBA = -normAB; // opposite direction

        B.applyKnockback(normAB * (forceA * knockbackScale));
        A.applyKnockback(normBA * (forceB * knockbackScale));
    }
}

// Sword vs. Body => kill
void checkSwordHitsBody(Bot &attacker, Bot &victim)
{
    if (!attacker.isAlive() || !victim.isAlive())
        return;

    // Attacker's sword
    sf::Vector2f swordStart, swordEnd;
    attacker.getSwordLine(swordStart, swordEnd);

    // Victim's body line
    sf::Vector2f bodyStart, bodyEnd;
    victim.getBodyLine(bodyStart, bodyEnd);

    if (linesIntersect(swordStart, swordEnd, bodyStart, bodyEnd))
    {
        victim.kill();
    }
}

// Combined check for collisions between two bots
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
    sf::RenderWindow window(sf::VideoMode(800, 600),
                            "Leaning/Poking Bots + Walls + Collisions");
    window.setFramerateLimit(60);

    std::vector<uint> topology = {11, 100, 5};
    Scalar evolutionRate = 0.005;
    Scalar mutationRate = 0.1;
    neural net(topology, evolutionRate, mutationRate);

    // Two bots, left (not flipped) vs. right (flipped)
    Bot botA(250.f, 400.f, false);
    Bot botB(550.f, 400.f, true);

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

        // Update bots if they're alive
        if (botA.isAlive())
            botA.update(botB.getFootPos());
        if (botB.isAlive())
            botB.update(botA.getFootPos());

        // Check collisions between the two bots
        handleCollisions(botA, botB);

        // **Check for walls** => kill if foot crosses the boundary
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

        // Draw the walls (optional)
        // left wall
        sf::VertexArray lw(sf::Lines, 2);
        lw[0].position = sf::Vector2f(leftWallX, 0.f);
        lw[0].color = sf::Color::Magenta;
        lw[1].position = sf::Vector2f(leftWallX, 600.f);
        lw[1].color = sf::Color::Magenta;
        window.draw(lw);

        // right wall
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
