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
    // Original constructor (unchanged)
    Bot(float footX, float footY, float sword, bool flipped = false)
        : m_footPos(footX, footY),
          m_speed(0.05f),
          m_bodyLength(120.f),
          m_shoulderOffset(40.f),
          m_bodyAngle(3.1415f),
          m_armAngle(0.f),
          m_elbowAngle(0.f),
          m_wristAngle(0.f),
          m_armLength(30.f),
          m_forearmLength(40.f),
          m_handLength(sword),
          m_isAlive(true),
          m_flipped(flipped),
          m_collision_amount(1),
          m_momentum(0.f, 0.f),
          m_angle_momentum(0)
    {
        // Circles for joints   
        m_jointCircle.setRadius(2.f);
        m_jointCircle.setOrigin(5.f, 5.f);
        m_jointCircle.setFillColor(sf::Color::Red);

        m_head.setRadius(10.f);
        m_head.setOrigin(10.f, 10.f);
        m_head.setFillColor(sf::Color::Blue);

        // If flipped, rotate the entire arm 180 degrees
        if (m_flipped)
            m_armAngle = 3.14159f; // ~180 deg
    }

    // *** NEW Overloaded Constructor ***
    // Allows specifying speed, body length, and sword length at creation
    Bot(float footX, float footY, float sword, float spd, float bodyLen, bool flipped)
        : m_footPos(footX, footY),
          m_speed(spd),
          m_bodyLength(bodyLen),
          m_shoulderOffset(40.f),
          m_bodyAngle(3.1415f),
          m_armAngle(0.f),
          m_elbowAngle(0.f),
          m_wristAngle(0.f),
          m_armLength(40.f),
          m_forearmLength(50.f),
          m_handLength(sword),
          m_isAlive(true),
          m_flipped(flipped),
          m_collision_amount(1),
          m_momentum(0.f, 0.f),
          m_angle_momentum(0)
    {
        // Circles for joints
        m_jointCircle.setRadius(2.f);
        m_jointCircle.setOrigin(2.f, 2.f);
        m_jointCircle.setFillColor(sf::Color::Red);

        m_head.setRadius(10.f);
        m_head.setOrigin(10.f, 10.f);
        m_head.setFillColor(sf::Color::Blue);

        // If flipped, rotate the entire arm 180 degrees
        if (m_flipped)
            m_armAngle = 3.14159f; // ~180 deg
    }

    // New update method: we pass in 5 values (0..1)
    // [0] => foot X, [1] => armAngle, [2] => elbowAngle, [3] => wristAngle, [4] => ???
    void updateFromNN(const std::array<float, 5> &controls)
    {
        sf::Vector2f last_sword_pos = getSwordTip(getWristPos(getElbowPos(getShoulderPos())));
        last_sword_pos = sf::Vector2f(last_sword_pos.x - m_footPos.x, last_sword_pos.y);
        float last_wrist_angle = m_wristAngle;
        float last_elbow_angle = m_elbowAngle;
        float last_arm_angle = m_armAngle;
        float last_body_angle = m_bodyAngle;
        if (!m_isAlive)
            return;

        // Define boundaries for the foot's x position.
        const float minX = 100.f;
        const float maxX = 700.f;

        // Determine foot movement direction.
        float direction = m_flipped ? 1.0f : -1.0f;

        // (1) Move footX
        m_footPos.x -= 1.3f * direction * controls[0];

        float Pi = 3.14159f;

        // (2) For arm angle:
        m_armAngle = m_flipped ? (-m_armAngle) : m_armAngle;
        //m_armAngle = std::max(0.2f * Pi, std::min(0.6f * Pi, m_armAngle + controls[1] * m_speed));
        m_armAngle += controls[1] * m_speed;
        m_armAngle = m_flipped ? (-m_armAngle) : m_armAngle;

        // (3) For elbow angle:
        m_elbowAngle = m_flipped ? (-m_elbowAngle) : m_elbowAngle;
        //m_elbowAngle = std::max(0.f * Pi, std::min(0.4f * Pi, m_elbowAngle + controls[2] * m_speed));
        m_elbowAngle += controls[2] * m_speed;
        m_elbowAngle = m_flipped ? (-m_elbowAngle) : m_elbowAngle;

        // (4) For wrist angle:
        m_wristAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;
        //m_wristAngle = std::max(-0.4f * Pi, std::min(0.2f * Pi, m_wristAngle + controls[3] * m_speed));
        m_wristAngle += controls[3] * m_speed;
        m_wristAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;

        m_bodyAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;
        m_bodyAngle  = std::max(-0.3f * Pi, std::min(0.3f * Pi, m_bodyAngle + controls[4] * m_speed));
        m_bodyAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;

        sf::Vector2f newTip = getSwordTip(getWristPos(getElbowPos(getShoulderPos())));
        newTip = sf::Vector2f(newTip.x - m_footPos.x, newTip.y);
        m_momentum += newTip - last_sword_pos;
        m_angle_momentum = 100 *
                            4 * std::abs(m_armAngle - last_arm_angle) +
                           3 * std::abs(m_elbowAngle - last_elbow_angle) +
                           std::abs(m_wristAngle - last_wrist_angle) +
                           10 * std::abs(m_bodyAngle - last_body_angle);
    }

    void kill() { m_isAlive = false; }
    bool isAlive() const { return m_isAlive; }
    sf::Vector2f getFootPos() const { return m_footPos; }

    // Knockback
    void applyKnockback(float disp)
    {
        if (m_isAlive)
            m_footPos.x -= disp;
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

    sf::Vector2f getMomentum()
    {
        return m_momentum;
    }

    float getAngleMomentum()
    {
        return m_angle_momentum;
    }

    float getSwordBodyAngle() const
    {
        sf::Vector2f swordStart, swordEnd;
        getSwordLine(swordStart, swordEnd);
        return angleBetween(swordEnd, swordStart);
    }

    std::array<float, 7> getAllyValues() const
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

        return {normFootX, normBodyAngle, normArmAngle, normElbowAngle, normWristAngle,
                m_momentum.x, m_momentum.y};
    }

    int getCollisionAmount() const
    {
        return m_collision_amount;
    }
    void incrementCollisionAmount()
    {
        m_collision_amount += 0.5f;
    }

    std::array<float, 7> getEnemyValues() const
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

        return {normFootX, normBodyAngle, normArmAngle, normElbowAngle, normWristAngle,
                m_momentum.x, m_momentum.y};
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

    sf::Vector2f getShoulderPos() const
    {
        sf::Vector2f headPos = getHeadPos();
        sf::Vector2f footDir = m_footPos - headPos;
        float len = std::sqrt(footDir.x * footDir.x + footDir.y * footDir.y);
        if (len < 1e-6f)
            return headPos;
        sf::Vector2f unitDir(footDir.x / len, footDir.y / len);
        return headPos + unitDir * m_shoulderOffset;
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

    bool m_isAlive;
    bool m_flipped;
    int m_collision_amount;
    sf::Vector2f m_momentum;
    float m_angle_momentum;

    // Visuals
    sf::CircleShape m_jointCircle;
    sf::CircleShape m_head;
};

class neural
{
public:
    // Constructor with default parameters.
    neural(std::vector<uint> topology, Scalar evolutionRate = 0.005f, Scalar mutationRate = 0.1f)
    {
        this->topology = topology;
        this->evolutionRate = evolutionRate;
        this->mutationRate = mutationRate;
        allocateLayersAndWeights();
    }

    // Copy constructor (deep copy)
    neural(const neural &other)
    {
        topology = other.topology;
        evolutionRate = other.evolutionRate;
        mutationRate = other.mutationRate;
        allocateLayersAndWeights();
        for (size_t i = 0; i < weights.size(); i++)
        {
            *weights[i] = *other.weights[i];
        }
    }

    // Copy assignment operator (deep copy)
    neural &operator=(const neural &other)
    {
        if (this != &other)
        {
            for (Matrix *w : weights)
            {
                delete w;
            }
            for (RowVector *layer : neuronLayers)
            {
                delete layer;
            }
            weights.clear();
            neuronLayers.clear();

            topology = other.topology;
            evolutionRate = other.evolutionRate;
            mutationRate = other.mutationRate;
            allocateLayersAndWeights();
            for (size_t i = 0; i < weights.size(); i++)
            {
                *weights[i] = *other.weights[i];
            }
        }
        return *this;
    }

    ~neural()
    {
        for (Matrix *w : weights)
            delete w;
        for (RowVector *layer : neuronLayers)
            delete layer;
    }

    // Forward propagation
    void propagateForward(RowVector &input)
    {
        neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

        for (size_t i = 1; i < topology.size(); i++)
        {
            (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
            neuronLayers[i]->block(0, 0, 1, topology[i]) =
                neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr([this](Scalar x)
                                                                       { return activationFunction(x); });
        }
    }

    // Returns final layer as values mapped from [0,1] to [-1,1].
    std::vector<Scalar> getOutput() const
    {
        const RowVector *outputLayer = neuronLayers.back();
        std::vector<Scalar> output(topology.back());
        for (size_t i = 0; i < topology.back(); i++)
        {
            output[i] = 2 * (*outputLayer)(i)-1;
        }
        return output;
    }

    // Mutate weights
    void updateWeights()
    {
        for (size_t i = 0; i < weights.size(); i++)
        {
            for (size_t r = 0; r < weights[i]->rows(); r++)
            {
                for (size_t c = 0; c < weights[i]->cols(); c++)
                {
                    Scalar randVal = static_cast<Scalar>(rand()) / RAND_MAX;
                    if (randVal < mutationRate)
                    {
                        Scalar mutation = evolutionRate * (2.0f * static_cast<Scalar>(rand()) / RAND_MAX - 1.0f);
                        weights[i]->coeffRef(r, c) += mutation;
                    }
                }
            }
        }
    }

    Scalar activationFunction(Scalar x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Clone method
    neural clone() const
    {
        return neural(*this);
    }

    std::vector<Matrix *> weights;
    std::vector<RowVector *> neuronLayers;
    std::vector<uint> topology;
    Scalar evolutionRate;
    Scalar mutationRate;

private:
    void allocateLayersAndWeights()
    {
        for (size_t i = 0; i < topology.size(); i++)
        {
            if (i == topology.size() - 1)
                neuronLayers.push_back(new RowVector(topology[i]));
            else
                neuronLayers.push_back(new RowVector(topology[i] + 1));

            if (i != topology.size() - 1)
                neuronLayers.back()->coeffRef(topology[i]) = 1.0f;

            if (i > 0)
            {
                if (i != topology.size() - 1)
                {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                    *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i] + 1);
                }
                else
                {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                    *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i]);
                }
            }
        }
    }
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
        B.applyKnockback(-5);
        A.applyKnockback(5);
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

template <size_t N>
Eigen::RowVectorXf arrayToEigen(const std::array<float, N> &arr)
{
    Eigen::RowVectorXf vec(N);
    for (size_t i = 0; i < N; ++i)
    {
        vec(i) = arr[i];
    }
    return vec;
}

// Construct the 15-dimensional input for the net controlling botA, given botB as enemy
Eigen::RowVectorXf getInputForBot(Bot botA, Bot botB, float timer)
{
    timer/=1000;
    auto botAAlly = botA.getAllyValues();   // std::array<float, 8>
    auto botBEnemy = botB.getEnemyValues(); // std::array<float, 7>

    Eigen::RowVectorXf vecA = arrayToEigen(botAAlly);
    Eigen::RowVectorXf vecB = arrayToEigen(botBEnemy);

    Eigen::RowVectorXf inputForNet(15);
    inputForNet << vecA, vecB, timer;
    return inputForNet;
}

void drawWalls(sf::RenderWindow &window){
    sf::VertexArray horizontalLine(sf::Lines, 2);
    horizontalLine[0].position = sf::Vector2f(0.f, 400.f);
    horizontalLine[0].color = sf::Color::Magenta;
    horizontalLine[1].position = sf::Vector2f(700.f, 400.f);
    horizontalLine[1].color = sf::Color::Magenta;
    window.draw(horizontalLine);

    sf::VertexArray rw(sf::Lines, 2);
    rw[0].position = sf::Vector2f(700, 0.f);
    rw[0].color = sf::Color::Magenta;
    rw[1].position = sf::Vector2f(700, 400.f);
    rw[1].color = sf::Color::Magenta;
    window.draw(rw);
}

int one_round(neural net1, neural net2, Bot botA, Bot botB, int rounds, int endRounds, sf::RenderWindow &window){
    float rightWallX = 700.f;
    int timer = 800;
    std::vector<Scalar> output;
    std::array<float, 5> controlsA;
    std::array<float, 5> controlsB;
    while(timer>0){
        timer--;
        if(!botA.isAlive() || !botB.isAlive()){
            std::cout << "A wins, Round: " << rounds << std::endl;
            return 1;
        }
        else{
            //Update BotA. It moves first
            Eigen::RowVectorXf inputForNet1 = getInputForBot(botA, botB, timer);
            net1.propagateForward(inputForNet1);
            output = net1.getOutput();
            std::copy_n(output.begin(), 5, controlsA.begin());
            botA.updateFromNN(controlsA);

            //Update BotB. It moves second
            Eigen::RowVectorXf inputForNet2 = getInputForBot(botB, botA, timer);
            net2.propagateForward(inputForNet2);
            output = net2.getOutput();
            std::copy_n(output.begin(), 5, controlsB.begin());
            botB.updateFromNN(controlsB);


            //Deal with collisions and detect deaths: 
            handleCollisions(botA, botB);

            // Kill if cross walls
            if (botA.isAlive() && botB.isAlive())
            {
                float xA = botA.getFootPos().x;
                if (xA > rightWallX)
                    botA.kill();
                float xB = botB.getFootPos().x;
                if (xB > rightWallX)
                    botB.kill();
            }

            // Draw
            if(rounds>endRounds){
                std::cout<<"WhereDraw"<<std::endl;
                window.clear(sf::Color(50, 50, 50));
                botA.draw(window);
                botB.draw(window);
                drawWalls(window);
            }
        }
    }
    std::cout << "B wins, Round: " << rounds << std::endl;
    return 2;
}


neural betterLearn(neural net1, neural net2, int display_round, float swordA, float speedA, float bodyA)
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "Tournament Match");
    window.setFramerateLimit(600);
    int timer = 800;
    int rounds = 0;
    int lastLoss = 0;
    int consecutiveRounds = 0;
    Bot botA = Bot(150.f, 400.f, swordA, speedA, bodyA, false);
    Bot botB = Bot(650.f, 400.f, swordA, speedA, bodyA, true);
    neural lastNet1=net1.clone();
    neural lastNet2=net2.clone();
    int winner=0;
    int lastWinner=0;

    while (rounds < display_round+1000)
    {
        rounds++;
        winner=one_round(net1, net2, botA, botB, rounds, display_round, window);
        botA = Bot(150.f, 400.f, swordA, speedA, bodyA, false);
        botB = Bot(650.f, 400.f, swordA, speedA, bodyA, true);

        if(winner!=lastWinner){
            consecutiveRounds=0;
            lastNet1=net1.clone();
            lastNet2=net2.clone();
        }

        if(winner=1){
            if(consecutiveRounds>20){
                net2=lastNet2.clone();
                consecutiveRounds=0;
            }
            net2.updateWeights();
        }
        else{
            if(consecutiveRounds>20){
                net1=lastNet1.clone();
                consecutiveRounds=0;
            }
            net1.updateWeights();
        }
    }
    return net1;
}

////////////////////////////////////////////////////////////
// Main: 16-bot tournament
////////////////////////////////////////////////////////////
int main()
{
    std::vector<uint> topology = {15, 100, 100, 100, 5};
    Scalar evolutionRate = 0.1f;
    Scalar mutationRate = 0.5f;

    neural net1(topology, evolutionRate, mutationRate);
    neural net2(topology, evolutionRate, mutationRate);
    srand(static_cast<unsigned int>(time(0)));

    // Define 8 different sets of attributes.
    betterLearn(net1, net2, 0, 50.f, 0.5f, 10.f);
                                
    return 0;
}
