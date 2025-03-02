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
    // Constructor
    Bot(float footX, float footY, float sword, bool flipped = false)
        : m_footPos(footX, footY),
          m_speed(0.2f),
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

    float randomMovement() {
        // Generate a uniform random number r in [0,1]
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        // Scale to range [-0.2, 0.3]: range length is 0.5, then shift by -0.2.
        return -r*0.5f+0.2f;
    }
    // New update method: we pass in 5 values (0..1)
    // [0] => foot X, [1] => bodyAngle, [2] => armAngle, [3] => elbowAngle, [4] => wristAngle
    void updateFromNN(const std::array<float, 5>& controls)
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
        // If m_flipped is true, we want the bot to move right (positive x)
        // so that it faces the other bot; otherwise, move left.
        float direction = m_flipped ? 1.0f : -1.0f;
        
        // 1) Update foot x position.
        //m_footPos.x += 3.0f * direction * randomMovement();
        m_footPos.x -= 2.f * direction;
        
        float Pi = 3.14159f;
        
        // 2) Update body angle.
        // First, update and clamp the angle.
        // Let's assume the desired unclipped range for body angle is [0.3*Pi, 0.7*Pi]
        m_bodyAngle = std::max(0.9f * Pi, std::min(1.1f * Pi, m_bodyAngle + direction*controls[0] * m_speed));
        
        // For arm angle:
        m_armAngle = m_flipped ? (-m_armAngle) : m_armAngle;
        m_armAngle = std::max(0.2f * Pi, std::min(0.6f * Pi, m_armAngle + controls[1] * m_speed));
        m_armAngle = m_flipped ? (-m_armAngle) : m_armAngle;

        // For elbow angle:
        m_elbowAngle = m_flipped ? (-m_elbowAngle) : m_elbowAngle;
        m_elbowAngle = std::max(0.f * Pi, std::min(0.4f * Pi, m_elbowAngle + controls[2] * m_speed));
        m_elbowAngle = m_flipped ? (-m_elbowAngle) : m_elbowAngle;
        // For wrist angle:
        m_wristAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;
        m_wristAngle = std::max(-0.4f * Pi, std::min(0.2f * Pi, m_wristAngle + controls[3] * m_speed));
        m_wristAngle = m_flipped ? (-m_wristAngle) : m_wristAngle;

        sf::Vector2f newTip = getSwordTip(getWristPos(getElbowPos(getShoulderPos())));
        newTip = sf::Vector2f(newTip.x - m_footPos.x, newTip.y);
        m_momentum += newTip - last_sword_pos;
        //float dx = (m_last_foot_pos.x - m_footPos.x) * direction;
        m_angle_momentum = 100*
        4*std::abs(m_armAngle-last_arm_angle) + 
        3*std::abs(m_elbowAngle-last_elbow_angle) + 
        std::abs(m_wristAngle-last_wrist_angle) + 
        10*std::abs(m_bodyAngle-last_body_angle);
    }

    void launch_sword(){  
        m_elbowAngle *= 0.96f;
        m_wristAngle *= 0.96f;
        m_armAngle *= 0.99f;
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

    sf::Vector2f getMomentum(){
        return m_momentum;
    }

    float getAngleMomentum(){
        return m_angle_momentum;
    }

    float getSwordBodyAngle() const
    {
        sf::Vector2f bodyVec = getHeadPos() - m_footPos;
        sf::Vector2f swordStart, swordEnd;
        getSwordLine(swordStart, swordEnd);
        sf::Vector2f swordVec = swordEnd - swordStart;
        return angleBetween(bodyVec, swordVec);
    }

    std::array<float, 8> getAllyValues() const {
        // Adjust the foot x position if m_flipped.
        float footX = m_footPos.x;
        float temp_bodyAngle = m_bodyAngle;
        float temp_armAngle = m_armAngle;
        float temp_elbowAngle = m_elbowAngle;
        float temp_wristAngle = m_wristAngle;
        if (m_flipped) {
            footX = 800 - footX;
            float pi = 3.14159f;
            temp_bodyAngle = - m_bodyAngle;
            temp_armAngle = - m_armAngle;
            temp_elbowAngle =  - m_elbowAngle;
            temp_wristAngle = - m_wristAngle;
        }
        // Normalize footX from [100, 700] to [0, 1].
        float normFootX = (footX - 100.0f) / 600.0f;
        
        // Helper lambda to normalize an angle.
        auto normalizeAngle = [](float angle) -> float {
            if (angle < 0)
                angle += 2.0f * 3.14159f;
            return angle / (2.0f * 3.14159f);
        };
        
        float normBodyAngle  = normalizeAngle(temp_bodyAngle);
        float normArmAngle   = normalizeAngle(temp_armAngle);
        float normElbowAngle = normalizeAngle(temp_elbowAngle);
        float normWristAngle = normalizeAngle(temp_wristAngle);
        
        // Return ally values as [normFootX, normBodyAngle, normArmAngle, normElbowAngle, normWristAngle]
        return { normFootX, normBodyAngle, normArmAngle, normElbowAngle, normWristAngle, m_momentum.x, m_momentum.y, static_cast<float>(m_collision_amount) };
    }
    
    int getCollisionAmount() const {
        return m_collision_amount;
    }
    void incrementCollisionAmount() {
        m_collision_amount+=0.5f;
    }
    
    std::array<float, 7> getEnemyValues() const {
        float footX = m_footPos.x;
        float temp_bodyAngle = m_bodyAngle;
        float temp_armAngle = m_armAngle;
        float temp_elbowAngle = m_elbowAngle;
        float temp_wristAngle = m_wristAngle;
        if (m_flipped) {
            footX = 800 - footX;
            float pi = 3.14159f;
            temp_bodyAngle = - m_bodyAngle;
            temp_armAngle = - m_armAngle;
            temp_elbowAngle =  - m_elbowAngle;
            temp_wristAngle = - m_wristAngle;
        }
        // Normalize x values from [100,700] to [0,1]
        float normFootX = (footX - 100.0f) / 600.0f;

        auto normalizeAngle = [](float angle) -> float {
            if (angle < 0)
                angle += 2.0f * 3.14159f;
            return angle / (2.0f * 3.14159f);
        };

        // Normalize the body angle as before.
        float normBodyAngle  = normalizeAngle(temp_bodyAngle);
        float normArmAngle   = normalizeAngle(temp_armAngle);
        float normElbowAngle = normalizeAngle(temp_elbowAngle);
        float normWristAngle = normalizeAngle(temp_wristAngle);
    
        // Return enemy values as [normFootX, normBodyAngle, normSwordStartX, normSwordStartY, normSwordEndX, normSwordEndY]
        return { normFootX, normBodyAngle, normArmAngle, normElbowAngle, normWristAngle,  m_momentum.x, m_momentum.y};
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
    // Compute the head position assuming m_bodyAngle is measured from the horizontal.
    sf::Vector2f getHeadPos() const
    {
        // With 0 radians pointing to the right, use cosine for x and sine for y.
        float dx = m_bodyLength * std::sin(m_bodyAngle);
        float dy = m_bodyLength * std::cos(m_bodyAngle);
        return sf::Vector2f(m_footPos.x + dx, m_footPos.y + dy);
    }

    // Compute the shoulder position along the line from head to foot.
    sf::Vector2f getShoulderPos() const
    {
        sf::Vector2f headPos = getHeadPos();
        sf::Vector2f footDir = m_footPos - headPos;
        float len = std::sqrt(footDir.x * footDir.x + footDir.y * footDir.y);
        if (len < 1e-6f)
            return headPos;
        sf::Vector2f unitDir(footDir.x / len, footDir.y / len);
        // Offset the head position toward the foot by the shoulder offset.
        return headPos + unitDir * m_shoulderOffset;
    }

    // Compute the elbow position based on the shoulder position and m_armAngle.
    // m_armAngle is now measured from the horizontal.
    sf::Vector2f getElbowPos(const sf::Vector2f &shoulderPos) const
    {
        return sf::Vector2f(
            shoulderPos.x + m_armLength * std::sin(m_armAngle),
            shoulderPos.y + m_armLength * std::cos(m_armAngle)
        );
    }

    // Compute the wrist position by adding the forearm vector.
    // The global angle at the elbow is m_armAngle + m_elbowAngle.
    sf::Vector2f getWristPos(const sf::Vector2f &elbowPos) const
    {
        float elbowGlobalAngle = m_armAngle + m_elbowAngle;
        return sf::Vector2f(
            elbowPos.x + m_forearmLength * std::sin(elbowGlobalAngle),
            elbowPos.y + m_forearmLength * std::cos(elbowGlobalAngle)
        );
    }

    // Compute the sword tip position by extending from the wrist.
    // The global angle at the wrist is m_armAngle + m_elbowAngle + m_wristAngle.
    sf::Vector2f getSwordTip(const sf::Vector2f &wristPos) const
    {
        float wristGlobalAngle = m_armAngle + m_elbowAngle + m_wristAngle;
        return sf::Vector2f(
            wristPos.x + m_handLength * std::sin(wristGlobalAngle),
            wristPos.y + m_handLength * std::cos(wristGlobalAngle)
        );
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
    int m_collision_amount;
    sf::Vector2f m_momentum;
    float m_angle_momentum;

    // Visuals
    sf::CircleShape m_jointCircle;
    sf::CircleShape m_head;
};

class neural {
    public:
        // Constructor with default parameters.
        neural(std::vector<uint> topology, Scalar evolutionRate = 0.005f, Scalar mutationRate = 0.1f) {
            this->topology = topology;
            this->evolutionRate = evolutionRate;
            this->mutationRate = mutationRate;
            allocateLayersAndWeights();
        }
        
        // Copy constructor (deep copy)
        neural(const neural &other) {
            topology = other.topology;
            evolutionRate = other.evolutionRate;
            mutationRate = other.mutationRate;
            allocateLayersAndWeights();
            // Copy weights
            for (size_t i = 0; i < weights.size(); i++) {
                *weights[i] = *other.weights[i];
            }
        }
        
        // Copy assignment operator (deep copy)
        neural& operator=(const neural &other) {
            if (this != &other) {
                // First, free current memory.
                for (Matrix* w : weights) { delete w; }
                for (RowVector* layer : neuronLayers) { delete layer; }
                weights.clear();
                neuronLayers.clear();
                
                topology = other.topology;
                evolutionRate = other.evolutionRate;
                mutationRate = other.mutationRate;
                allocateLayersAndWeights();
                // Copy weights.
                for (size_t i = 0; i < weights.size(); i++) {
                    *weights[i] = *other.weights[i];
                }
            }
            return *this;
        }
        
        // Destructor: clean up allocated weights and neuron layers.
        ~neural() {
            for (Matrix* w : weights)
                delete w;
            for (RowVector* layer : neuronLayers)
                delete layer;
        }
        
        // Forward propagation: calculates activations through the network.
        void propagateForward(RowVector& input) {
            // Set the input layer (excluding the bias element).
            neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
        
            // For each subsequent layer, compute weighted sum and apply activation.
            for (size_t i = 1; i < topology.size(); i++) {
                (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
                neuronLayers[i]->block(0, 0, 1, topology[i]) =
                    neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr([this](Scalar x) {
                        return activationFunction(x);
                    });
            }
        }
        
        // getOutput returns the activated outputs from the final layer, mapped from [0,1] to [-1,1].
        std::vector<Scalar> getOutput() const {
            const RowVector* outputLayer = neuronLayers.back();
            std::vector<Scalar> output(topology.back());
            for (size_t i = 0; i < topology.back(); i++) {
                output[i] = 2 * (*outputLayer)(i) - 1;
            }
            return output;
        }
        
        // Mutate weights randomly based on the mutation rate.
        void updateWeights() {
            for (size_t i = 0; i < weights.size(); i++) {
                for (size_t r = 0; r < weights[i]->rows(); r++) {
                    for (size_t c = 0; c < weights[i]->cols(); c++) {
                        Scalar randVal = static_cast<Scalar>(rand()) / RAND_MAX;
                        if (randVal < mutationRate) {
                            Scalar mutation = evolutionRate * (2.0f * static_cast<Scalar>(rand()) / RAND_MAX - 1.0f);
                            weights[i]->coeffRef(r, c) += mutation;
                        }
                    }
                }
            }
        }
        
        // Activation function: sigmoid.
        Scalar activationFunction(Scalar x) {
            return 1.0f / (1.0f + std::exp(-x));
        }
        
        // Clone method to create a copy of the neural network.
        neural clone() const {
            return neural(*this); // Uses the copy constructor.
        }
        
        // Members.
        std::vector<Matrix*> weights;
        std::vector<RowVector*> neuronLayers;
        std::vector<uint> topology;
        Scalar evolutionRate;
        Scalar mutationRate;
        
    private:
        // Helper function to allocate neuron layers and weight matrices.
        void allocateLayersAndWeights() {
            // Create neuron layers.
            for (size_t i = 0; i < topology.size(); i++) {
                if (i == topology.size() - 1)
                    neuronLayers.push_back(new RowVector(topology[i]));  // Output layer.
                else
                    neuronLayers.push_back(new RowVector(topology[i] + 1)); // Hidden/input layers with bias.
        
                if (i != topology.size() - 1)
                    neuronLayers.back()->coeffRef(topology[i]) = 1.0f;
        
                // Allocate weight matrices.
                if (i > 0) {
                    if (i != topology.size() - 1) {
                        // Hidden layers: previous layer (with bias) to current layer (with bias).
                        weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                        *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i] + 1);
                    } else {
                        // Output layer: previous layer (with bias) to output layer (no bias).
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
        float angleA = A.getSwordBodyAngle();
        float angleB = B.getSwordBodyAngle();

        float forceSinA = std::fabs(std::cos(angleA));
        float forceSinB = std::fabs(std::cos(angleB));

        int collisions = A.getCollisionAmount();
        float knockbackScale = collisions*0.4f;
        float aMom = forceSinA*std::sqrt(A.getMomentum().x * A.getMomentum().x + A.getMomentum().y * A.getMomentum().y); //euclidean distance
        float bMom = forceSinB*std::sqrt(B.getMomentum().x * B.getMomentum().x + B.getMomentum().y * B.getMomentum().y);
        float aAom = A.getAngleMomentum();
        float bAom = B.getAngleMomentum();
        // float forceB = -( knockbackScale) * (std::abs(aMom) + std::abs(aAom));
        // float forceA = ( knockbackScale) * (0.05*std::abs(bMom) + std::abs(bAom));
        float forceB = -forceSinB*std::max(5.f, std::min(85.f, ( knockbackScale * std::abs(bAom)*std::abs(bMom))));
        float forceA = forceSinA*std::max(5.f, std::min(85.f, ( knockbackScale * std::abs(aAom)*std::abs(aMom))));
        B.applyKnockback(forceB);
        A.applyKnockback(forceA);
        if(forceA != 0 || forceB !=0) A.incrementCollisionAmount();
        // A.launch_sword();
        // B.launch_sword();
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
Eigen::RowVectorXf arrayToEigen(const std::array<float, N>& arr) {
    Eigen::RowVectorXf vec(N);
    for (size_t i = 0; i < N; ++i) {
        vec(i) = arr[i];
    }
    return vec;
}

Eigen::RowVectorXf getInputForBot(Bot botA, Bot botB){
    // For net1: combine botA's ally values with botB's enemy values.
    auto botAAlly = botA.getAllyValues();   // std::array<float, 8>
    auto botBEnemy = botB.getEnemyValues();   // std::array<float, 7>

    // Convert each array to an Eigen row vector.
    Eigen::RowVectorXf vecA = arrayToEigen(botAAlly);
    Eigen::RowVectorXf vecB = arrayToEigen(botBEnemy);

    Eigen::RowVectorXf inputForNet(15);
    inputForNet << vecA, vecB;  // Concatenates the two vectors
    return inputForNet;
}

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////
neural learn(bool switch_bot, neural net1, neural net2, int round_count){
    int afterRounds=round_count;
    int endRounds=round_count+=50;
    int timer = 500;
    int rounds=0;
    int lastLoss=0; 
    int consecutiveRounds=0;
    sf::RenderWindow window(sf::VideoMode(800, 600), "NN Control Example");
    window.setFramerateLimit(300);

    // Create two bots
    Bot botA(150.f, 400.f, 130.f, false);
    Bot botB(650.f, 400.f, 130.f, true);

    // Example "walls"
    float leftWallX = 100.f;
    float rightWallX = 700.f;


    while (window.isOpen() && rounds<endRounds)
    {
        if(timer<0){
            timer=1000;
            botA = Bot(150.f, 400.f, 130.f, false);
            botB = Bot(650.f, 400.f, 130.f, true);
            net1.updateWeights();
            net2.updateWeights();
            std::cout << "TIMER RUNG! Round: " << rounds << std::endl;
        }
        timer--;

        if (botA.isAlive() && timer> 0){
            Eigen::RowVectorXf inputForNet1(15);
            inputForNet1 = getInputForBot(botA, botB);
            net1.propagateForward(inputForNet1);
            std::vector<Scalar> output = net1.getOutput();
            // Convert the output to a std::array<float, 5>
            std::array<float, 5> controlsA;
            std::copy_n(output.begin(), 5, controlsA.begin());
            botA.updateFromNN(controlsA);
        }
        else{
            timer=1000;
            botA = Bot(150.f, 400.f, 130.f, false);
            botB = Bot(650.f, 400.f, 130.f, true);
            rounds+=1;
            std::cout << "B WINS! Round: " << rounds << std::endl;
            consecutiveRounds+=1;
            if(lastLoss!=1 || consecutiveRounds>25){
                if( switch_bot ) net1=net2.clone();
                lastLoss=1;
                consecutiveRounds=0;
            }
            net1.updateWeights();
            continue;
        }
        if (botB.isAlive()){
            Eigen::RowVectorXf inputForNet2(15);
            inputForNet2 = getInputForBot(botB, botA);
            net2.propagateForward(inputForNet2);
            std::vector<Scalar> output = net2.getOutput();
            // Convert the output to a std::array<float, 5>
            std::array<float, 5> controlsB;
            std::copy_n(output.begin(), 5, controlsB.begin());
            botB.updateFromNN(controlsB);
        }
        else{
            timer=1000;
            botA = Bot(150.f, 400.f, 90.f, false);
            botB = Bot(650.f, 400.f, 90.f, true);
            rounds+=1;
            std::cout << "A WINS! Round: " << rounds << std::endl;
            consecutiveRounds+=1;
            if(lastLoss!=2 || consecutiveRounds>25){
                if( switch_bot ) net2 = net1.clone();
                lastLoss=2;
                consecutiveRounds = 0;
            }
            net2.updateWeights();
            continue;
        }


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

        if(rounds>=afterRounds){
            window.display();
                    // Poll events
            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                    window.close();
            }
        }
    }
    return net1;
}


int main()
{
    std::vector<uint> topology = {15, 100, 100, 5};
    Scalar evolutionRate = 0.1;
    Scalar mutationRate = 0.5;
    neural net1(topology, evolutionRate, mutationRate);
    neural net2(topology, evolutionRate, mutationRate);
    neural net3(topology, evolutionRate, mutationRate);
    neural net4(topology, evolutionRate, mutationRate);

    srand(static_cast<unsigned int>(time(0)));
    neural champ1 = learn(true, net1, net2, 500);
    neural champ2 = learn(true, net3, net4, 2500);
    neural champ3 = learn(false, champ1, champ2, 250);
    return 0;
}
