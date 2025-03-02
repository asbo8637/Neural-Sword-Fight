#include <SFML/Graphics.hpp>
#include <random>
#include <cmath>
#include <Eigen/dense>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef unsigned int uint;

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

class neural
{
    public:
    neural(std::vector<uint> topology, Scalar evolutionRate = Scalar(0.005),  Scalar mutationRate = Scalar(0.1));
    // function for forward propagation of data
    void propagateForward(RowVector& input);

    void updateWeights();

    std::vector<Matrix*> weights;
    std::vector<RowVector*> neuronLayers;
    std::vector<uint> topology;
    Scalar evolutionRate;
    Scalar mutationRate;

    Scalar activationFunction(Scalar x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    neural::neural(std::vector<uint> topology, Scalar evolutionRate, Scalar mutationRate)
    {
        this->topology = topology;
        this->mutationRate = mutationRate;
        this->evolutionRate = evolutionRate;
        for (uint i = 0; i < topology.size(); i++) {
            // Initialize neuron layers. For non-output layers, add one extra neuron for bias.
            if (i == topology.size() - 1)
                neuronLayers.push_back(new RowVector(topology[i]));
            else
                neuronLayers.push_back(new RowVector(topology[i] + 1));
    
            // Set the bias neuron to 1.0 for all non-output layers.
            if (i != topology.size() - 1) {
                neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            }
    
            // Initialize weights matrix (starting from the second layer)
            if (i > 0) {
                if (i != topology.size() - 1) {
                    // For hidden layers, dimensions include bias for both previous and current layer.
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                    *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i] + 1);
                }
                else {
                    // For the output layer, previous layer includes bias, but output layer does not.
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                    *weights.back() = Matrix::Random(topology[i - 1] + 1, topology[i]);
                }
            }
        }
    };
    

    void neural::propagateForward(RowVector& input)
    {
        // set the input to input layer
        // block returns a part of the given vector or matrix
        // block takes 4 arguments : startRow, startCol, blockRows, blockCols
        neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
    
        // propagate the data forward and then 
        // apply the activation function to your network
        // unaryExpr applies the given function to all elements of CURRENT_LAYER
        for (uint i = 1; i < topology.size(); i++) {
            // already explained above
            (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr([this](Scalar x) { return activationFunction(x); });
        }
    }

    
    void neural::updateWeights()
    {
        // Define a mutation probability (chance to mutate each weight)
        // For example, a mutationRate of 0.1 means there's a 10% chance to mutate each weight.
        Scalar mutationProbability = 0.1;
    
        // Iterate over each weight matrix
        for (uint i = 0; i < weights.size(); i++) {
            for (uint r = 0; r < weights[i]->rows(); r++) {
                for (uint c = 0; c < weights[i]->cols(); c++) {
                    // Generate a random value between 0 and 1
                    Scalar randVal = static_cast<Scalar>(rand()) / RAND_MAX;
                    // Check if this weight should be mutated
                    if (randVal < mutationProbability) {
                        // Generate a mutation value in the range [-1, 1]
                        Scalar mutation = evolutionRate * (2.0f * static_cast<Scalar>(rand()) / RAND_MAX - 1.0f);
                        // Update the weight
                        weights[i]->coeffRef(r, c) += mutation;
                    }
                }
            }
        }
    }    
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "2D Bot with Body & Offset Shoulder");
    window.setFramerateLimit(60);

    std::vector<uint> topology = {11, 100, 5};
    Scalar evolutionRate = 0.005;
    Scalar mutationRate = 0.1;
    neural net(topology, evolutionRate, mutationRate);

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