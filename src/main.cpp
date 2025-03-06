#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <Eigen/dense>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include "bot.cpp"
#include "neural.cpp"
#include <algorithm>
#include <random>

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
    bSwordStart.x+=1.f;
    aSwordStart.x-=1.f;

    if (linesIntersect(aSwordStart, aSwordEnd, bSwordStart, bSwordEnd))
    {
        float Bforce = -120.f*(800-B.getFootPos().x)/800.f;
        float Aforce = 120.f*A.getFootPos().x/800.f;
        B.applyKnockback(Bforce);
        A.applyKnockback(Aforce);
        if(A.get_m_momentum()>B.get_m_momentum()){
            A.incrementScore();
        }
        else{
            B.incrementScore();
        }
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
    // checkSwordHitsBody(A, B);
    // checkSwordHitsBody(B, A);
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
Eigen::RowVectorXf getInputForBot(Bot botA, Bot botB, float timer, bool flipScore)
{
    timer/=500;
    auto botAAlly = botA.getAllyValues();   // std::array<float, 6>
    auto botBEnemy = botB.getAllyValues();

    Eigen::RowVectorXf vecA = arrayToEigen(botAAlly);
    Eigen::RowVectorXf vecB = arrayToEigen(botBEnemy);
    float distance = (botB.getFootPos().x - botA.getFootPos().x)/800.f;

    float scoreDif = flipScore ?  botB.getScore() - botA.getScore() : botA.getScore() - botB.getScore();
    Eigen::RowVectorXf inputForNet(15);
    inputForNet << vecA, vecB, timer, distance, scoreDif;
    return inputForNet;
}

void drawWalls(sf::RenderWindow &window){
    sf::VertexArray horizontalLine(sf::Lines, 2);
    horizontalLine[0].position = sf::Vector2f(0.f, 400.f);
    horizontalLine[0].color = sf::Color::Magenta;
    horizontalLine[1].position = sf::Vector2f(800.f, 400.f);
    horizontalLine[1].color = sf::Color::Magenta;
    window.draw(horizontalLine);

    // sf::VertexArray rw(sf::Lines, 2);
    // rw[0].position = sf::Vector2f(700, 0.f);
    // rw[0].color = sf::Color::Magenta;
    // rw[1].position = sf::Vector2f(700, 400.f);
    // rw[1].color = sf::Color::Magenta;
    //window.draw(rw);
}

int one_round(neural net1, neural net2, Bot &botA, Bot &botB, int rounds, bool display, sf::RenderWindow &window){
    int timer = 500;
    std::vector<Scalar> output;
    std::array<float, 5> controlsA;
    std::array<float, 5> controlsB;
    while(timer>0){
        timer--;
        if(!botA.isAlive()){
            return 2;
        }
        if(!botB.isAlive()){
            return 1;
        }
        else{
            //Get Input Output BotB.
            Eigen::RowVectorXf inputForNet2 = getInputForBot(botB, botA, timer, true);
            net2.propagateForward(inputForNet2);
            output = net2.getOutput();
            std::copy_n(output.begin(), 5, controlsB.begin());

            //Get Input Output BotA.
            Eigen::RowVectorXf inputForNet1 = getInputForBot(botA, botB, timer, false);
            net1.propagateForward(inputForNet1);
            output = net1.getOutput();
            std::copy_n(output.begin(), 5, controlsA.begin());
            
            //Update Bots
            botB.updateFromNN(controlsB);
            botA.updateFromNN(controlsA);


            //Deal with collisions and detect deaths: 
            handleCollisions(botA, botB);

            //Kill if cross walls
            if (botA.isAlive() && botB.isAlive())
            {
                float xB = botB.getFootPos().x;
                if (xB > 800 || xB <0) botB.kill();
                float xA = botA.getFootPos().x;
                if (xA > 800 || xA <0) botA.kill();
            }

            // Draw
            if(display){
                window.clear(sf::Color(50, 50, 50));
                botA.draw(window);
                botB.draw(window);
                drawWalls(window);
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
    }
    return 3;
}

std::vector<neural> createInitialPopulation(int populationSize) {
    std::vector<neural> population;
    std::vector<uint> topology = {15, 256, 256, 5};
    Scalar evolutionRate = 0.05f;
    Scalar mutationRate = 0.2f;

    population.reserve(populationSize);  // Reserve space for efficiency

    for (int i = 0; i < populationSize; ++i) {
        neural net(topology, evolutionRate, mutationRate);
        population.push_back(net);
    }

    return population;
}


void generationLearn(float swordA, float speedA, float bodyA, int popSize)
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "Tournament Match");
    window.setFramerateLimit(600);
    int rounds = 0;
    Bot botA = Bot(150.f, 400.f, swordA, speedA, bodyA, false);
    Bot botB = Bot(650.f, 400.f, swordA, speedA, bodyA, true);
    int winner=0;

    std::vector<neural> population = createInitialPopulation(popSize);
    std::random_device rd;
    std::mt19937 rng(rd());

    while (true)
    {
        std::shuffle(population.begin(), population.end(), rng);
        rounds++;
        for(int i=0; i<popSize; i+=2){
            botA = Bot(250.f, 400.f, swordA, speedA, bodyA, false);
            botB = Bot(550.f, 400.f, swordA, speedA, bodyA, true);
            winner=one_round(population[i], population[i+1], botA, botB, rounds, i==0, window);

            if(winner==3){
                if(i==0) std::cout << "DRAW! A score: " << botA.getScore() << " / B score: " << botB.getScore() << std::endl;
                winner = botA.getScore()>=botB.getScore() ? 1 : 2;
            }
            if(winner==1){
                if(i==0) std::cout << "A WINS! Generation: " << rounds << std::endl;
                population[i+1]=population[i].clone();
                population[i+1].updateWeights();
            }
            else if(winner==2){
                if(i==0) std::cout << "B WINS! Generation: " << rounds << std::endl;
                population[i]=population[i+1].clone();
                population[i].updateWeights();
            }
        }
    }
}




int main()
{
    srand(static_cast<unsigned int>(time(0)));
    generationLearn(60.f, 0.12f, 100.f, 128);
                                
    return 0;
}
