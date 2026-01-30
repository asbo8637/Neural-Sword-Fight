#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <Eigen/dense>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <atomic>
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
        const float knockbackBase = 300.f;
        float Bforce = -knockbackBase * (800 - B.getFootPos().x) / 800.f;
        float Aforce = knockbackBase * A.getFootPos().x / 800.f;
        if(A.get_m_momentum()==B.get_m_momentum())
        {
            A.incrementScore();
            B.incrementScore();
        }
        else if(A.get_m_momentum()>B.get_m_momentum()){
            A.incrementScore();
        }
        else{
            B.incrementScore();
        }
        B.applyKnockback(Bforce);
        A.applyKnockback(Aforce);
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
Eigen::RowVectorXf getInputForBot(Bot botA, Bot botB, float timer, bool flipScore)
{
    timer /= 500;
    auto botAAlly = botA.getAllyValues();   // std::array<float, 6>
    auto botBEnemy = botB.getAllyValues();

    Eigen::RowVectorXf vecA = arrayToEigen(botAAlly);
    Eigen::RowVectorXf vecB = arrayToEigen(botBEnemy);
    float distance = (botB.getFootPos().x - botA.getFootPos().x) / 800.f;
    distance = std::max(-1.f, std::min(1.f, distance));

    float scoreDif = flipScore ? botB.getScore() - botA.getScore() : botA.getScore() - botB.getScore();
    const float scoreScale = 10.f;
    float scoreNorm = std::tanh(scoreDif / scoreScale);
    Eigen::RowVectorXf inputForNet(15);
    inputForNet << vecA, vecB, timer, distance, scoreNorm;
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

struct RoundResult
{
    int winner;
    bool timedOut;
    int scoreA;
    int scoreB;
};

RoundResult one_round(neural net1, neural net2, Bot &botA, Bot &botB, int rounds, bool display, sf::RenderWindow &window, float &playbackDelayMs, bool &renderPaused){
    int timer = 500;
    std::vector<Scalar> output;
    std::array<float, 5> controlsA;
    std::array<float, 5> controlsB;
    while(timer>0){
        timer--;
        if(!botA.isAlive()){
            return {2, false, botA.getScore(), botB.getScore()};
        }
        if(!botB.isAlive()){
            return {1, false, botA.getScore(), botB.getScore()};
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

            // Win by reaching the enemy side
            const float winLeft = 200.f;
            const float winRight = 600.f;
            if (botA.getFootPos().x >= winRight)
                return {1, false, botA.getScore(), botB.getScore()};
            if (botB.getFootPos().x <= winLeft)
                return {2, false, botA.getScore(), botB.getScore()};


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
                if (!renderPaused)
                    window.display();

                // Poll events
                sf::Event event;
                while (window.pollEvent(event))
                {
                    if (event.type == sf::Event::Closed)
                        window.close();
                    if (event.type == sf::Event::KeyPressed)
                    {
                        if (event.key.code == sf::Keyboard::P)
                        {
                            renderPaused = !renderPaused;
                            if (renderPaused)
                                playbackDelayMs = 0.f;
                        }
                        if (event.key.code == sf::Keyboard::Add || event.key.code == sf::Keyboard::Equal)
                        {
                            playbackDelayMs = std::max(0.f, playbackDelayMs - 2.f);
                        }
                        else if (event.key.code == sf::Keyboard::Subtract || event.key.code == sf::Keyboard::Hyphen)
                        {
                            playbackDelayMs = std::min(200.f, playbackDelayMs + 2.f);
                        }
                    }
                }
                if (playbackDelayMs > 0.f)
                    sf::sleep(sf::milliseconds(static_cast<int>(playbackDelayMs)));
            }
        }
    }
    int scoreA = botA.getScore();
    int scoreB = botB.getScore();
    int winner = 0;
    if (scoreA > scoreB)
    {
        winner = 1;
    }
    else if (scoreB > scoreA)
    {
        winner = 2;
    }
    else
    {
        // Tie-breaker: closer to enemy side wins.
        float distA = 800.f - botA.getFootPos().x; // A's enemy side is right.
        float distB = botB.getFootPos().x;        // B's enemy side is left.
        if (distA < distB)
            winner = 1;
        else if (distB < distA)
            winner = 2;
        else
            winner = 1;
    }
    return {winner, true, scoreA, scoreB};
}

std::vector<neural> createInitialPopulation(int populationSize) {
    std::vector<neural> population;
    std::vector<uint> topology = {15, 256, 128, 64, 16, 5};
    Scalar evolutionRate = 0.03f;
    Scalar mutationRate = 0.06f;

    population.reserve(populationSize);  // Reserve space for efficiency

    for (int i = 0; i < populationSize; ++i) {
        neural net(topology, evolutionRate, mutationRate);
        population.push_back(net);
    }

    return population;
}


void generationLearn(float swordA, float speedA, float bodyA, int popSize)
{
    const float gameWidth = 800.f;
    const float gameHeight = 600.f;
    const unsigned int windowWidth = 1280;
    const unsigned int windowHeight = 720;
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Tournament Match");
    {
        sf::View view(sf::FloatRect(0.f, 0.f, gameWidth, gameHeight));
        float windowRatio = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);
        float viewRatio = gameWidth / gameHeight;
        if (windowRatio > viewRatio)
        {
            float width = viewRatio / windowRatio;
            float left = (1.f - width) * 0.5f;
            view.setViewport(sf::FloatRect(left, 0.f, width, 1.f));
        }
        else
        {
            float height = windowRatio / viewRatio;
            float top = (1.f - height) * 0.5f;
            view.setViewport(sf::FloatRect(0.f, top, 1.f, height));
        }
        window.setView(view);
    }
    window.setFramerateLimit(600);
    int rounds = 0;
    float playbackDelayMs = 2.f;
    bool renderPaused = false;
    Bot botA = Bot(150.f, 400.f, swordA, speedA, bodyA, false);
    Bot botB = Bot(650.f, 400.f, swordA, speedA, bodyA, true);
    int winner=0;

    std::vector<neural> population = createInitialPopulation(popSize);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> upsetDist(0.f, 1.f);

    while (true)
    {
        rounds++;
        if(rounds%10==0){
            std::shuffle(population.begin(), population.end(), rng);
        }
        else{
            for (int base = 0; base < popSize; base += popSize/4)
            {
                int end = std::min(base + popSize/4, popSize);
                std::shuffle(population.begin() + base, population.begin() + end, rng);
            }
        }

        const int pairCount = popSize / 2;
        std::vector<RoundResult> results(pairCount);
        std::atomic<int> nextPair(1);

        unsigned int hwThreads = std::thread::hardware_concurrency();
        int workerCount = 0;
        if (pairCount > 1)
        {
            if (hwThreads == 0)
                workerCount = 1;
            else
                workerCount = static_cast<int>(std::max(1u, hwThreads - 1));
            workerCount = std::min(workerCount, pairCount - 1);
        }

        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (int w = 0; w < workerCount; ++w)
        {
            workers.emplace_back([&](){
                float dummyDelay = 0.f;
                bool dummyPaused = false;
                while (true)
                {
                    int pairIndex = nextPair.fetch_add(1);
                    if (pairIndex >= pairCount)
                        break;
                    int i = pairIndex * 2;
                    Bot localA(250.f, 400.f, swordA, speedA, bodyA, false);
                    Bot localB(550.f, 400.f, swordA, speedA, bodyA, true);
                    results[pairIndex] = one_round(population[i], population[i+1], localA, localB, rounds, false, window, dummyDelay, dummyPaused);
                }
            });
        }

        // Run the displayed match on the main thread.
        botA = Bot(250.f, 400.f, swordA, speedA, bodyA, false);
        botB = Bot(550.f, 400.f, swordA, speedA, bodyA, true);
        results[0] = one_round(population[0], population[1], botA, botB, rounds, true, window, playbackDelayMs, renderPaused);

        for (auto &t : workers)
            t.join();

        for (int pairIndex = 0; pairIndex < pairCount; ++pairIndex)
        {
            int i = pairIndex * 2;

            winner = results[pairIndex].winner;
            if (winner == 1 || winner == 2)
            {
                if (upsetDist(rng) < 0.01f)
                    winner = (winner == 1) ? 2 : 1;
            }
            if (pairIndex == 0)
            {
                if (results[pairIndex].timedOut)
                {
                    std::cout << "DRAW! A score: " << results[pairIndex].scoreA
                              << " / B score: " << results[pairIndex].scoreB << std::endl;
                }
                if (winner == 1) std::cout << "A WINS! Generation: " << rounds << std::endl;
                else if (winner == 2) std::cout << "B WINS! Generation: " << rounds << std::endl;
            }

            if (winner == 1)
            {
                population[i+1] = population[i].clone();
                population[i+1].updateWeights();
            }
            else if (winner == 2)
            {
                population[i] = population[i+1].clone();
                population[i].updateWeights();
            }
        }
    }
}




int main()
{  
    srand(static_cast<unsigned int>(time(0)));
    generationLearn(60.f, 0.14f, 100.f, 800);
                                
    return 0;
}
