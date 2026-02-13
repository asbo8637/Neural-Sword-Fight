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
#include <numeric>
#include <memory>
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
    bSwordStart.x += 1.f;
    aSwordStart.x -= 1.f;

    if (linesIntersect(aSwordStart, aSwordEnd, bSwordStart, bSwordEnd))
    {
        const float knockbackBase = 500.f;
        float Bforce = -knockbackBase * (800 - B.getFootPos().x) / 800.f;
        float Aforce = knockbackBase * A.getFootPos().x / 800.f;
        float momentum_dif = A.get_m_momentum() - B.get_m_momentum();
        A.incrementScore(-momentum_dif);
        B.incrementScore(momentum_dif);
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

// Construct the input for the net controlling botA, given botB as enemy
Eigen::RowVectorXf getInputForBot(const Bot &botA, const Bot &botB, float timer, bool flipScore)
{
    timer /= 500;
    auto botAAlly = botA.getAllyValues(); // std::array<float, 6>
    auto botBEnemy = botB.getAllyValues();

    Eigen::RowVectorXf vecA = arrayToEigen(botAAlly);
    Eigen::RowVectorXf vecB = arrayToEigen(botBEnemy);
    float distance = (botB.getFootPos().x - botA.getFootPos().x) / 800.f;
    distance = std::max(-1.f, std::min(1.f, distance));
    if (flipScore)
        distance = -distance;

    float scoreDif = flipScore ? botB.getScore() - botA.getScore() : botA.getScore() - botB.getScore();
    const float scoreScale = 30.f;
    float scoreNorm = std::tanh(scoreDif / scoreScale);
    const float constantInput = 1.0f;
    Eigen::RowVectorXf inputForNet(26);
    inputForNet << vecA, vecB, timer, distance, scoreNorm, constantInput;
    return inputForNet;
}

void drawWalls(sf::RenderWindow &window)
{
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
    // window.draw(rw);
}

struct RoundResult
{
    int winner;
    bool timedOut;
    int scoreA;
    int scoreB;
};

struct TrainConfig
{
    int populationSize;
    int stepsPerMatch;
    int batchPasses; // Full round-robin passes per generation
    int eliteCount;
    int selectionPool;
    int tournamentSize;
    int renderEvery;
    bool headless;
    Scalar evolutionRate;
    Scalar mutationRate;
    std::vector<uint> topology;
};

RoundResult one_round(neural &net1, neural &net2, Bot &botA, Bot &botB, int maxSteps, bool display, sf::RenderWindow *window, float &playbackDelayMs, bool &renderPaused)
{
    int timer = maxSteps;
    std::vector<Scalar> output;
    std::array<float, 5> controlsA;
    std::array<float, 5> controlsB;
    float playbackConstant = 8.f;
    while (timer > 0)
    {
        timer--;
        if (!botA.isAlive())
        {
            return {2, false, botA.getScore(), botB.getScore()};
        }
        if (!botB.isAlive())
        {
            return {1, false, botA.getScore(), botB.getScore()};
        }
        else
        {
            // Get Input Output BotB.
            Eigen::RowVectorXf inputForNet2 = getInputForBot(botB, botA, timer, true);
            net2.propagateForward(inputForNet2);
            output = net2.getOutput();
            std::copy_n(output.begin(), 5, controlsB.begin());

            // Get Input Output BotA.
            Eigen::RowVectorXf inputForNet1 = getInputForBot(botA, botB, timer, false);
            net1.propagateForward(inputForNet1);
            output = net1.getOutput();
            std::copy_n(output.begin(), 5, controlsA.begin());

            // Update Bots
            botB.updateFromNN(controlsB);
            botA.updateFromNN(controlsA);

            // Win by reaching the enemy side
            const float winLeft = 200.f;
            const float winRight = 600.f;
            if (botA.getFootPos().x >= winRight)
                return {1, false, botA.getScore(), botB.getScore()};
            if (botB.getFootPos().x <= winLeft)
                return {2, false, botA.getScore(), botB.getScore()};

            // Deal with collisions and detect deaths:
            bool wasAliveA = botA.isAlive();
            bool wasAliveB = botB.isAlive();
            handleCollisions(botA, botB);

            // Kill if cross walls
            if (botA.isAlive() && botB.isAlive())
            {
                float xB = botB.getFootPos().x;
                if (xB > 800 || xB < 0)
                    botB.kill();
                float xA = botA.getFootPos().x;
                if (xA > 800 || xA < 0)
                    botA.kill();
            }

            // Draw
            if (display && window)
            {
                bool drawDeadA = wasAliveA && !botA.isAlive();
                bool drawDeadB = wasAliveB && !botB.isAlive();
                const sf::Color deadColor(128, 0, 128);
                window->clear(sf::Color(50, 50, 50));
                botA.draw(*window, drawDeadA, drawDeadA ? &deadColor : nullptr);
                botB.draw(*window, drawDeadB, drawDeadB ? &deadColor : nullptr);
                drawWalls(*window);
                if (!renderPaused)
                    window->display();

                // Poll events
                sf::Event event;
                while (window->pollEvent(event))
                {
                    if (event.type == sf::Event::Closed)
                        window->close();
                    if (event.type == sf::Event::KeyPressed)
                    {
                        if (event.key.code == sf::Keyboard::P)
                        {
                            renderPaused = !renderPaused;
                            if (renderPaused)
                                playbackDelayMs = 0.f;
                            else
                                playbackDelayMs = playbackConstant;
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
        // Timeout: farther to enemy side wins.
        float distA = 800.f - botA.getFootPos().x; // A's enemy side is right.
        float distB = botB.getFootPos().x;         // B's enemy side is left.
        winner = (distB < distA) ? 2 : 1;
    }
    return {winner, true, scoreA, scoreB};
}

std::vector<neural> createInitialPopulation(const TrainConfig &cfg)
{
    std::vector<neural> population;
    const int populationSize = cfg.populationSize;

    population.reserve(populationSize); // Reserve space for efficiency

    for (int i = 0; i < populationSize; ++i)
    {
        neural net(cfg.topology, cfg.evolutionRate, cfg.mutationRate);
        population.push_back(net);
    }

    return population;
}

void generationLearn(float swordA, float speedA, float bodyA, const TrainConfig &cfg)
{
    const float gameWidth = 800.f;
    const float gameHeight = 600.f;
    const unsigned int windowWidth = 1280;
    const unsigned int windowHeight = 720;
    std::unique_ptr<sf::RenderWindow> window;
    float playbackConstant = 8.f;
    float playbackDelayMs = playbackConstant;
    bool renderPaused = false;
    if (!cfg.headless)
    {
        window = std::make_unique<sf::RenderWindow>(sf::VideoMode(windowWidth, windowHeight), "Tournament Match");
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
        window->setView(view);
        window->setFramerateLimit(600);
    }
    int rounds = 0;
    Bot botA = Bot(150.f, 400.f, swordA, speedA, bodyA, false);
    Bot botB = Bot(650.f, 400.f, swordA, speedA, bodyA, true);

    const int popSize = cfg.populationSize;
    if (popSize < 2)
        return;
    if (cfg.topology.size() < 2)
    {
        std::cerr << "TrainConfig.topology must be set in main()." << std::endl;
        return;
    }
    std::vector<neural> population = createInitialPopulation(cfg);
    std::vector<float> fitness(popSize, 0.f);
    std::vector<int> order(popSize);
    std::vector<int> ranked(popSize);
    std::random_device rd;
    std::mt19937 rng(rd());
    const float winBonus = 40.0f;
    const float scoreWeight = 0.1f;
    auto applyMatchFitness = [&](int idxA, int idxB, const RoundResult &result)
    {
        float scale = result.timedOut ? 0.5f : 1.0f;
        float scoreA = scoreWeight * static_cast<float>(result.scoreA);
        float scoreB = scoreWeight * static_cast<float>(result.scoreB);
        float bonusA = (result.winner == 1) ? winBonus : 0.0f;
        float bonusB = (result.winner == 2) ? winBonus : 0.0f;
        fitness[idxA] += scale * (scoreA + bonusA);
        fitness[idxB] += scale * (scoreB + bonusB);
    };

    while (true)
    {
        rounds++;
        std::fill(fitness.begin(), fitness.end(), 0.f);

        const int pairable = popSize - (popSize % 2);
        const int pairCount = pairable / 2;

        for (int pass = 0; pass < std::max(1, cfg.batchPasses); ++pass)
        {
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), rng);

            std::atomic<int> nextPair(0);
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
                workers.emplace_back([&]()
                                     {
                    float dummyDelay = 0.f;
                    bool dummyPaused = false;
                    while (true)
                    {
                        int pairIndex = nextPair.fetch_add(1);
                        if (pairIndex >= pairCount)
                            break;
                        int i = order[pairIndex * 2];
                        int j = order[pairIndex * 2 + 1];
                        Bot localA(100.f, 400.f, swordA, speedA, bodyA, false);
                        Bot localB(600.f, 400.f, swordA, speedA, bodyA, true);
                        RoundResult result = one_round(population[i], population[j], localA, localB, cfg.stepsPerMatch, false, nullptr, dummyDelay, dummyPaused);
                        applyMatchFitness(i, j, result);
                    } });
            }

            // Main thread takes a slice too.
            float dummyDelay = 0.f;
            bool dummyPaused = false;
            while (true)
            {
                int pairIndex = nextPair.fetch_add(1);
                if (pairIndex >= pairCount)
                    break;
                int i = order[pairIndex * 2];
                int j = order[pairIndex * 2 + 1];
                Bot localA(250.f, 400.f, swordA, speedA, bodyA, false);
                Bot localB(550.f, 400.f, swordA, speedA, bodyA, true);
                RoundResult result = one_round(population[i], population[j], localA, localB, cfg.stepsPerMatch, false, nullptr, dummyDelay, dummyPaused);
                applyMatchFitness(i, j, result);
            }

            for (auto &t : workers)
                t.join();
        }

        std::iota(ranked.begin(), ranked.end(), 0);
        std::sort(ranked.begin(), ranked.end(), [&](int a, int b)
                  { return fitness[a] > fitness[b]; });

        float avgFitness = std::accumulate(fitness.begin(), fitness.end(), 0.f) / static_cast<float>(popSize);
        std::cout << "Gen " << rounds
                  << " best=" << fitness[ranked[0]]
                  << " avg=" << avgFitness << std::endl;

        if (!cfg.headless && window && window->isOpen() && (rounds % std::max(1, cfg.renderEvery) == 0))
        {
            botA = Bot(250.f, 400.f, swordA, speedA, bodyA, false);
            botB = Bot(550.f, 400.f, swordA, speedA, bodyA, true);
            const int renderPoolSize = std::min(100, popSize);
            std::uniform_int_distribution<int> renderRankDist(0, renderPoolSize - 1);
            int rankA = renderRankDist(rng);
            int rankB = renderRankDist(rng);
            while (rankB == rankA)
                rankB = renderRankDist(rng);
            int renderA = ranked[rankA];
            int renderB = ranked[rankB];
            one_round(population[renderA], population[renderB], botA, botB, cfg.stepsPerMatch, true, window.get(), playbackDelayMs, renderPaused);
            if (!window->isOpen())
                return;
        }

        int eliteCount = std::min(cfg.eliteCount, popSize);
        int poolCount = std::min(cfg.selectionPool, popSize);
        if (poolCount < eliteCount)
            poolCount = eliteCount;
        int tourSize = std::min(cfg.tournamentSize, poolCount);
        std::uniform_int_distribution<int> poolDist(0, poolCount - 1);

        auto selectParent = [&]() -> int
        {
            int best = ranked[poolDist(rng)];
            for (int t = 1; t < tourSize; ++t)
            {
                int cand = ranked[poolDist(rng)];
                if (fitness[cand] > fitness[best])
                    best = cand;
            }
            return best;
        };

        std::vector<neural> nextGen;
        nextGen.reserve(popSize);
        for (int e = 0; e < eliteCount; ++e)
            nextGen.push_back(population[ranked[e]].clone());

        while (static_cast<int>(nextGen.size()) < popSize)
        {
            int parentIndex = selectParent();
            neural child = population[parentIndex].clone();
            child.updateWeights();
            nextGen.push_back(std::move(child));
        }

        population.swap(nextGen);
    }
}

int main()
{
    srand(static_cast<unsigned int>(time(0)));
    TrainConfig cfg;
    cfg.populationSize = 1000;
    cfg.stepsPerMatch = 500;
    cfg.batchPasses = 3;
    cfg.eliteCount = 20;
    cfg.selectionPool = 240;
    cfg.tournamentSize = 4;
    cfg.evolutionRate = 0.03f;
    cfg.mutationRate = 0.08f;
    cfg.topology = {26, 400, 5};
    cfg.headless = false;
    cfg.renderEvery = 1;


    generationLearn(100.f, 0.08f, 100.f, cfg);

    return 0;
}
