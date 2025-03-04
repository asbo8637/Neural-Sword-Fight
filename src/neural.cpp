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