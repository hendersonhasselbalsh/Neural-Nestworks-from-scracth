#include "DataManager.h"

DataManager::DataManager()
{
}


double DataManager::RandomNormalDistributionValue(double min, double max)
{
    // Create a random number generator
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Define mean and standard deviation for the normal distribution
    double mean = (min + max) / 2.0;
    double stddev = (max - min) / 6.0; // approximately 99.7% within [a, b]

    // Create a normal distribution
    std::normal_distribution<double> d(mean, stddev);

    // Generate random numbers within the range [a, b]
    double number = 0.0;
    do {
        number = d(gen);
    } while (number < min || number > max);

    return number;
}


double DataManager::RandomUniformDistribution(double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribuicao(min, max);
    return distribuicao(gen);
}


void DataManager::XaviverInitialization(Eigen::MatrixXd& weights, size_t inputSize, size_t outputSize)
{
	double n_i = (double)inputSize;
	double n_o = (double)outputSize;

	double range = std::sqrt(2.0 / (n_i + n_o));

	for (size_t neuronIndex = 0; neuronIndex < weights.rows(); neuronIndex++) {  
		for (size_t weightIndex = 1; weightIndex < weights.cols(); weightIndex++) { 
			double weight  =  DataManager::RandomNormalDistributionValue(-1.0, 1.0) * range;
			weights(neuronIndex, weightIndex) = weight;
		}
	}
}


std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> DataManager::ExtractCorrespondingVectors(Eigen::MatrixXd& dLdU_vecs, Eigen::MatrixXd& inputVecs)
{
    assert(dLdU_vecs.cols() == inputVecs.cols() && "they must have same qnt of vectors");


    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> correspondinVec;

    for (size_t vec = 0; vec < inputVecs.cols(); vec++) {
        Eigen::MatrixXd dLdU_vec = dLdU_vecs.col(vec);
        Eigen::MatrixXd dLdW_vec = inputVecs.col(vec);

        correspondinVec.push_back( {dLdU_vec, dLdW_vec} );
    }

    return correspondinVec;
}


std::vector<Eigen::MatrixXd> DataManager::ExtractVectors(Eigen::MatrixXd& inputVecs)
{
    std::vector<Eigen::MatrixXd> individualVec;

    for (size_t vec = 0; vec < inputVecs.cols(); vec++) {
        Eigen::MatrixXd vector = inputVecs.col(vec);

        individualVec.push_back(vector);
    }

    return individualVec;
}


Eigen::MatrixXd DataManager::PrepareVectorAsDenseLayerInput(Eigen::MatrixXd& vec)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(vec.rows()+1, vec.cols());
    input.block(1, 0, vec.rows(), vec.cols()) = vec;

    return input;
}