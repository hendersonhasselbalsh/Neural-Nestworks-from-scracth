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


std::vector<std::pair<Eigen::MatrixXd, size_t>> DataManager::ExtractVectors(Eigen::MatrixXd& inputVecs)
{
    std::vector<std::pair<Eigen::MatrixXd, size_t>> individualVec;

    for (size_t vec = 0; vec < inputVecs.cols(); vec++) {
        Eigen::MatrixXd vector = inputVecs.col(vec);

        individualVec.push_back({ vector, vec });
    }

    return individualVec;
}


void DataManager::Shuffle(std::vector<std::pair<Eigen::MatrixXd, size_t>>* data)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle( (*data).begin(), (*data).end(), g);
}


std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> DataManager::BuildBatch(std::vector<std::pair<Eigen::MatrixXd, size_t>>& datas, long batchSize, size_t classes)
{
    DataManager::Shuffle( &datas );

    long start = 0;
    long end = batchSize;
    long dataSize = datas.size();
    size_t vectorConponent = datas[0].first.rows();

    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> batchs;

    while (start < dataSize) {
        size_t vectorQnt = std::min(batchSize, dataSize-start); 

        end = start + vectorQnt;

        Eigen::MatrixXd batchInput = Eigen::MatrixXd(vectorConponent, vectorQnt);
        Eigen::MatrixXd batchCorrectY = Eigen::MatrixXd::Zero(classes, vectorQnt);

        for (size_t i = 0; i < vectorQnt; i++) {
            auto& data = datas[start+i];
            batchInput.col(i) = data.first;
            batchCorrectY(data.second,i) = 1.0;
        }

        batchs.push_back( { batchInput, batchCorrectY } );

        start = end;
    }

    return batchs;
}



