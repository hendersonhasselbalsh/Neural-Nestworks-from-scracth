#include "cnn.h"

CNN::CNN()
{

}


std::vector<double> CNN::Forward(Eigen::MatrixXd& input)
{
	Eigen::MatrixXd processedOutput = input;
	for (auto& processUnit: _processingUnits) {
		processedOutput = processUnit->Forward( processedOutput );
	}

	std::vector<double> flatedOutput = Utils::FlatMatrix( processedOutput );
	std::vector<double> predictedOutput = _mlp.Foward( flatedOutput );

	return predictedOutput;
}


