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


	_reshapeRows  =  processedOutput.rows();
	_reshapeCols  =  processedOutput.cols();


	std::vector<double> flatedOutput = Utils::FlatMatrix( processedOutput );
	flatedOutput.insert(flatedOutput.begin(), 1.0);


	std::vector<double> predictedOutput = _mlp.Foward( flatedOutput );

	return predictedOutput;
}



std::vector<double> CNN::Backward(std::vector<double>& predictedValues, std::vector<double>& correctValues)
{
	std::vector<double> dLoss_dProcessedOutput  =  _mlp.Backward(predictedValues, correctValues);

	Eigen::MatrixXd dLoss_dProcessedOutput = Utils::ReshapeMatrix(dLoss_dProcessedOutput, _reshapeRows, _reshapeCols);

	for (size_t i = _processingUnits.size()-1; i > 0; i--) {
		dLoss_dProcessedOutput  =  _processingUnits[i]->Backward( dLoss_dProcessedOutput );
	}


	dLoss_dProcessedOutput  =  _processingUnits[i]->Backward(dLoss_dProcessedOutput);

	return Utils::FlatMatrix(dLoss_dProcessedOutput);
}


