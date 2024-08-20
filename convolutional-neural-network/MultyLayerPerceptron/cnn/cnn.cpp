#include "cnn.h"

CNN::CNN()
{
	//_mlp = MLP();
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

	Eigen::MatrixXd dLoss_dOutput = Utils::ReshapeMatrix(dLoss_dProcessedOutput, _reshapeRows, _reshapeCols);

	for (size_t i = _processingUnits.size()-1; i > 0; i--) {
		dLoss_dOutput  =  _processingUnits[i]->Backward( dLoss_dOutput );
	}


	dLoss_dOutput  =  _processingUnits[0]->Backward(dLoss_dOutput);

	return Utils::FlatMatrix(dLoss_dOutput);
}


