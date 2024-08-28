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










void CNN::Training(std::vector<CNN_DATA> trainingDataSet, std::function<void(void)> callback)
{
    size_t epoch = 0;
    while (epoch < _maxEpochs) {

        for (auto& data : trainingDataSet) {

            auto input = data.input;
            auto correctOutput = data.label;

            std::vector<double> predictedOutput = Forward(input);
            Backward(predictedOutput, correctOutput);
        }

        //--- shuffle
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(trainingDataSet.begin(), trainingDataSet.end(), g);

		//--- custom task
        callback();

		//--- decrese learning rate
		double error = _mlp._error / (double)trainingDataSet.size();
		UpdateLearningRate(epoch, error);
		_mlp._error = 0.0;


        epoch++;
    }
}



void CNN::UpdateLearningRate(size_t epoch, double error)
{
	for (auto& processUnit: _processingUnits) {
		processUnit->UpdateLearningRate(epoch, error, _UpdateLeraningRate);
	}

	_mlp.ChangeLearningRate(epoch,error);
}


