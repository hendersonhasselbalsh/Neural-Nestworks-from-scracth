#pragma once


#include "multy-layer-perceptron.h"


MLP::MLP()
{
}

MLP::~MLP()
{
}



std::vector<double> MLP::Foward(std::vector<double> input)
{
	std::vector<double> layerOutput  =  input;

	for (auto& layer : _layers) {
		layerOutput  =  layer.CalculateLayerOutputs( layerOutput );
	}

	return std::vector<double>( layerOutput.begin()+1, layerOutput.end() );          // return the predicted output
}



std::vector<double> MLP::Backward(std::vector<double> correctValues, std::vector<double> predictedValues)
{
	int layerIndex  =  _layers.size() - 1;


	//--- update last layer
	_layers[layerIndex--].LastLayerLossGradient(correctValues, predictedValues);


	// update hidden layers
	for (layerIndex; layerIndex >= 0; layerIndex--) {
		std::vector<double> gradients  =  _layers[layerIndex+1].LossPartialWithRespectToInput();
		_layers[layerIndex].HiddenLayerLossGradient( gradients );
	}

	return  _layers[0].LossPartialWithRespectToInput();
}


std::vector<double> MLP::Backward(std::vector<double> lossGradientWithRespectToOutput)
{
	int layerIndex  =  _layers.size() - 1;


	//--- update last layer
	_layers[layerIndex--].HiddenLayerLossGradient(lossGradientWithRespectToOutput);


	// update hidden layers
	for (layerIndex; layerIndex >= 0; layerIndex--) {
		std::vector<double> gradients  =  _layers[layerIndex+1].LossPartialWithRespectToInput();
		_layers[layerIndex].HiddenLayerLossGradient( gradients );
	}

	return  _layers[0].LossPartialWithRespectToInput();
}



void MLP::Training(std::vector<TrainigData> trainigSet, std::function<void(void)> callback)
{
	bool keepGoing  =  true;
	size_t epoch  =  0;
	size_t trainingSetSize  =  trainigSet.size();

	while (keepGoing) {

		for (size_t i = 0; i < trainingSetSize; i++) {
			std::vector<double> input  =  trainigSet[i].INPUT;
			std::vector<double> label  =  trainigSet[i].LABEL;

			input.insert(input.begin(), 1.0);

			std::vector<double> predictedOutput  =  Foward( input );
			Backward(label, predictedOutput);
		}

		callback();

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(trainigSet.begin(), trainigSet.end(), g);

		ChangeLearningRate(epoch, 0.0);

		epoch++;
		if (epoch > _maxEpochs) {  keepGoing = false;  }
	}

}


void MLP::Training(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback)
{
	std::vector<TrainigData> _trainingSet;

	for (auto data : trainigSet) {
		std::vector<double> label = ParseLabelToVector( data.labelIndex );
		_trainingSet.push_back({ data.input, label });
	}

	std::cout << "\nstart training:\n\n";
	Training(_trainingSet, callback);

	BuildJson();
}



std::vector<double> MLP::Classify(std::vector<double> input)
{
	input.insert(input.begin(), 1.0);
	return Foward(input);
}

size_t MLP::Classify(std::vector<double> input, std::function<size_t(std::vector<double>)> ParseOutputToLabel)
{
	input.insert(input.begin(), 1.0);
	std::vector<double> givemOutput  =  Foward( input );
	size_t label  =  ParseOutputToLabel( givemOutput );
	return label;
}

void MLP::Classify(std::vector<std::vector<double>> inputs, std::function<void(std::vector<double>)> CallBack)
{
	for (auto& input : inputs) {
		input.insert(input.begin(), 1.0);
		std::vector<double> givemOutput  =  Foward(input);
		CallBack(givemOutput);
	}
}

void MLP::Classify(std::vector<MLP_DATA> inputSet, std::function<void(std::vector<double>)> CallBack)
{
	for (auto& inputData : inputSet) {
		inputData.input.insert(inputData.input.begin(), 1.0);
		std::vector<double> givemOutput  =  Foward(inputData.input);
		CallBack(givemOutput);
	}
}



void MLP::ChangeLearningRate(size_t epoch, double accuracy)
{
	if (WhenToUpdateLeraningRate(epoch, accuracy)) {
		for (auto& layer : _layers) {
			double newRate  =  HowToUpdateLeraningRate(epoch, accuracy, layer._neuronLerningRate);
			for (auto& neuron : layer._neurons) {
				neuron.Set<Neuron::Attribute::LEARNING_RATE, double>(newRate);
			}
		}
	}
}



Layer& MLP::operator[](size_t layerIndex)
{
	return _layers[layerIndex];
}

Layer& MLP::LastLayer()
{
	size_t lastLayerIndex = _layers.size() - 1;
	return _layers[lastLayerIndex];
}



Json MLP::ToJson() const
{
	Json mlpJson;

	mlpJson["inputSize"] = _inputSize;

	for (auto& layer : _layers) {
		mlpJson["layers"].push_back(layer.ToJson());
	}

	return mlpJson;
	
}


void MLP::BuildJson()
{
	if (_outFile != "") {
		Json json = ToJson();

		std::ofstream arquivoSaida(_outFile);


		if (arquivoSaida.is_open()) {
			arquivoSaida << json.dump(4);
			arquivoSaida.close();
		} else {
			std::cerr << "\n\n[ERROR]: could not open file !!! \n\n";
		}
	}
}


