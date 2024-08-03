#pragma once

#include "layer.h"


template const auto Layer::Get<Layer::Attribute::ACCUMULATED_OUTPUTS>() const;
template const auto Layer::Get<Layer::Attribute::ALL_NEURONS>() const;
template const auto Layer::Get<Layer::Attribute::ALL_NEURONS_GRADIENTS>() const;
template const auto Layer::Get<Layer::Attribute::INPUT_SIZE>() const;
template const auto Layer::Get<Layer::Attribute::LAYER_ERRORS>() const;
template const auto Layer::Get<Layer::Attribute::LAYER_OUTPUTS>() const;
template const auto Layer::Get<Layer::Attribute::NUMBER_OF_NEURONS>() const;
template const auto Layer::Get<Layer::Attribute::OUTPUT_SIZE>() const;
template const auto Layer::Get<Layer::Attribute::RECEIVED_INPUT>() const;

template void Layer::Set<Layer::Attribute::LEARNING_RATE, double>(double value);




Layer::Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun, double neuronLerningRate, ILostFunction* _lostFunction)
	: _inputSize(inputSize), _activationFunction(actFun), _neuronLerningRate(neuronLerningRate)
{
	size_t outputSize  =  _neurons.size() + 1;
	_neurons = std::vector<Neuron>(neuronQuantity, Neuron(inputSize, actFun, neuronLerningRate, _lostFunction, outputSize));

	size_t layerOutputSize = _neurons.size() + 1;
	_outputs =  std::vector<double>(layerOutputSize, 1.0);
}



Layer::~Layer() { }




std::vector<double> Layer::CalculateLayerOutputs(std::vector<double> inputs)
{ 
	int outputIndex = 1;

	_receivedInput  =  inputs;

	for (auto& neuron : _neurons) {
		double output = neuron.CalculateOutput( inputs );						
		_outputs[outputIndex++]  =  output;
	}

	return _outputs;
}



void Layer::LastLayerLossGradient(std::vector<double> correctValues, std::vector<double> predictedValues)
{
	assert(correctValues.size() == _neurons.size() && predictedValues.size() == _neurons.size());

	int index = 0;

	for (auto& neuron : _neurons) {
		double correctValue  =  correctValues[index];
		double predictedValue = predictedValues[index++];

		neuron.CalculateLossGradient(correctValue, predictedValue, _receivedInput);
		neuron.UpdateWeights();
	}
}


void Layer::HiddenLayerLossGradient(std::vector<double> lossPartialWithRespectToActivation)
{
	assert(_neurons.size() == lossPartialWithRespectToActivation.size());

	int index = 0;

	for (auto& neuron : _neurons) {
		neuron.CalculateLossGradient(lossPartialWithRespectToActivation[index++], _receivedInput);
		neuron.UpdateWeights();									
	}
}


std::vector<double> Layer::LossPartialWithRespectToInput()
{
	std::vector<double> gradients  =  std::vector<double>( (size_t)_inputSize, 0.0 );

	for (int inputIndex = 0; inputIndex < _inputSize; inputIndex++) {
		for (auto& neuron : _neurons) {
			gradients[inputIndex]  +=  neuron.LossPartialWithRespectToInput(inputIndex);
		}
	}

	return gradients; 
}


void Layer::UpdateAllNeurons()
{
	for (auto& neuron : _neurons) {
		neuron.UpdateWeights();												
	}
}


Neuron& Layer::operator[](size_t neuronIndex)
{
	return (Neuron&) _neurons[neuronIndex];
}


std::ostream& operator<<(std::ostream& os, Layer layer)
{
	for (auto& n : layer._neurons) { os << n << "  "; }
	std::cout << "\n";
	return os;
}



Json Layer::ToJson() const
{
	Json layerJson;
	layerJson["inputSize"]  =  _inputSize;
	layerJson["actFunc"]  =  _activationFunction->ToString();
	layerJson["learningRate"]   =  _neuronLerningRate;
	for (const auto& neuron : _neurons) {  layerJson["neurons"].push_back( neuron.ToJson() );  }
	return { {"layer", layerJson} };
}



Layer Layer::LoadWeightsFromJson(const Json& j)
{
	_activationFunction  =  Utils::StringToActivationFunction( j.at("layer").at("actFunc").get<std::string>() );
	_neuronLerningRate  =  j.at("layer").at("learningRate").get<double>();
	_inputSize  =  j.at("layer").at("inputSize").get<size_t>();

	int neuronIndex = 0;
	for (const auto& neuronJson : j.at("layer").at("neurons")) {
		Neuron n = Neuron(_inputSize, _activationFunction, _neuronLerningRate);
		n.LoadWeightsFromJson(neuronJson);

		(*this)._neurons[neuronIndex] = n;

		neuronIndex++;
	}

	return (*this);
}


