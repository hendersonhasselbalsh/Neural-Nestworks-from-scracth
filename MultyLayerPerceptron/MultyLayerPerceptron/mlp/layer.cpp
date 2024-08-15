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
	_neurons = std::vector<Neuron>(neuronQuantity, Neuron(inputSize, actFun, neuronLerningRate, _lostFunction));

	size_t layerOutputSize = _neurons.size() + 1;
	_outputs =  std::vector<double>(layerOutputSize, 1.0);

	_accumulatedLayerOutouts =  std::vector<double>(layerOutputSize, 0.0);

	_alpha = 1.0;
	_beta  = 0.0;
}



Layer::~Layer() { }




std::vector<double> Layer::CalculateLayerOutputs(std::vector<double> inputs, std::vector<double>* means, std::vector<double>* devs)
{ 
	int outputIndex = 1;

	double* alpha = nullptr;
	double* beta = nullptr;
 
	if (means!=nullptr && devs!=nullptr) { 
		alpha = &_alpha;
		beta = &_beta;
		Utils::DataNorm(&inputs, means, devs); 
		_receivedInput  =  inputs;
		Utils::ScalateAndShift(&inputs, alpha, beta); 
	}

	_receivedInput  =  inputs;

	for (auto& neuron : _neurons) {
		double output = neuron.CalculateOutput(inputs/*, alpha, beta*/);
		_accumulatedLayerOutouts[outputIndex] += output;							
		_outputs[outputIndex++]  =  output;
	}

	return _outputs;
}



double Layer::GradientAtIndex(int index)
{
	double lostGradWithRespectToInput  =  0.0;
	for (auto& neuron : _neurons) {
		lostGradWithRespectToInput  +=  neuron.Gradient(index);
	}
	return lostGradWithRespectToInput;
}



std::vector<double> Layer::Gradients()
{
	std::vector<double> gradients  =  std::vector<double>( (size_t)_inputSize, 0.0 );

	for (int i = 0; i < _inputSize; i++) {
		gradients[i]  =  GradientAtIndex(i);
	}

	return gradients; 
}



std::vector<double> Layer::UpdateLastLayerNeurons(std::vector<double> correctValues, std::vector<double> predictedValues, std::vector<double> inputs, size_t* batchSize, bool isBatchNorm)
{
	assert(correctValues.size() == _neurons.size() && predictedValues.size() == _neurons.size());

	if (isBatchNorm) { UpdateBatchNormParms(); }

	int index = 0;
	size_t numberOfNeurons  =  _neurons.size();
	std::vector<double> errors  =  std::vector<double>( numberOfNeurons, 1.0 );

	for (auto& neuron : _neurons) {
		double correctValue  =  correctValues[index];
		double predictedValue = predictedValues[index];
		errors[index++]  =  neuron.CalculateError(correctValue,predictedValue, batchSize);  // [DISCOMENT THIS]
		neuron.UpdateWeights( inputs );
	}

	_accumulatedLayerOutouts = std::vector<double>( (size_t)(_neurons.size()+1), 0.0 );
	_accumulatedLayerOutouts[0] = 1.0;

	return errors;
}



void Layer::UpdateHiddenLayerNeurons(std::vector<double> nextLayerGradients, std::vector<double> inputs, size_t* batchSize, bool isBatchNorm)
{
	assert(_neurons.size() == nextLayerGradients.size());

	if (isBatchNorm) { UpdateBatchNormParms(); }

	int index = 0;

	for (auto& neuron : _neurons) {
		double gradient = neuron.CalculateGradient(nextLayerGradients[index++], batchSize);         // calcula  (dL/dU) = (dO/dU) * (dL/dO)
		neuron.UpdateWeights( inputs );													            // calcula  (dL/dW) = (dU/dW) * (dL/dU) = (dU/dW) * (dO/dU) * (dL/dO) 
	}

	_accumulatedLayerOutouts = std::vector<double>( (size_t)(_neurons.size()+1), 0.0 );
	_accumulatedLayerOutouts[0] = 1.0;
}



std::vector<double> Layer::CalculateAccumulatedError(std::vector<double> correctValues, std::vector<double> predictedValue)
{
	size_t index = 0;
	std::vector<double> neuronsGradients = std::vector<double>(_neurons.size(), 0.0);

	for (auto& neuron : _neurons) {
		neuron.CalculateError( correctValues[index], predictedValue[index] );
		double gradient  =  neuron.Get<Neuron::Attribute::GRADIENT_DL_DU>();
		neuronsGradients[index++]  =  gradient;
	}

	return neuronsGradients;
}



void Layer::UpdateBatchNormParms()
{
	double lostGradWithRespectToAlpha  =  0.0;
	double lostGradWithRespectToBeta   =  0.0;

	for (size_t i = 0; i < _neurons.size(); i++) {
		double neuronGrad  =  _neurons[i].Gradient(i);
		lostGradWithRespectToAlpha  +=  neuronGrad * _receivedInput[i];    
		lostGradWithRespectToBeta   +=  neuronGrad;
	}

	_alpha  =  _alpha - _neuronLerningRate * lostGradWithRespectToAlpha;
	_beta   =  _beta  - _neuronLerningRate * lostGradWithRespectToBeta;

	
	//double ALPHA_TRESHOLD = 5.8;  // [-1.8; 1.8]   // parameter clipping (not needed now, but who knows ... )
	//double BETA_TRESHOLD = 1.8;

	//if (std::abs(_alpha) > ALPHA_TRESHOLD) {  _alpha =  ALPHA_TRESHOLD * (_alpha/std::abs(_alpha)); }
	//if (std::abs(_beta) > BETA_TRESHOLD) {  _beta =  BETA_TRESHOLD * (_beta/std::abs(_beta)); }
}


Neuron& Layer::operator[](size_t neuronIndex)
{
	return (Neuron&) _neurons[neuronIndex];
}



std::vector<double> Layer::MeanAccumulatedOutput(double batchSize)
{
	std::vector<double> meanOutput = _accumulatedLayerOutouts;
	for (auto& out : meanOutput) { out = out / batchSize; }

	_accumulatedLayerOutouts = std::vector<double>( (size_t)(_neurons.size()+1), 0.0 );
	_accumulatedLayerOutouts[0] = 1.0;

	return meanOutput;
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


