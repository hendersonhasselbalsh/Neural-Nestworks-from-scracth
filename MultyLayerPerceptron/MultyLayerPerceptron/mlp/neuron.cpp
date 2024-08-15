#include "neuron.h"


template const auto Neuron::Get<Neuron::Attribute::ACTIVATION_FUNC>() const;
template const auto Neuron::Get<Neuron::Attribute::BIAS>() const;
template const auto Neuron::Get<Neuron::Attribute::ERROR>() const;
template const auto Neuron::Get<Neuron::Attribute::GRADIENT_DL_DU>() const;
template const auto Neuron::Get<Neuron::Attribute::LOST_FUNC>() const;
template const auto Neuron::Get<Neuron::Attribute::OUTPUT>() const;
template const auto Neuron::Get<Neuron::Attribute::U>() const;
template const auto Neuron::Get<Neuron::Attribute::WEIGHTS>() const;

template void Neuron::Set<Neuron::Attribute::ACTIVATION_FUNC, IActivationFunction*>(IActivationFunction* value);
template void Neuron::Set<Neuron::Attribute::LOST_FUNC, ILostFunction*>(ILostFunction* value);
template void Neuron::Set<Neuron::Attribute::LEARNING_RATE, double>(double value);




Neuron::Neuron(size_t inputSize, IActivationFunction* actFun, double leraningRate, ILostFunction* lostFunc)
    : _inputSize(inputSize+1), _learningRate(leraningRate), activationFunction(actFun), _lostFunction(lostFunc)
{
    _error = 0.0;
    _output = 0.0;
    _u = 0.0;
    _gradient = 0.0;

    _accumulatedU = 0.0;

    _weights = std::vector<double>(_inputSize, 1.0);

    for (auto& weight : _weights) {
        weight  =  Utils::RandomNormalDistributionValue(-0.3, 0.3);
    }
}



Neuron::~Neuron() { }



double Neuron::CalculateOutput(std::vector<double> inputs)
{
    /*if (alpha!=nullptr && beta!=nullptr) { Utils::ScalateAndShift(&inputs, alpha, beta); }*/

    _u = Utils::ScalarProduct(inputs, _weights);
    _output = activationFunction->f(_u);

    _accumulatedU += _u;

    return _output;
}


double Neuron::CalculateError(double correctValue, double predictedValue, size_t* batchSize)
{
    if (batchSize != nullptr) {  _u = _accumulatedU / (double)(*batchSize); }

    double du = activationFunction->df(_u);                                                                         // (dO/dU)  =  df( U ) 
    _gradient = du * _lostFunction->df(predictedValue, correctValue);                                               // (dL/dU) dervate of lost with respect to U dC/dU

    _error = _lostFunction->f(predictedValue, correctValue);

    return _error;
}


double Neuron::CalculateGradient(double gradientLostWithRespectToOutput, size_t* batchSize)
{
    if (batchSize != nullptr) {  _u = _accumulatedU / (double)(*batchSize); }

    double du = activationFunction->df(_u);
    _gradient  =  du * gradientLostWithRespectToOutput;                                                        // (dL/dU) dervate of lost with respect to U dC/dU  for hiden layers

    return _gradient;
}


void Neuron::UpdateWeights(std::vector<double> receivedInputs)
{
    assert(receivedInputs.size() == _weights.size());

    for (size_t i = 0; i < _weights.size(); i++) {
        double gradOfLostWithRespectToWeight = _gradient * receivedInputs[i];                                 // (dL/dW) derivation of lost with respect to this layer weight
        _weights[i]  =  _weights[i] - _learningRate * gradOfLostWithRespectToWeight; 
        
        if (std::isnan(_weights[i])) {
            int DEBUG = 0;
        }
    
    }

    _accumulatedU = 0.0;
}


const double Neuron::Gradient(size_t index)
{
    return _weights[index] * _gradient;
}



double& Neuron::operator[](int weightIndex)
{
    return (double&)_weights[weightIndex];
}



std::ostream& operator<<(std::ostream& os, Neuron neuron)
{
    os << "[ ";
    for (auto& w : neuron._weights) { os << w << " ; "; }
    os << "\b\b ]";

    return os;
}



Neuron Neuron::operator=(const Neuron& neuron)
{
    if (this != &neuron) {
        _weights = neuron._weights;
        activationFunction = neuron.activationFunction;
        _inputSize = neuron._inputSize;
        _error = neuron._error;
        _output = neuron._output;
        _u = neuron._u;
        _gradient = neuron._gradient;
        _learningRate = neuron._learningRate;
    }
    return *this;
}



Json Neuron::ToJson() const
{
    Json json ={
        {"bias", _weights[0] },
        {"weights", _weights}
    };

    return json;
}



std::vector<double> Neuron::LoadWeightsFromJson(const Json& j)
{
    (*this)._weights = j.at("weights").get<std::vector<double>>();
    return (*this)._weights;
}

