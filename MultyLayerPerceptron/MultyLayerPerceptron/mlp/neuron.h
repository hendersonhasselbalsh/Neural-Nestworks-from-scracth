#pragma once

#include <nlohmann/json.hpp>
#include "../utils/basic-includes.h"
#include "activation-functions.h"
#include "lost-function.h"
#include "../utils/utils.h"

using Json = nlohmann::json;


class Neuron {

	private:
	//--- atributos fundamentais do meuronios
		double _learningRate;
		std::vector<double> _weights;
		IActivationFunction* activationFunction;
		ILostFunction* _lostFunction;
		double _gradient;												// gradient of lost With Respect To U (dL/dU) = (dO/dU) * (dL/dO)


	//--- variaveis de armazenamento
		size_t _inputSize;                                              // tamanho do vetor de entrada
		double _error;	                                                // erro do do valor de saida
		double _output;	                                                // valor de saida
		double _u;		                                                // \sun( x_i * w_i ) + bias

		double _accumulatedU;

		

	public:
	//--- construtor
		Neuron(size_t inputSize, IActivationFunction* actFun = new Sigmoid(), double leraningRate = 0.01, ILostFunction* lostFunc = nullptr);
		~Neuron();


	//--- pricipais methods
		double CalculateOutput(std::vector<double> inputs);
		double CalculateError(double correctValue, double predictedValue, size_t* batchSize = nullptr);
		double CalculateGradient(double gradientLostWithRespectToOutput, size_t* batchSize = nullptr);
		void UpdateWeights(std::vector<double> receivedInputs);
		const double Gradient(size_t index);


	//--- get e sets
		double& operator[](int weightIndex);
		friend std::ostream& operator<<(std::ostream& os, Neuron neuron);
		Neuron operator=(const Neuron& neuron);

		Json ToJson() const;
		std::vector<double> LoadWeightsFromJson(const Json& j);


		enum class Attribute { 
			BIAS, ERROR, U, GRADIENT_DL_DU, LEARNING_RATE,      // double types
			WEIGHTS, OUTPUT, 									// std::vector<double> type
			ACTIVATION_FUNC,									// IActivationFunction* TYPE
			LOST_FUNC											// ILostFunction* 
		};
		template <Attribute attrib> const auto Get() const;
		template <Attribute attrib, typename T> void Set(T value);

};




template<Neuron::Attribute attrib>
const auto Neuron::Get() const
{
	if constexpr (attrib == Neuron::Attribute::ACTIVATION_FUNC) {
		return activationFunction;
	}
	else if constexpr (attrib == Neuron::Attribute::LOST_FUNC) {
		return _lostFunction;
	}
	else if constexpr (attrib == Attribute::WEIGHTS) {
		return _weights;
	} 
	else if constexpr (attrib == Attribute::BIAS) {
		return _weights[0];
	} 
	else if constexpr (attrib == Attribute::ERROR) {
		return _error;
	} 
	else if constexpr (attrib == Attribute::U) {
		return _u;
	} 
	else if constexpr (attrib == Attribute::OUTPUT) {
		return _output;
	} 
	else if constexpr (attrib == Attribute::GRADIENT_DL_DU) {
		return _gradient;
	} 
	else {
		assert(false  &&  "Unsupported attribute");
	}
}



template<Neuron::Attribute attrib, typename T>
void Neuron::Set(T value)
{
	if constexpr (attrib == Neuron::Attribute::ACTIVATION_FUNC) {
		static_assert( std::is_same_v<T, IActivationFunction*>  &&  "wrong type");
		activationFunction  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::LOST_FUNC) {
		static_assert(std::is_same_v<T, ILostFunction*>  &&  "wrong type");
		_lostFunction  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::LEARNING_RATE) {
		static_assert(std::is_same_v<T, double>  &&  "wrong type");
		_learningRate  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::GRADIENT_DL_DU) {
		static_assert(std::is_same_v<T, double>  &&  "wrong type");
		_gradient  =  value;
	}
	else {
		assert(false && "Not settable attribute");
	}
}

