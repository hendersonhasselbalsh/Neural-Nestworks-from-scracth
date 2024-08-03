

#include "lstm-builder.h"

LSTMbuilder::LSTMbuilder()
{
	_lstm  =  LSTM();
}

LSTMbuilder::~LSTMbuilder()
{
}



LSTM LSTMbuilder::Build()
{
	size_t size = _forgetArchitecture[_forgetArchitecture.size()-1]._qntNeurons;


	_lstm._cellStateSaze  =  size;
	_lstm._hiddenStateSize = size;

	_lstm._cellState  =  std::vector<double>(size, 0.0);
	_lstm._previousCellState  =  std::vector<double>(size, 1.0);
	_lstm._hiddenState = std::vector<double>(size, 0.0);


	size_t _gateInputSize  =  _lstm._hiddenStateSize + _lstm._inputSize;


	// initialising forget MLP
	_lstm._forgetMLP  =  MlpBuilder()
		.InputSize( _gateInputSize )
		.Architecture( _forgetArchitecture )
		.Build();


	// initialising input MLP
	_lstm._inputMLP  =  MlpBuilder()
		.InputSize( _gateInputSize )
		.Architecture( _inputArchitecture )
		.Build();


	// initialising candidate MLP
	_lstm._candidateMLP  =  MlpBuilder()
		.InputSize(_gateInputSize)
		.Architecture( _candidateArchitecture )
		.Build();


	// initialising output MLP
	_lstm._outputMLP  =  MlpBuilder()
		.InputSize(_gateInputSize)
		.Architecture( _outputArchitecture )
		.Build();


	// initialising linear MLP
	_lstm._linearMLP  =  MlpBuilder()
		.InputSize(_lstm._hiddenStateSize)
		.Architecture({
			LayerSignature(_classes, new Linear(), _learningRate)
		})
		.Build();


	return _lstm;
}



LSTMbuilder LSTMbuilder::InputSize(size_t size)
{
	_lstm._inputSize = size;

	return (*this);
}



LSTMbuilder LSTMbuilder::CellStateSize(size_t size)
{
	_forgetArchitecture = { LayerSignature(size, new Sigmoid(), _learningRate) };
	_inputArchitecture = { LayerSignature(size, new Sigmoid(), _learningRate) };
	_candidateArchitecture = { LayerSignature(size, new Tanh(), _learningRate) };
	_outputArchitecture = { LayerSignature(size, new Sigmoid(), _learningRate) };


	return (*this);
}

LSTMbuilder LSTMbuilder::LearningRate(double rate)
{
	_lstm._learningRate = rate;
	_learningRate = rate;

	return (*this);
}

LSTMbuilder LSTMbuilder::LossFunction(ILostFunction* func)
{
	_lstm._lossFunc = func;

	return (*this);
}

LSTMbuilder LSTMbuilder::OutputClasses(size_t classes)
{
	_classes  =  classes; 

	return (*this);
}

LSTMbuilder LSTMbuilder::ForgetArchitecture(std::vector<LayerSignature> architect)
{
	_forgetArchitecture = architect;

	return (*this);
}

LSTMbuilder LSTMbuilder::InputArchitecture(std::vector<LayerSignature> architect)
{
	_inputArchitecture = architect;

	return (*this);
}

LSTMbuilder LSTMbuilder::CandidateArchitecture(std::vector<LayerSignature> architect)
{
	_candidateArchitecture = architect;

	return (*this);
}

LSTMbuilder LSTMbuilder::OutputArchitecture(std::vector<LayerSignature> architect)
{
	_outputArchitecture = architect;

	return (*this);
}



LSTMbuilder LSTMbuilder::ForgetArchitecture(std::vector<size_t> architect, double learningRate)
{
	if (learningRate == 0.0) { learningRate = 0.01; }

	for (auto& neuronQtd : architect) {
		_forgetArchitecture.push_back( LayerSignature(neuronQtd, new Sigmoid(), learningRate));
	}

	return (*this);
}

LSTMbuilder LSTMbuilder::InputArchitecture(std::vector<size_t> architect, double learningRate)
{
	if (learningRate == 0.0) { learningRate = 0.01; }

	for (auto& neuronQtd : architect) {
		_inputArchitecture.push_back( LayerSignature(neuronQtd, new Sigmoid(), learningRate) );
	}

	return (*this);
}

LSTMbuilder LSTMbuilder::CandidateArchitecture(std::vector<size_t> architect, double learningRate)
{
	if (learningRate == 0.0) { learningRate = 0.01; }

	for (auto& neuronQtd : architect) {
		_candidateArchitecture.push_back( LayerSignature(neuronQtd, new Tanh(), learningRate));
	}

	return (*this);
}

LSTMbuilder LSTMbuilder::OutputArchitecture(std::vector<size_t> architect, double learningRate)
{
	if (learningRate == 0.0) { learningRate = 0.01; }

	for (auto& neuronQtd : architect) {
		_outputArchitecture.push_back( LayerSignature(neuronQtd, new Sigmoid(), learningRate) );
	}

	return (*this);
}


