#include "MLPbuilder.h"

MLPbuilder::MLPbuilder()
{
	_mlp = MLP();
	_mlp._useAdam = false;
}


MLPbuilder MLPbuilder::InputSize(size_t inputSize)
{
	_inputSize = inputSize;
	return (*this);
}


MLPbuilder MLPbuilder::LossFunction(ILossFunction* lossFunc)
{
	_mlp._lossFunc = lossFunc;
	return (*this);
}


MLPbuilder MLPbuilder::MaxEpochs(size_t maxEpochs)
{
	_mlp._max_epochs = maxEpochs;
	return (*this);
}


MLPbuilder MLPbuilder::BatchSize(long batchSize)
{
	_mlp._batchSize = batchSize;
	return (*this);
}


MLPbuilder MLPbuilder::OutputClasses(size_t classes)
{
	_mlp._outputClasses = classes;
	return (*this);
}


MLPbuilder MLPbuilder::Architecture(std::vector<ILayer*> layers)
{
	_mlp._layers = layers;

	for (auto& layer : _mlp._layers) {
		size_t outputSize;
		layer->Initialize(_inputSize, &outputSize);
		_inputSize = outputSize;
	}

	_mlp._outputClasses = _inputSize; // last layer output size

	return (*this);
}

MLPbuilder MLPbuilder::UseAdam(double beta)
{
	assert(beta <= 0.0 && beta >= 1.0);

	_mlp._beta = beta;
	_mlp._useAdam = true;

	return (*this);
}


MLPbuilder MLPbuilder::ShuffleData(bool shuffle)
{
	_mlp._shuffleData = shuffle;
	return (*this);
}


MLP MLPbuilder::Build()
{
	return _mlp;
}
