#include "lstm.h"



LSTM::LSTM()
{
}


LSTM::~LSTM()
{
}



std::vector<double> LSTM::Foward(std::vector<double> input)
{
	_previousCellState  =  _cellState;

	_gatesInput  =  std::vector<double>();
	_gatesInput.insert(_gatesInput.end(), _hiddenState.begin(), _hiddenState.end());
	_gatesInput.insert(_gatesInput.end(), input.begin(), input.end());


	_gatesInput.insert(_gatesInput.begin(), 1.0);  // input for bias


	_forgetActivation  =  _forgetMLP.Foward( _gatesInput );
	_inputActivation   =  _inputMLP.Foward( _gatesInput );
	_candidateActivation  =  _candidateMLP.Foward( _gatesInput );
	_outputActivation  =  _outputMLP.Foward( _gatesInput );


	// cell state: C<t>  =  f [x] C<t-1>   [+]   i [x] pc 
	std::vector<double> fMULTc = std::vector<double>(_forgetActivation.size(), 0.0);
	std::vector<double> iMULTpc = std::vector<double>(_forgetActivation.size(), 0.0);
	Utils::PointwiseMult(&_forgetActivation, &_previousCellState, &fMULTc);
	Utils::PointwiseMult(&_inputActivation, &_candidateActivation, &iMULTpc);
	Utils::PointwiseAdd(&fMULTc, &iMULTpc, &_cellState);


	// hidden state: H<t>  =  tanh(C<t>)  [x]  o 
	std::vector<double> tanh_cellState  =  Utils::PointwiseTanh( _cellState );
	Utils::PointwiseMult(&tanh_cellState, &_outputActivation, &_hiddenState);


	std::vector<double> hiddenState  =  _hiddenState;
	hiddenState.insert(hiddenState.begin(), 1.0);


	std::vector<double> predicted_y;

	//_linearOutput  =  _linearMLP.Foward( hiddenState );  // use softmax
	//predicted_y  =  Softmax( _linearOutput );            // use softmax

	predicted_y  =  _linearMLP.Foward(hiddenState);  // don't use softmax


	return predicted_y;
}



void LSTM::Backward(std::vector<double> predictedY, std::vector<double> correctY)
{
	std::vector<double> dsoftmax  =  dSoftmax( _linearOutput );

	// update linear MLP 
	std::vector<double> dLoss_dLinearOutput = LossPartialWithRespectToLinearOutput(predictedY, correctY, dsoftmax);
	_linearMLP.Backward( dLoss_dLinearOutput );

	
	std::vector<double> dLoss_dHiddenState = LossPartialWithRespectToHiddenState(predictedY, correctY, dsoftmax);


	// uppdate output mlp
	std::vector<double> dLoss_dOutput = LossPartialWithRespectToOutput( dLoss_dHiddenState );
	_outputMLP.Backward( dLoss_dOutput );


	std::vector<double> dLoss_dCellState  =  LossPartialWithRespectToCellState( dLoss_dHiddenState );


	// update candidate MLP
	std::vector<double> dLoss_dCandidate = LossPartialWithRespectToCandidate( dLoss_dCellState );
	_candidateMLP.Backward( dLoss_dCandidate );


	// update input MLP
	std::vector<double> dLoss_dInput = LossPartialWithRespectToInput( dLoss_dCellState );
	_inputMLP.Backward( dLoss_dInput );


	// update forget MLP 
	std::vector<double> dLoss_dForget = LossPartialWithRespectToForget( dLoss_dCellState );
	_forgetMLP.Backward( dLoss_dForget );
}





std::vector<double> LSTM::LossPartialWithRespectToLinearOutput(std::vector<double>& predictedY, std::vector<double>& correctY, std::vector<double>& dSoftmax)
{
	std::vector<double> dLoss_dLinearOutput = std::vector<double>(predictedY.size(), 0.0);

	for (size_t i = 0; i < dLoss_dLinearOutput.size(); i++) {
		double dLoss_dy  =  _lossFunc->df(predictedY[i], correctY[i]);
		//dLoss_dLinearOutput[i]  =  dLoss_dy * dSoftmax[i];     // <-- use softmax
		dLoss_dLinearOutput[i]  =  dLoss_dy;                     // <-- don't use softmax
	}

	return dLoss_dLinearOutput;
}


std::vector<double> LSTM::LossPartialWithRespectToHiddenState(std::vector<double>& predictedY, std::vector<double>& correctY, std::vector<double>& dSoftmax)
{
	std::vector<double> dLoss_dHiddenState = std::vector<double>(_hiddenState.size(), 0.0);

	Layer& linearWeights  =  _linearMLP.LastLayer();

	for (size_t i = 0; i < _hiddenState.size(); i++) {
		double sum  =  0.0;

		for (size_t j = 0; j < predictedY.size(); j++) {
			//sum  +=  _lossFunc->df(predictedY[j],correctY[j])  *  dSoftmax[j]  *  linearWeights(j,i);    // <-- use softmax
			sum  +=  _lossFunc->df(predictedY[j], correctY[j])  *  linearWeights(j,i);                 // <-- don't use softmax
		}

		dLoss_dHiddenState[i]  =  sum;
	}

	return dLoss_dHiddenState;
}


std::vector<double> LSTM::LossPartialWithRespectToCellState(std::vector<double>& dLoss_dHiddenState)
{
	std::vector<double> dLoss_dCellState = std::vector<double>(dLoss_dHiddenState.size(), 0.0);

	IActivationFunction* tanh = new Tanh();

	for (size_t i = 0; i < dLoss_dHiddenState.size(); i++) {
		dLoss_dCellState[i]  =  dLoss_dHiddenState[i]  *  _outputActivation[i] * tanh->df(_cellState[i]);
	}

	return dLoss_dCellState;
}


std::vector<double> LSTM::LossPartialWithRespectToOutput(std::vector<double>& dLoss_dHiddenState)
{
	std::vector<double> dLoss_dOutput = std::vector<double>(dLoss_dHiddenState.size(), 0.0);

	for (size_t i = 0; i < dLoss_dHiddenState.size(); i++) {
		dLoss_dOutput[i]  =  dLoss_dHiddenState[i] * std::tanh( _cellState[i] );
	}

	return dLoss_dOutput;
}


std::vector<double> LSTM::LossPartialWithRespectToCandidate(std::vector<double>& dLoss_dCellState)
{
	std::vector<double> dLoss_dCandidate = std::vector<double>(dLoss_dCellState.size(), 0.0);

	for (size_t i = 0; i < dLoss_dCellState.size(); i++) {
		dLoss_dCandidate[i]  =  dLoss_dCellState[i] * _inputActivation[i];
	}

	return dLoss_dCandidate;
}



std::vector<double> LSTM::LossPartialWithRespectToInput(std::vector<double>& dLoss_dCellState)
{
	std::vector<double> dLoss_dInput = std::vector<double>(dLoss_dCellState.size(), 0.0);

	for (size_t i = 0; i < dLoss_dCellState.size(); i++) {
		dLoss_dInput[i]  =  dLoss_dCellState[i] * _candidateActivation[i];
	}

	return dLoss_dInput;
}


std::vector<double> LSTM::LossPartialWithRespectToForget(std::vector<double>& dLoss_dCellState)
{
	std::vector<double> dLoss_dForget = std::vector<double>(dLoss_dCellState.size(), 0.0);

	for (size_t i = 0; i < dLoss_dCellState.size(); i++) {
		dLoss_dForget[i]  =  dLoss_dCellState[i] * _previousCellState[i];
	}

	return dLoss_dForget;
}





std::vector<double> LSTM::Softmax(const std::vector<double>& input)
{
	std::vector<double> output(input.size());

	double sum = 0.0;
	for (double val : input) {
		sum += std::exp(val);
	}

	for (size_t i = 0; i < input.size(); ++i) {
		output[i] = std::exp(input[i]) / sum;
	}

	return output;
}



std::vector<double> LSTM::dSoftmax(const std::vector<double>& input)
{
	std::vector<double> result  =  Softmax(input);

	for (auto& r : result) {
		r  =  r * (1.0 - r);
	}

	return result;
}


