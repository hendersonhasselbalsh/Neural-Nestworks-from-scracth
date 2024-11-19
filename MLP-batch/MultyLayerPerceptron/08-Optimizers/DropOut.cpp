#include "DropOut.h"


Dropout::Dropout(double prob)
{
	_prob = prob;
}


Eigen::MatrixXd Dropout::DropoutVector(size_t vectorSize)
{
	_dropOutVector = Eigen::MatrixXd::Ones(vectorSize, 1);

	std::random_device rd;
	std::mt19937 gen(rd()); 
	std::bernoulli_distribution dist(_prob); 

	for (size_t i = 0; i < vectorSize; i++) {
		if ( dist(gen) ) { _dropOutVector(i,0) = 0.0; }
	}

	return _dropOutVector;
}


Eigen::MatrixXd Dropout::Forward(Eigen::MatrixXd& input)
{ 
	_dropOutVector = DropoutVector( input.rows() );

	for (auto inputVec : input.colwise()) {
		inputVec = inputVec.cwiseProduct( _dropOutVector );
	}

	return input; // as dropout output
}


Eigen::MatrixXd Dropout::Backward(Eigen::MatrixXd& dL_dO)
{
	/*for (auto inputVec : dL_dO.colwise()) { 
		inputVec = inputVec.cwiseProduct(_dropOutVector);    
	}*/

	return dL_dO; // as dropout output
}


