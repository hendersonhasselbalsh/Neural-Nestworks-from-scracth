#include "ActivationFunction.h"

//------------------------------
//  LINEAR
//------------------------------

Linear::Linear()
{
    //_receivedBatchWeightedSum = nullptr;
}

Linear::~Linear()
{
    //delete _receivedBatchWeightedSum;
}

Eigen::MatrixXd Linear::Forward(Eigen::MatrixXd& batchWeightedSum) 
{
    return batchWeightedSum;
}

Eigen::MatrixXd Linear::Backward(Eigen::MatrixXd& dL_dbatchActivation)
{
    return dL_dbatchActivation;
}



//------------------------------
//  SIGMOID
//------------------------------

Sigmoid::Sigmoid()
{
    //_activatedBach = nullptr;
}

Sigmoid::~Sigmoid()
{
    //delete _activatedBach;
}

double Sigmoid::f_sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

Eigen::MatrixXd Sigmoid::Forward(Eigen::MatrixXd& batchWeightedSum)
{
    _activatedBach = batchWeightedSum;

    for (size_t i = 0; i < batchWeightedSum.rows(); i++) {
        for (size_t j = 0; j < batchWeightedSum.cols(); j++) {
            _activatedBach(i,j) = f_sigmoid( batchWeightedSum(i,j) );
        }
    }

    return _activatedBach;
}

Eigen::MatrixXd Sigmoid::Backward(Eigen::MatrixXd& dL_dbatchActivation)
{
    Eigen::MatrixXd& output = _activatedBach;
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(output.rows(), output.cols());

    Eigen::MatrixXd dActivation_dU = output.cwiseProduct(ones - output);

    Eigen::MatrixXd dL_dU = dL_dbatchActivation.cwiseProduct(dActivation_dU);

    return dL_dU;
}
