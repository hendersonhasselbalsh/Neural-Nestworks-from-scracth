#include "Tanh.h"


Tanh::Tanh()
{
}


double Tanh::f(double x)
{
    return std::tanh(x);
}


double Tanh::df(double x)
{
    double cosh = std::cosh(x);
    return 1.0 / (cosh*cosh);    // sech^2(x)
}


Eigen::MatrixXd Tanh::Activation(Eigen::MatrixXd& weitedSumVec)
{
    size_t i = 0;
    Eigen::MatrixXd activatedVec = Eigen::MatrixXd(weitedSumVec.rows(), 1);

    for (auto weightedSum : weitedSumVec.rowwise()) {
        double U = weightedSum(0,0);
        activatedVec(i++, 0) = Tanh::f(U);
    }

    return activatedVec;
}


Eigen::MatrixXd Tanh::dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec)
{
    size_t i = 0;
    Eigen::MatrixXd dA_dU = Eigen::MatrixXd(weitedSumVec.rows(), 1); 

    for (auto weitedSum : weitedSumVec.rowwise()) { 
        double U = weitedSum(0, 0);
        dA_dU(i++, 0) = Tanh::df(U); 
    }

    return dA_dU;
}

Eigen::MatrixXd Tanh::Forward(Eigen::MatrixXd& weightedSumBatch)
{
    _receivedInputBatch = weightedSumBatch;
    Eigen::MatrixXd activation = Eigen::MatrixXd(weightedSumBatch.rows(), weightedSumBatch.cols()); 

    for (auto& [weightedSumVec, vecIndex] : DataManager::ExtractVectors(weightedSumBatch)) {
        activation.col(vecIndex) = Tanh::Activation(weightedSumVec); 
    }

    return activation;
}

Eigen::MatrixXd Tanh::Backward(Eigen::MatrixXd& dLdA_batch) 
{
    size_t i = 0;
    Eigen::MatrixXd dL_dU = Eigen::MatrixXd(dLdA_batch.rows(), dLdA_batch.cols());

    for (auto& [dLdA, weightedSum] : DataManager::ExtractCorrespondingVectors(dLdA_batch, _receivedInputBatch)) {
        Eigen::MatrixXd dAdU = dActivation_dWeightedSum(weightedSum);
        dL_dU.col(i++) = dLdA.cwiseProduct(dAdU);
    }

    return dL_dU;
}