#include "ReLU.h"



//------------------------------
//  ReLU
//------------------------------

ReLU::ReLU()
{
}


double ReLU::f(double x)
{
    return std::max(0.0, x);
}


double ReLU::df(double x)
{
    if (x >= 0) { return 1.0; } else { return 0; }
}


Eigen::MatrixXd ReLU::Activation(Eigen::MatrixXd& weitedSumVec)
{
    size_t i = 0;
    Eigen::MatrixXd activatedVec = Eigen::MatrixXd(weitedSumVec.rows(), 1);

    for (auto weitedSum : weitedSumVec.rowwise()) {
        double U = weitedSum(0, 0);
        activatedVec(i++, 0) = ReLU::f(U);
    }

    return activatedVec;
}


Eigen::MatrixXd ReLU::dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec)
{
    size_t i = 0;
    Eigen::MatrixXd dA_dU = Eigen::MatrixXd(weitedSumVec.rows(), 1);

    for (auto weitedSum : weitedSumVec.rowwise()) {
        double U = weitedSum(0, 0);
        dA_dU(i++, 0) = ReLU::df(U);
    }

    return dA_dU;
}


Eigen::MatrixXd ReLU::Forward(Eigen::MatrixXd& weightedSumBatch)
{
    _receivedBachInput = weightedSumBatch;
    Eigen::MatrixXd activation = Eigen::MatrixXd(weightedSumBatch.rows(), weightedSumBatch.cols());

    for (auto& [weightedSumVec, vecIndex] : DataManager::ExtractVectors(weightedSumBatch)) {
        activation.col(vecIndex) = ReLU::Activation(weightedSumVec);
    }

    return activation;
}


Eigen::MatrixXd ReLU::Backward(Eigen::MatrixXd& dLdA_batch)
{
    size_t i = 0;
    Eigen::MatrixXd dL_dU = Eigen::MatrixXd(dLdA_batch.rows(), dLdA_batch.cols());

    for (auto& [dLdA, weightedSum] : DataManager::ExtractCorrespondingVectors(dLdA_batch, _receivedBachInput)) {
        Eigen::MatrixXd dAdU = dActivation_dWeightedSum(weightedSum);
        dL_dU.col(i++) = dLdA.cwiseProduct(dAdU);
    }

    return dL_dU;
}


