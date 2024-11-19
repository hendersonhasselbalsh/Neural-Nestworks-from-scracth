#include "Sigmoid.h"


//------------------------------
//  SIGMOID
//------------------------------

Sigmoid::Sigmoid()
{
}


Sigmoid::~Sigmoid()
{
}


double Sigmoid::f(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}


Eigen::MatrixXd Sigmoid::Activation(Eigen::MatrixXd& weitedSumVec)
{
    size_t i = 0;
    Eigen::MatrixXd activatedVec = Eigen::MatrixXd(weitedSumVec.rows(), 1);

    for (auto weitedSum : weitedSumVec.rowwise()) {
        double U = weitedSum(0, 0);
        activatedVec(i++, 0) = Sigmoid::f(U);
    }

    return activatedVec;
}


Eigen::MatrixXd Sigmoid::dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec)
{
    return Eigen::MatrixXd();  // not usesd in this particular case
}


Eigen::MatrixXd Sigmoid::Forward(Eigen::MatrixXd& batchWeightedSum)
{
    _activatedBach = batchWeightedSum;

    for (auto& [weightedSumVec, vecIndex] : DataManager::ExtractVectors(batchWeightedSum)) {
        _activatedBach.col(vecIndex) = Sigmoid::Activation(weightedSumVec);
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