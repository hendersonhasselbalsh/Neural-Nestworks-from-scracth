#include "pooling.h"

//--------------
// MAX POOLING
//--------------

MaxPool::MaxPool(size_t poolRow, size_t poolCol)
{
	_poolRow = poolRow;
	_poolCol = poolCol;
}

MaxPool::~MaxPool()
{
}



Eigen::MatrixXd MaxPool::Forward(Eigen::MatrixXd& input)
{
    const int rows = input.rows() / _poolRow;
    const int cols = input.cols() / _poolCol;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);
    _dLoss_dPool  =  Eigen::MatrixXd::Zero(input.rows(), input.cols());

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            Eigen::Index maxRow, maxCol;

            double maxElement = input.block(i*_poolRow, j*_poolCol, _poolRow, _poolCol).maxCoeff(&maxRow, &maxCol);
            result(i, j)  =  maxElement;

            _dLoss_dPool(i*_poolRow + maxRow, j*_poolCol + maxCol)  =  1.0;
        }
    }
    return result;
}

Eigen::MatrixXd MaxPool::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
    for (int i = 0; i < dLoss_dOutput.rows(); i++) {
        for (int j = 0; j < dLoss_dOutput.cols(); j++) {
            _dLoss_dPool.block(i*_poolRow, j*_poolCol, _poolRow, _poolCol) *= dLoss_dOutput(i, j);
        }
    }

    return _dLoss_dPool;
}







//--------------
// MIN POOLING
//--------------

MinPool::MinPool(size_t poolRow, size_t poolCol)
{
    _poolRow = poolRow;
    _poolCol = poolCol;
}

MinPool::~MinPool()
{
}

Eigen::MatrixXd MinPool::Forward(Eigen::MatrixXd& input)
{
    const int rows = input.rows() / _poolRow;
    const int cols = input.cols() / _poolCol;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);
    _dLoss_dPool  =  Eigen::MatrixXd::Zero(input.rows(), input.cols());

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            Eigen::Index maxRow, maxCol;

            double maxElement = input.block(i*_poolRow, j*_poolCol, _poolRow, _poolCol).minCoeff(&maxRow, &maxCol); 
            result(i, j)  =  maxElement;

            _dLoss_dPool(i*_poolRow + maxRow, j*_poolCol + maxCol)  =  1.0;
        }
    }
    return result;
}

Eigen::MatrixXd MinPool::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
    for (int i = 0; i < dLoss_dOutput.rows(); i++) {
        for (int j = 0; j < dLoss_dOutput.cols(); j++) {
            _dLoss_dPool.block(i*_poolRow, j*_poolCol, _poolRow, _poolCol) *= dLoss_dOutput(i, j);
        }
    }

    return _dLoss_dPool;
}







//--------------
// AVERAGE POOLING
//--------------

AveragePool::AveragePool(size_t poolRow, size_t poolCol)
{
    _poolRow = poolRow;
    _poolCol = poolCol;
}

AveragePool::~AveragePool()
{
}

Eigen::MatrixXd AveragePool::Forward(Eigen::MatrixXd& input)
{
    _inputRow = input.rows();
    _inputCol = input.cols();

    const int rows = input.rows() / _poolRow;
    const int cols = input.cols() / _poolCol;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    _size = (double)(input.rows() * input.cols());

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            Eigen::Index maxRow, maxCol;

            double maxElement = input.block(i*_poolRow, j*_poolCol, _poolRow, _poolCol).sum();
            result(i, j)  =  maxElement / _size;
        }
    }
    return result;
}

Eigen::MatrixXd AveragePool::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
    Eigen::MatrixXd dLoss_dPool = Eigen::MatrixXd::Constant(_inputRow, _inputRow, 1.0/_size) ;

    for (int i = 0; i < dLoss_dOutput.rows(); i++) {
        for (int j = 0; j < dLoss_dOutput.cols(); j++) {
            dLoss_dPool.block(i*_poolRow, j*_poolCol, _poolRow, _poolCol) *= dLoss_dOutput(i, j);
        }
    }

    return dLoss_dPool;
}
