#include "ConvolutionCell.h"



//ConvolutionCell::ConvolutionCell(size_t filterSize, double learnRate)
//{
//    _learningRate = learnRate;
//	_filter  =  Eigen::MatrixXd::Ones(filterSize, filterSize);
//
//    for (size_t i = 0; i < _filter.rows(); i++) {
//        for (size_t j = 0; j < _filter.cols(); j++) {
//            _filter(i,j)  =  Utils::RandomUniformDistribution(-1.0, 1.0);
//        }
//    }
//}

ConvolutionCell::ConvolutionCell(size_t filterRow, size_t filterCol, double learnRate)
{
    _learningRate = learnRate;
    _filter  =  Eigen::MatrixXd::Ones(filterRow, filterCol);

    //double range = (double)std::max(filterRow, filterCol);
    double range = 1.0 / (double)std::max(filterRow, filterCol);

    for (size_t i = 0; i < _filter.rows(); i++) {
        for (size_t j = 0; j < _filter.cols(); j++) {
            //_filter(i, j)  =  (1.0 / (double)(filterRow*filterCol));
            _filter(i, j)  =  Utils::RandomUniformDistribution(-range, range);
        }
    }

    //--- DEBUG
    std::cout << "\n\nFILTER:\n" << _filter << "\n\n";
    //--- END DEBUG
}


ConvolutionCell::~ConvolutionCell()
{
}





Eigen::MatrixXd ConvolutionCell::Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter)
{
    assert(input.size() > filter.size()  &&  "size should not be bigger than input");

    const size_t filterRows = filter.rows();
    const size_t filterCols = filter.cols();
    const size_t rows = (input.rows() - filterRows) + 1;
    const size_t cols = (input.cols() - filterCols) + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double sum = input.block(i, j, filterRows, filterCols).cwiseProduct(filter).sum();
            result(i, j) = sum;
        }
    }

    return result;
}



Eigen::MatrixXd ConvolutionCell::Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t padding)
{
    assert(input.rows() + 2*padding, input.cols() + 2*padding > filter.size()  &&  "size should not be bigger than input");

    size_t filterRows = filter.rows();
    size_t filterCols = filter.cols();
    size_t rows = input.rows() - filterRows + 2*padding + 1;
    size_t cols = input.cols() - filterCols + 2*padding + 1;

    Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(input.rows() + 2*padding, input.cols() + 2*padding);
    padded.block(padding, padding, input.rows(), input.cols()) = input;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double sum = padded.block(i, j, filterRows, filterCols).cwiseProduct(filter).sum();
            result(i, j) = sum;   // dont use bias here
        }
    }

    return result;
}


Eigen::MatrixXd ConvolutionCell::Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t rowPadding, size_t colPadding)
{
    assert(input.rows() + 2*rowPadding, input.cols() + 2*colPadding > filter.size()  &&  "size should not be bigger than input");

    size_t filterRows = filter.rows();
    size_t filterCols = filter.cols();
    size_t rows = input.rows() - filterRows + 2*rowPadding + 1;
    size_t cols = input.cols() - filterCols + 2*colPadding + 1;

    Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(input.rows() + 2*rowPadding, input.cols() + 2*colPadding);
    padded.block(rowPadding, colPadding, input.rows(), input.cols()) = input;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double sum = padded.block(i, j, filterRows, filterCols).cwiseProduct(filter).sum();
            result(i, j) = sum;
        }
    }

    return result;
}





Eigen::MatrixXd ConvolutionCell::Forward(Eigen::MatrixXd& input)
{
    _receivedInput = input;
    Eigen::MatrixXd convolvedInput = Convolute(input, _filter);
    return convolvedInput;
}



Eigen::MatrixXd ConvolutionCell::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
    Eigen::MatrixXd dLoss_dFilter  =  Convolute(_receivedInput, dLoss_dOutput);


    _filter  =  _filter - _learningRate * dLoss_dFilter;


    size_t rowPaddingSize = dLoss_dOutput.rows() - 1;
    size_t colPaddingSize = dLoss_dOutput.cols() - 1;

    Eigen::MatrixXd rotated_filter = Utils::Rotate_180Degree( _filter );
    Eigen::MatrixXd dLoss_dInput  =  Convolute(rotated_filter, dLoss_dOutput, rowPaddingSize, colPaddingSize);

    return dLoss_dInput;
}

