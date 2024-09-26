#include "ConvolutionCell.h"



//ConvolutionCell::ConvolutionCell(size_t filterRow, size_t filterCol, double learnRate)
//{
//    _paddingSize = Padding{ 0, 0 };
//
//    _learningRate = learnRate;
//    _filter  =  Eigen::MatrixXd::Ones(filterRow, filterCol);
//
//    //double range = 1.0 / (double)std::max(filterRow, filterCol);
//    double range = 1.0 / (double)(filterRow * filterCol);
//
//
//
//    for (size_t i = 0; i < _filter.rows(); i++) {
//        for (size_t j = 0; j < _filter.cols(); j++) {
//            //_filter(i, j)  =  (1.0 / (double)(filterRow*filterCol));
//            _filter(i, j)  =  Utils::RandomUniformDistribution(-range, range);
//        }
//    }
//
//    //--- DEBUG
//    std::cout << "\n\nFILTER:\n" << _filter << "\n\n";
//    //--- END DEBUG
//}

//ConvolutionCell::ConvolutionCell(Filter filterSize, double learnRate)
//{
//    _paddingSize = Padding{ 0, 0 };
//
//    _learningRate = learnRate;
//    _filter  =  Eigen::MatrixXd::Ones(filterSize._row, filterSize._col);
//
//    double range = 1.0 / (double)(filterSize._row * filterSize._col);
//
//
//    for (size_t i = 0; i < _filter.rows(); i++) {
//        for (size_t j = 0; j < _filter.cols(); j++) {
//            //_filter(i, j)  =  (1.0 / (double)(filterRow*filterCol));
//            _filter(i, j)  =  Utils::RandomUniformDistribution(-range, range);
//        }
//    }
//
//    //--- DEBUG
//    //std::cout << "\n\nFILTER:\n" << _filter << "\n\n";
//    //--- END DEBUG
//}

//ConvolutionCell::ConvolutionCell(Filter filterSize, Padding padding, double learnRate)
//{
//    _paddingSize = padding;
//
//    _learningRate = learnRate;
//    _filter  =  Eigen::MatrixXd::Ones(filterSize._row, filterSize._col);
//
//    double range = 1.0 / (double)(filterSize._row * filterSize._col);
//
//
//    for (size_t i = 0; i < _filter.rows(); i++) {
//        for (size_t j = 0; j < _filter.cols(); j++) {
//            //_filter(i, j)  =  (1.0 / (double)(filterRow*filterCol));
//            _filter(i, j)  =  Utils::RandomUniformDistribution(-range, range);
//        }
//    }
//
//    //--- DEBUG
//    //std::cout << "\n\nFILTER:\n" << _filter << "\n\n";
//    //--- END DEBUG
//}




ConvolutionCell::ConvolutionCell(size_t filterQnt, Filter filterSize, double learnRate)
{
    _filterQnt = filterQnt; 

    _paddingSize = Padding{ 0, 0 };

    _learningRate = learnRate; 
    _filter  = Eigen::Tensor<double, 3>(filterQnt, filterSize._row, filterSize._col);

    double range = 1.0 / (double)(filterSize._row * filterSize._col); 

    for (size_t z = 0; z < filterQnt; z++) {
        for (size_t i = 0; i < filterSize._row; i++) {
            for (size_t j = 0; j < filterSize._col; j++) {
                _filter(z, i, j)  =  Utils::RandomUniformDistribution(-range, range);  
            }
        }
    }
}


ConvolutionCell::ConvolutionCell(size_t filterQnt, Filter filterSize, Padding padding, double learnRate)
{
    _filterQnt = filterQnt;

    _paddingSize = padding; 

    _learningRate = learnRate;
    _filter  = Eigen::Tensor<double, 3>(filterQnt, filterSize._row, filterSize._col);

    double range = 1.0 / (double)(filterSize._row * filterSize._col);

    for (size_t z = 0; z < filterQnt; z++) {
        for (size_t i = 0; i < filterSize._row; i++) {
            for (size_t j = 0; j < filterSize._col; j++) {
                _filter(z, i, j)  =  Utils::RandomUniformDistribution(-range, range);
            }
        }
    }
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




void ConvolutionCell::UpdateLearningRate(size_t epoch, double error, std::function<void(size_t, double, double&)> UpdateRule)
{
    UpdateRule(epoch, error, _learningRate);
}



Eigen::Tensor<double, 3> ConvolutionCell::Forward(Eigen::Tensor<double, 3>& input)
{
    _receivedInput = input;

    size_t convRows = input.dimension(1) - _filter.dimension(1) + 2 * _paddingSize._row + 1;
    size_t convCols = input.dimension(1) - _filter.dimension(1) + 2 * _paddingSize._col + 1;

    Eigen::Tensor<double, 3> convResult = Eigen::Tensor<double, 3>(_filterQnt, convRows, convCols);

    for (size_t f = 0; f < _filter.dimension(0); f++) {
        Eigen::MatrixXd filter  =  Utils::TensorSlice(_filter, f);
        Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(convRows, convCols);

        for (size_t ip = 0; ip < input.dimension(0); ip++) {
            Eigen::MatrixXd inputMat = Utils::TensorSlice(input, ip);

            auto conv  =  Convolute(inputMat, filter);
            sum = sum + conv;
        }

        convResult.chip(f, 0)  =  sum;
    }

    return convResult;
}



Eigen::Tensor<double, 3> ConvolutionCell::Backward(Eigen::Tensor<double, 3>& dLoss_dOutputs)
{
    // dL_dFilters
    for (size_t f = 0; f < _filter.dimension(0); f++) {
        Eigen::MatrixXd dLoss_dConv = Utils::TensorSlice(dLoss_dOutputs, f);

        Eigen::MatrixXd filter = Utils::TensorSlice(_filter, f);
        Eigen::MatrixXd dLoss_dFilter = Eigen::MatrixXd::Zero(filter.rows(), filter.cols());

        for (size_t ip = 0; ip < _receivedInput.dimension(0); ip++) {
            Eigen::MatrixXd input  =  Utils::TensorSlice(_filter, ip);

            dLoss_dFilter  =  dLoss_dFilter  +  Convolute(input, dLoss_dConv);
        }

        _filter.chip(f, 0)  =  filter - _learningRate * dLoss_dConv; 
    }

    // dLoss_dInput
    size_t rowPadding = dLoss_dOutputs.dimension(1) - 1;
    size_t colPadding = dLoss_dOutputs.dimension(2) - 1;

    auto dim  =  _receivedInput.dimensions();
    Eigen::Tensor<double, 3> dL_dInput  =  Eigen::Tensor<double, 3>(dim.at(0), dim.at(1), dim.at(2));

    
    for (size_t f = 0; f < _filter.dimension(0); f++) {
        Eigen::MatrixXd filter  =  Utils::TensorSlice(_filter, f);
        Eigen::MatrixXd dLoss_dConv  =  Utils::TensorSlice(dLoss_dOutputs, f); 

        auto rotatedFilter  =  Utils::Rotate_180Degree(filter);
        auto val = Convolute(rotatedFilter, dLoss_dConv, rowPadding, colPadding);

        for (size_t ip = 0; ip < _receivedInput.dimension(0); ip++) {
            dL_dInput.chip(ip,0)  +=  val; 
        }
    }
    
    return dL_dInput;
}


/*
Eigen::MatrixXd ConvolutionCell::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
    Eigen::MatrixXd dLoss_dFilter  =  Convolute(_receivedInput, dLoss_dOutput);


    _filter  =  _filter - _learningRate * dLoss_dFilter;


    

    Eigen::MatrixXd rotated_filter = Utils::Rotate_180Degree( _filter );
    Eigen::MatrixXd dLoss_dInput  =  Convolute(rotated_filter, dLoss_dOutput, rowPaddingSize, colPaddingSize);

    return dLoss_dInput;
}*/
