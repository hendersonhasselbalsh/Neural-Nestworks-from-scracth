#include "DataLoader.h"

DataLoader::DataLoader(std::string sourceDir)
{
	_sourceDir = sourceDir;
}


std::vector<std::string> DataLoader::SplitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = input.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(input.substr(start, end - start));
        start = end + delimiter.length();
        end = input.find(delimiter, start);
    }

    result.push_back(input.substr(start, end));  // Handle the last token

    return result;
}


Eigen::MatrixXd DataLoader::ImageToMatrix(cv::Mat mat)
{
    if (mat.channels() > 1) { cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY); }

    size_t index = 0;
    Eigen::MatrixXd matrix = Eigen::MatrixXd(mat.rows * mat.cols, 1);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            double pixel  =  (double)mat.at<uchar>(i, j) / 255.0;
            matrix(index++, 0) =  pixel;
        }
    }

    return matrix;
}


Eigen::MatrixXd DataLoader::ImageToInputVector(const std::string& imageFile)
{
    Eigen::MatrixXd input = DataLoader::ImageToMatrix(cv::imread(imageFile)); 
    return input;
}


size_t DataLoader::ImageToInputLabel(const std::string& imageFile) 
{
    std::string labelStr = DataLoader::SplitString(imageFile, "_")[0];
    size_t classLabel = (size_t)std::stoi(labelStr);
    return classLabel;
}


std::vector<std::pair<Eigen::MatrixXd, size_t>> DataLoader::Load()
{
    std::vector<std::pair<Eigen::MatrixXd, size_t>> datas;

    size_t l = -1;
    for (const auto& sample : std::filesystem::directory_iterator(_sourceDir)) {
        if (std::filesystem::is_regular_file(sample.path())) {

            Eigen::MatrixXd inputVector = ImageToInputVector( sample.path().string() ); 
            size_t inputClass = ImageToInputLabel( sample.path().filename().string( ));

            datas.push_back({inputVector, inputClass });
            
            if (inputClass != l) {
                l = inputClass;
                std::cout << "load data: [" << (inputClass+1)*10 << "%]\n";
            }
        }
    }
	
    return datas;
}

//-------------------------------------------------------------------------------------------------------

size_t Evaluator::GetLabel(Eigen::MatrixXd& vec)
{
    size_t predicted = 100, _;
    vec.maxCoeff(&predicted, &_);
    return predicted;
}


double Evaluator::Eval_MLP(MLP& mlp, std::vector<std::pair<Eigen::MatrixXd, size_t>>& datas)
{
    double H = (double) datas.size();
    double error = 0.0;
    Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(10,10);
    Softmax softmax = Softmax();

    for (auto& [input, correct_lable] : datas) {
        Eigen::MatrixXd predicted = mlp.CalculateOutput(input);
        Eigen::MatrixXd prob = softmax.Activation(predicted);
        size_t predicted_label = GetLabel(prob);

        if (correct_lable != predicted_label) { error += 1.0; }

        confusion(predicted_label, correct_lable)++;
    }
    std::cout << "error: " << (error/H) << "\n\n";
    std::cout << confusion << "\n\n";

    return (error/H); 
}
