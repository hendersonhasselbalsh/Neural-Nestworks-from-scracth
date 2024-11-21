#pragma once

#include "../basic-includes.h"
#include <opencv2/opencv.hpp>
#include "../04-MLP/MLP.h"

class DataLoader {
	private:
		std::string _sourceDir;

	public:
		DataLoader(std::string sourceDir);
		
		static std::vector<std::string> SplitString(const std::string& input, const std::string& delimiter);
		static Eigen::MatrixXd ImageToMatrix(cv::Mat mat); 
		static Eigen::MatrixXd ImageToInputVector(const std::string& imageFile);
		static size_t ImageToInputLabel(const std::string& imageFile);

		std::vector<std::pair<Eigen::MatrixXd, size_t>> Load();

};


class Evaluator {
	public:
		static size_t GetLabel(Eigen::MatrixXd& vec);
		static double Eval_MLP(MLP& mlp, std::vector<std::pair<Eigen::MatrixXd, size_t>> datas);
};
