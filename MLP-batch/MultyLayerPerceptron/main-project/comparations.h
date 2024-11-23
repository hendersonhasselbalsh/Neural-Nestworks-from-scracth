#pragma once

#include "GnuplotWrapper.h"

#include "../basic-includes.h"
#include <opencv2/opencv.hpp>
#include "../04-MLP/MLP.h"
#include "DataLoader.h"

using Charts = std::vector<std::pair<std::string, std::string>>; 



class Comparations {
	public:
		static void Plot_Chart(Charts charts);
		static void Plot_ComparisonChart(const std::string& file1, const std::string& name1, const std::string& file2, const std::string& name2);
		static void With_WithOut_Adam(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData);
		static void BatchSize(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData);
		static void LossFunction(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData);
		static void ActivationFunction(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData);
		static void Adam_beta(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData);
};
