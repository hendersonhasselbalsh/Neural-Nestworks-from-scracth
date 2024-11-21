#pragma once

#include "GnuplotWrapper.h"

#include "../basic-includes.h"
#include <opencv2/opencv.hpp>
#include "../04-MLP/MLP.h"
#include "DataLoader.h"




class Comparations {
	public:
		static void Plot_ComparisonChart(const std::string& file1, const std::string& name1, const std::string& file2, const std::string& name2);
		static void With_WithOut_Adam(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData);
};
