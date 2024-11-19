#pragma once

#include "../basic-includes.h"

class DataManager {
	public:
		DataManager();

        static double RandomNormalDistributionValue(double min, double max);

        static double RandomUniformDistribution(double min, double max);

		static void XaviverInitialization(Eigen::MatrixXd& weigts, size_t inputSize, size_t outpsize);

		static std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> ExtractCorrespondingVectors(Eigen::MatrixXd& dLdU_vecs, Eigen::MatrixXd& inputVecs);

		static std::vector<std::pair<Eigen::MatrixXd, size_t>> ExtractVectors(Eigen::MatrixXd& inputVecs);

		static void Shuffle(std::vector<std::pair<Eigen::MatrixXd, size_t>>* data);

		static std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> BuildBatch(
			std::vector<std::pair<Eigen::MatrixXd,size_t>>& datas
			, long batchSize
			, size_t classes
		);

};		

