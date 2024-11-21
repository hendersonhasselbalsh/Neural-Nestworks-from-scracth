#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IOptimizationLayer.h"
#include "../02-Dense-Layer/DenseLayer.h"
#include "../08-Optimizers/Optimizers.h"


class LayerManager {
	public:
		static bool IsDenseLayer(ILayer* layer);
		static bool IsBatchNorm(ILayer* layer);
		static bool IsOptimizationLayer(ILayer* layer);
		static Eigen::MatrixXd PrepareVectorAsDenseLayerInput(Eigen::MatrixXd& vec);
};
