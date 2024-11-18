#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../02-Dense-Layer/DenseLayer.h"


class LayerManager {
	public:
		static bool IsDenseLayer(ILayer* layer);
		static Eigen::MatrixXd PrepareVectorAsDenseLayerInput(Eigen::MatrixXd& vec);
};
