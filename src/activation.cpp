#include "activation.h"

Activation ActivationFunctions::sigmoid = {
	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
	{
		return 1.0 / (1.0 + (-x.array()).exp());
	},
	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
	{
		return x.array() * (1.0 - x.array());
	}
};

// Activation ActivationFunctions::relu = {
// 	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
// 	{
// 		return x.array().max(0.0);
// 	},
// 	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
// 	{
//         return x.array().max(0.0).array();
// 	}
// };

// Activation ActivationFunctions::leaky_relu = {
// 	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
// 	{
// 		return x.array().max(0.0) + 0.01 * x.array().min(0.0);
// 	},
// 	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
// 	{
//         return (x.array() > 0.0).select(x.array(), 0.01 * x.array());
// 	}
// };

Activation ActivationFunctions::tanh = {
	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
	{
		return x.array().tanh();
	},
	[](const Eigen::VectorXd &x) -> Eigen::VectorXd
	{
		return 1.0 - x.array().square();
	}
};