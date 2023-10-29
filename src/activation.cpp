#include "activation.h"

Activation sigmoid = {
	[](double x) -> double {
		return 1.0 / (1.0 + exp(-x));
	},
	[](double x) -> double {
		double s = 1.0 / (1.0 + exp(-x));
		return s * (1.0 - s);
	}
};

Activation relu = {
	[](double x) -> double {
		return x > 0.0 ? x : 0.0;
	},
	[](double x) -> double {
		return x > 0.0 ? 1.0 : 0.0;
	}
};

Activation leaky_relu = {
	[](double x) -> double {
		return x > 0.0 ? x : 0.01 * x;
	},
	[](double x) -> double {
		return x > 0.0 ? 1.0 : 0.01;
	}
};

Activation tanh = {
	[](double x) -> double {
		return std::tanh(x);
	},
	[](double x) -> double {
		return 1 / std::pow(std::cosh(x), 2.0);
	}
};