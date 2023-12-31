#include "activation.h"

Activation ActivationFunctions::sigmoid = {
	[](double x) -> double
	{
		return 1.0 / (1.0 + exp(-x));
	},
	[](double x) -> double
	{
		return x * (1.0 - x);
	}};

Activation ActivationFunctions::relu = {
	[](double x) -> double
	{
		return x > 0.0 ? x : 0.0;
	},
	[](double x) -> double
	{
		return x > 0.0 ? 1.0 : 0.0;
	}};

Activation ActivationFunctions::leaky_relu = {
	[](double x) -> double
	{
		return x > 0.0 ? x : 0.01 * x;
	},
	[](double x) -> double
	{
		return x > 0.0 ? 1.0 : -0.01;
	}};

Activation ActivationFunctions::tanh = {
	[](double x) -> double
	{
		return std::tanh(x);
	},
	[](double x) -> double
	{
		return 1.0 - x * x;
	}};