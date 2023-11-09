#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <Eigen/Dense>

struct Activation
{
	Eigen::VectorXd (*function)(const Eigen::VectorXd&);
	Eigen::VectorXd (*derivative)(const Eigen::VectorXd&);
};

class ActivationFunctions
{
public:
	static Activation sigmoid;
	// static Activation relu;
	// static Activation leaky_relu;
	static Activation tanh;
};


#endif // ACTIVATION_H