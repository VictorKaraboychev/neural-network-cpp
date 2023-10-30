#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

struct Activation
{
	double (*function)(double);
	double (*derivative)(double);
};

class ActivationFunctions
{
public:
	static Activation sigmoid;
	static Activation relu;
	static Activation leaky_relu;
	static Activation tanh;
};


#endif // ACTIVATION_H