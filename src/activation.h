#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

struct Activation {
	double (*function)(double);
	double (*derivative)(double);
};

#endif // ACTIVATION_H