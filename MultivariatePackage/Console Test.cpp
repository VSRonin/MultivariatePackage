#include <iostream>
#include "UniformDistribution.h"
using namespace std;
int main(int argc, char* argv[])
{
	Multivariate::IndependenceCopula BivariateUnif(2);
	Eigen::Vector2d Coordinates;
	Coordinates(0)=0.3;
	Coordinates(1)=0.5;
	cout << "Density: " << BivariateUnif.GetDensity()
		<< endl << "Cumulative Density: " << BivariateUnif.GetCumulativeDesity(Coordinates)
		<< endl;
	return 0;
}
