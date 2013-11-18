#include <iostream>
#include "NormalDist.h"

int main(int argc, char* argv[])
{
	std::cout.precision(4); //Set the output to have 4 decimal digits
	//Construct the mean vector with values [1 ; -1]
	Eigen::Vector2d MeanVector;
	MeanVector(0)=1.0;
	MeanVector(1)=-1.0;
	//Construct the Var-Cov matrix so that the variance of both margins is 3 and the covariance is 1
	Eigen::Matrix2d VarMatrix;
	VarMatrix(0,0)=3;
	VarMatrix(0,1)=1;
	VarMatrix(1,0)=1;
	VarMatrix(1,1)=3;
	//Construct a Bivariate Normal with the above parameters
	Multivariate::NormalDistribution BivarNorm(2,MeanVector,VarMatrix);
	//Set the random number generator seed so that results are 100% reproducible
	BivarNorm.SetRandomSeed(99);
	//Compute and print the density and the cumulative density of the distribution in the point [1 ; 0.5]
	Eigen::Vector2d TempCoords;
	TempCoords(0)=1.0;
	TempCoords(1)=0.5;
	double Density=BivarNorm.GetDensity(TempCoords);
	double CumDensity=BivarNorm.GetCumulativeDesity(TempCoords);
	
	std::cout << "Density in [1 ; 0.5] = " << std::fixed << Density
		<< std::endl << "Cumulative Density in [1 ; 0.5] = " << std::fixed << CumDensity;
	// Compute the 83rd percentile
	Eigen::Vector2d Quant83=BivarNorm.GetQuantile(0.83);
	std::cout << std::endl << "83rd Percentile = [" << Quant83.transpose()<< " ]";

	//Simulate 10 realizations and print the results
	Eigen::Matrix<double,10,2> Samples=BivarNorm.ExtractSamples(10);
	std::cout << std::endl << "Simulation Results:" << std::endl << "   Var 1\t Var2"<< std::endl << Samples << std::endl;
	return 0;
}