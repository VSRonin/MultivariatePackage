#include <iostream>
#include <vector>
#include "NormalDist.h"
int main(int argc, char* argv[])
{
	std::cout.precision(4); //Set the output to have 4 decimal digits
	//Construct a Bivariate Normal Distribution
	Multivariate::NormalDistribution BivarNorm(2);
	//Set the random number generator seed so that results are 100% reproducible
	BivarNorm.SetRandomSeed(88);
	// Set the mean vector to [1 ; -1]
	std::vector<double> MeanVector(2);
	MeanVector[0]=1.0;
	MeanVector[1]=-1.0;
	BivarNorm.SetMeanVector(MeanVector);
	//Set the variance of both margins to 3 and the covariance to 1
	std::vector<double> VarMatrix(4);
	VarMatrix[0]=3.0;
	VarMatrix[1]=1.0;
	VarMatrix[2]=1.0;
	VarMatrix[3]=3.0;
	BivarNorm.SetVarCovMatrix(VarMatrix);
	//Compute and print the density and the cumulative density of the distribution in the point [1 ; 0.5]
	std::vector<double> TempCoords;
	TempCoords.push_back(1.0);
	TempCoords.push_back(0.5);
	double Density=BivarNorm.GetDensity(TempCoords);
	double CumDensity=BivarNorm.GetCumulativeDesity(TempCoords);
	std::cout << "Density in [1 ; 0.5] = " << std::fixed << Density
		<< std::endl << "Cumulative Density in [1 ; 0.5] = " << std::fixed << CumDensity ;
	//Simulate 10 realizations and print the results
	std::map<unsigned int,std::vector<double> > Samples=BivarNorm.ExtractSamplesMap(10);
	std::cout << std::endl << "Simulation Results:"<< std::endl << "Var 1\t| Var2"<< std::endl << "_________________";
	for(int j=0;j<10;j++){
		std::cout << std::endl;
		for(unsigned int i=0;i<BivarNorm.GetDimension();i++){
			std::cout << std::fixed << Samples.at(i).at(j);
			if(i<BivarNorm.GetDimension()-1) std::cout << "\t| ";
		}
	}
	std::cout << std::endl;
	return 0;
}
