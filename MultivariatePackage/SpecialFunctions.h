#ifndef SpecialFunctions_h__
#define SpecialFunctions_h__
namespace Multivariate{
	//! Simulates a draw from a stable distribution
	/*!
	\param Alpha stability parameter. Must be in the interval (0,2]
	\param Beta skewness parameter. Must be in the interval [-1,1]
	\param Gamma scale parameter. Must be greater than 0
	\param Delta location parameter.
	\param RandNumGenSeed the seed for the random number generation
	\return A draw from a stable distribution
	\detail This function generates a random sample from a generic stable distribution
	
	For more information, please refer to [Wikipedia](http://en.wikipedia.org/wiki/Stable_distribution) 
	
	The algorithm used is the one illustrated by [Nolan (2009)](http://academic2.american.edu/~jpnolan/stable/chap1.pdf) 

	*/
	double SimulateStable(double Alpha,double Beta,double Gamma,double Delta,unsigned int RandNumGenSeed=0);
}
#endif // SpecialFunctions_h__