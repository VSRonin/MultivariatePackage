#ifndef SpecialFunctions_h__
#define SpecialFunctions_h__
#include <boost/math/distributions/gamma.hpp>
namespace Multivariate{
	double GammaFunction(double x){
		boost::math::gamma_distribution<double> GammaDist(x,1.0);
		return exp(-1.0)/pdf(GammaDist,1.0);
	}
}
#endif // SpecialFunctions_h__