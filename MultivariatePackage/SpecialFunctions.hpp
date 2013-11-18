#ifndef SpecialFunctions_h__
#define SpecialFunctions_h__
#include <boost/math/distributions/gamma.hpp>
namespace Multivariate{
	//! Computes the Gamma function
	/*!
	\param t The point in which the function should be evaluated 
	\return The value of the gamma function
	\details Evaluates the \f$ \Gamma \f$ function as \f$ \Gamma(t) = \int_0^\infty  x^{t-1} e^{-x}\,{\rm d}x. \f$
	
	For more information, please refer to [Wikipedia](http://en.wikipedia.org/wiki/Gamma_function) 
	*/
	double GammaFunction(double t){
		boost::math::gamma_distribution<double> GammaDist(t,1.0);
		return exp(-1.0)/pdf(GammaDist,1.0);
	}
}
#endif // SpecialFunctions_h__