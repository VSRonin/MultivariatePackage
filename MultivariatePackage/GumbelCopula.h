#ifndef GumbelCopula_h__
#define GumbelCopula_h__
#include "AbstarctArchimedeanCopula.h"
#include <boost/math/tools/roots.hpp>
namespace Multivariate{	
	class GumbelCopula : public AbstractArchimedean{
	private:
		double GeneratorFunction(double x)const;
		double GeneratorInverseFunction(double x)const;
		double GeneratorFunctionDerivative(double x)const;
		double GeneratorInverseFunctionDerivative(double x)const;
		double SimulateGeneratorInverseFourier()const;
		bool CheckValidity();
	public:
		GumbelCopula(unsigned int Dimension);
		GumbelCopula(unsigned int Dimension,double theta);
		bool SetTheta(double t){if(Theta>=1.0) {Theta=t; return true;} return false;}
		Eigen::VectorXd GetQuantile(double Prob)const;
		double GetLowerTailDependence() const;//! \todo Implement GetUpperTailDependence
		bool SetLowerTailDependence(double ltd);//! \todo Implement SetUpperTailDependence

		template <class F, class T> friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits);
		template <class F, class T>	friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);
		template <class F, class T> friend void boost::math::tools::detail::handle_zero_derivative(F f,T& last_f0,const T& f0,T& delta,T& result,T& guess,const T& min,const T& max);
	};
}
#endif // GumbelCopula_h__