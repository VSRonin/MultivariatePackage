#ifndef ClaytonCopula_h__
#define ClaytonCopula_h__
#include "AbstarctArchimedeanCopula.h"
#include <boost/math/tools/roots.hpp>
namespace Multivariate{	
	class ClaytonCopula : public AbstractArchimedean{
	private:
		double GeneratorFunction(double x)const;
		double GeneratorInverseFunction(double x)const;
		double GeneratorFunctionDerivative(double x)const;
		double GeneratorInverseFunctionDerivative(double x)const;
		double SimulateGeneratorInverseFourier()const;
		bool CheckValidity();
	public:
		ClaytonCopula(unsigned int Dimension);
		ClaytonCopula(unsigned int Dimension,double theta);
		bool SetTheta(double t){if(Theta>0.0) {Theta=t; return true;} return false;}
		Eigen::VectorXd GetQuantile(double Prob)const;

		template <class F, class T> friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits);
		template <class F, class T>	friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);
		template <class F, class T> friend void boost::math::tools::detail::handle_zero_derivative(F f,T& last_f0,const T& f0,T& delta,T& result,T& guess,const T& min,const T& max);
	};
}
#endif // ClaytonCopula_h__