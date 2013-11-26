#ifndef ClaytonCopula_h__
#define ClaytonCopula_h__
#include "AbstarctArchimedeanCopula.h"
#include <boost/math/tools/roots.hpp>
namespace Multivariate{
	//! Clayton Copula Distribution
	/*!
	\details This class provides the functionality of calculating the probability density value, cumulative probability density value, inverse cumulative probability density and generate random samples from a Clayton copula.

	Defining:
		- \f$ k \f$ as the dimensionality of the copula
		- \f$ \theta \f$ as the parameter that models the dependence. \f$ \theta=\frac{2 \tau}}{1- \tau} \f$ where \f$ \tau \f$ is the [Kendall's tau](http://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient) statistic
		- \f$ \psi(x) \f$ as \f$ \frac{x^{- \theta} -1}{\theta} \f$
		- \f$ \psi^{-1} (x) \f$ as \f$ (1+ \theta x)^{-1/ \theta} \f$
		
	The Clayton copula funtion is defined as: \f$ C(x_1 , \cdots ,x_k )=\f$ \psi^{-1}( \psi(x_1) + \cdots + \psi(x_k)) \f$

	If you construct multiple instances of this class, to avoid the generated samples to be the same, you should supply a different seed. To do so, for example, you can call `MyDistribution.SetRandomSeed(MyDistribution.GetCurrentSeed()+1U);`

	Please refer to the \ref examplesPage page for usage examples.

	\remark This class is re-entrant
	\date November 2013
	\license This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU Lesser General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.<br><br>
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU Lesser General Public License for more details.<br><br>
	Here, you can find a copy of the \ref LicensePage.
	Alternatively, see [gnu.org](http://www.gnu.org/licenses/).
	*/
	class ClaytonCopula : public AbstractArchimedean{
	private:
		double GeneratorFunction(double x)const;
		double GeneratorInverseFunction(double x)const;
		double GeneratorFunctionDerivative(double x)const;
		double GeneratorInverseFunctionDerivative(double x)const;
		double SimulateGeneratorInverseFourier()const;
		bool CheckValidity();
	public:
		//! Construct a Clayton copula with parameter 1
		/*!
		\param Dimension The dimensionality of the copula
		\details Construct a Clayton copula distribution with unitary dependence parameter
		
	
		In case the Dimension is less than 2 the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
		*/
		ClaytonCopula(unsigned int Dimension);
		//! Construct a Clayton copula with the given parameters
		/*!
		\param Dimension The dimensionality of the copula
		\param theta The dependence parameter.
		\details Construct a Clayton copula distribution with the dependence parameter theta.
		
		\f$ \theta=\frac{2 \tau}}{1- \tau} \f$ where \f$ \tau \f$ is the [Kendall's tau](http://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient) statistic
	
		In case:
		- The Dimension is less than 2
		- theta is less or equal to 0
	
		The class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
		*/
		ClaytonCopula(unsigned int Dimension,double theta);
		//! Set the dependence parameter
		/*!
		\param t The new dependence parameter
		\return A boolean that indicates if the parameter was changed successfully
		\details This function tries to set the dependence parameter of the distribution to the new value
		
		If the parameter is less or equal to 0 the function will return false and the parameter will not be changed.
		\sa SetKendallTau()
		*/
		bool SetTheta(double t){if(Theta>0.0) {Theta=t; CheckValidity(); return true;} return false;}
		//! Set the dependence parameter through the Kendall's tau
		/*!
		\param t The Kendall's tau statistic
		\return A boolean that indicates if the parameter was changed successfully
		\details This function tries to set \f$ \theta \f$, the dependence parameter of the distribution,  according to \f$ \theta = \frac{2 \tau}}{1- \tau}\f$
		
		If the parameter tau is not within the interval (0,1) the function will return false and the parameter will not be changed.
		\sa SetTheta()
		*/
		bool SetKendallTau(double t);
		//! Get the implied Kendall's tau
		/*!
		\return The value of the Kendall's tau of the distribution
		\details This function returns the value of the Kendall's tau statistic implied by the dependence parameter according to the relation \f$ \tau = \frac{\theta}}{2+ \theta} \f$
		
		If the distribution is invalid -1.0 is returned;
		\sa SetKendallTau()
		*/
		double GetKendallTau() const {if(AllValid) return Theta/(Theta+2.0); return -1.0;}
		//! Computes the inverse copula function in correspondence of the supplied probability
		/*!
		\param Prob The probability for which the corresponding quantile must be found
		\return A vector containing the coordinates of the quantile in the intervall [0;1]
		\details This function computes the inverse cumulative density function of the current distribution associated with the given probability.
	
		The solution is not unique.<br>
		Generally the system of equations \f$ C^{-1}(Coordinates_1 \cdots Coordinates_k)=Prob \f$ has k-1 degrees of freedom, where k is the dimensionality of the distribution.<br>
		The additional restriction imposed to get to an unique solution is that all the coordinates must be equal.

		If the coordinates supplied have any component that is greater than 1 or less than 0 or the distribution is invalid, an empty vector is returned.
		*/
		Eigen::VectorXd GetQuantile(double Prob)const;
		double GetLowerTailDependence() const;//! \todo Implement GetLowerTailDependence
		bool SetLowerTailDependence(double ltd);//! \todo Implement SetLowerTailDependence
		template <class F, class T> friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits);
		template <class F, class T>	friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);
		template <class F, class T> friend void boost::math::tools::detail::handle_zero_derivative(F f,T& last_f0,const T& f0,T& delta,T& result,T& guess,const T& min,const T& max);
	};
}
#endif // ClaytonCopula_h__