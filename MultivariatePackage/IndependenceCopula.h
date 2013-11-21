#ifndef IndependenceCopula_h__
#define IndependenceCopula_h__
#include "AbstarctCopula.h"
#include "UniformDistribution.h"
namespace Multivariate{
	//! Independence Copula Distribution
	/*!
	\details This class provides the functionality of calculating the probability density value, cumulative probability density value, inverse cumulative probability density and generate random samples from an Independence copula.

	Defining:
		- \f$ k \f$ as the dimensionality of the copula
	
	The independence copula distribution funtion is defined as: \f$ C(\textbf{x})= \prod_{n=1}^k x_n \f$ for \f$ \textbf{x} \in \textbf{(0,1)} \f$

	This is quite a trivial copula but is supplied for completeness

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
	class IndependenceCopula : public AbstarctCopula{
	private:
		UniformDistribution LocalVersion;
	public:
		//! Constructs an Independence copula
		/*!
		\param Dimension The dimensionality of the copula
		\details Construct an Independence copula.
	
		In case The Dimension less than 2 the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.

		If dimension is unspecified, a bivariate copula is constructed
		*/
		IndependenceCopula(unsigned int Dimension=2U){BaseDist=new UniformDistribution(Dimension); LocalVersion=static_cast<UniformDistribution*>(BaseDist);}
		~IndependenceCopula(){delete BaseDist;}
		//! Generates multiple simulations from the copula
		/*!
		\param NumSamples The number of simulation to run
		\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
		\details This function generates NumSamples simulation from the current copula and returns them in matrix form.
	
		If NumSamples is 0 or the copula is invalid, a null matrix is returned
		*/
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples)const {if(GetDimension()>1U) return LocalVersion.ExtractSamplesCDF(NumSamples); return Eigen::MatrixXd();}
		//! Computes the copula density function in correspondence of the supplied coordinates
		/*!
		\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
		\return The value of the copula density function
		\details This function computes the probability density function of the current copula associated with the given coordinates.
		
		The coordinates must all be between 0 and 1

		If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned
		*/
		double GetDensity(const Eigen::VectorXd& Coordinates)const {if(GetDimension()>1U) return LocalVersion.GetDensity(Coordinates); return -1.0;}
		//! Computes the copula function in correspondence of the supplied coordinates
		/*!
		\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
		\return The value of the copula function
		\details This function computes the probability density function of the current copula associated with the given coordinates.
		
		The coordinates must all be between 0 and 1

		If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned
		*/
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const {if(GetDimension()>1U) return LocalVersion.GetCumulativeDesity(Coordinates); return -1.0;}
		//! Computes the inverse copula function in correspondence of the supplied probability
		/*!
		\param Prob The probability for which the corresponding quantile must be found
		\return A vector containing the coordinates of the quantile in the intervall [0;1]
		\details This function computes the inverse cumulative density function of the current distribution associated with the given probability.
	
		The solution is unique only in the univariate case.<br>
		Generally the system of equations \f$ F^{-1}(Coordinates_1 \cdots Coordinates_k)=Prob \f$ has k-1 degrees of freedom, where k is the dimensionality of the distribution.<br>
		The additional restriction imposed to get to an unique solution is that each coordinate has equal distance from it's mean.

		If the probability supplied is greater than 1, less than 0 or the distribution is invalid, an empty vector is returned.
		*/
		Eigen::VectorXd GetQuantile(double Prob)const {if(GetDimension()>1U) return LocalVersion.GetCumulativeDesity(Prob); return Eigen::VectorXd;}
		using Multivariate::AbstarctCopula::GetDensity;
		using Multivariate::AbstarctCopula::GetCumulativeDesity;
	};
}
#endif // IndependenceCopula_h__