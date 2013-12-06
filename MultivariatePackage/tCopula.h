#ifndef tCopula_h__
#define tCopula_h__
#include "AbsaractCopula.h"
#include "tDist.h"
namespace Multivariate{
	//! Student's t Copula Distribution
	/*!
	\details This class provides the functionality of calculating the probability density value, cumulative probability density value, inverse cumulative probability density and generate random samples from a Student's t copula.

	Defining:
		- \f$ k \f$ as the dimensionality of the copula
		- \f$ v \f$ as the number of degrees of freedom of the copula
		- \f$ \t_v^{-1} \f$ as the inverse of the cumulative density function of the student's t distribution with \f$ v \f$ degrees of freedom
		- \f$ \t_{v , \rho} \f$ as the cumulative density function of the student's t distribution with \f$ v \f$ degrees of freedom, location \f$ \textbf{0} \f$ and scale matix \f$ \rho \f$ 
		- \f$ \boldsymbol{\Sigma}=\begin{bmatrix}
		\sigma^2_1 & \cdots & \sigma_{1,k}\\
		\vdots  & \ddots & \vdots  \\
		\sigma_{k,1} & \cdots & \sigma^2_k
		\end{bmatrix} \f$ as the scale matrix

		We can define \f$ \boldsymbol{\rho}(\boldsymbol{\Sigma})=\begin{bmatrix}
		1 & \cdots & \frac{\sigma_{1,k}}{\sigma_1 \sigma_k}\\
		\vdots  & \ddots & \vdots  \\
		\frac{\sigma_{k,1}}{\sigma_k \sigma_1} & \cdots & 1
		\end{bmatrix} \f$<br>
	The t copula distribution funtion is defined as: \f$ C(\textbf{x})=\t_{v , \rho}(\t_v^{-1}(u_1), \cdots , \t_v^{-1}(u_k)) \f$ 

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
	class tCopula : public AbstarctCopula{
	private:
		tDistribution *LocalVersion;
	public:
		//! Constructs a standard t copula
		/*!
		\param Dimension The dimensionality of the copula
		\details Construct a t copula with variance-covariance matrix set to the identity matrix.
	
		In case The Dimension less than 2 or the degrees of freedom are 0, the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.

		If dimension is unspecified, a bivariate copula with 1 degree of freedom  is constructed
		*/
		tCopula(unsigned int Dimension=2U, unsigned int DegFreedom=1U){BaseDist=new tDistribution(Dimension,DegFreedom); LocalVersion=static_cast<tDistribution*>(BaseDist);}
		//! Construct a t copula with the given parameters
		/*!
		\param Dimension The dimensionality of the copula
		\param DegFreedom The degrees of freedom of the copula
		\param ScalMatr The scale matrix
		\details Construct a t copula distribution.
		
		The scale matrix is standardized to have unitary diagonal

		In case:
		- The Dimension is less than 2
		- The degrees of freedom are 0
		- The scale matrix is not square
		- The scale matrix is not symmetric
		- The scale matrix is not semi-positive definite
		- The scale matrix has a number of rows different from the Dimension
	
		The class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
		*/
		tCopula(unsigned int Dimension, unsigned int DegFreedom,const Eigen::MatrixXd& ScalMatr);
		~tCopula(){delete BaseDist;}
		//! Set the scale matrix of the distribution
		/*!
		\param ScalMatr the new variance scale matrix of the distribution 
		\return A boolean determining if the scale matrix of the distribution was changed successfully
		\details This function tries to set the scale matrix of the distribution to the new one.
		
		The scale matrix is then standardized to have unitary diagonal

		In case:
		- The scale matrix is not square
		- The scale matrix is not symmetric
		- The scale matrix is not semi-positive definite
		- The scale matrix has a number of rows different from the Dimension

		The scale matrix of the distribution will not be changed and this function will return false

		\sa GetScaleMatrix()
		*/
		bool SetScaleMatrix(const Eigen::MatrixXd& ScalMatr);
		//! Set the scale matrix of the distribution
		/*!
		\param mVect a vector containing the elements of the new scale matrix of the distribution
		\param RowWise if it's set to true (the default) the matrix will be filled by row. If it's false it will be filled by columns
		\return A boolean determining if the scale matrix of the distribution was changed successfully
		\details This function tries to set the scale matrix of the distribution to the new one.

		Constructs a square matrix with number of rows equal to the dimensionality of the distribution, it is then filled with the values supplied in order according to the RowWise parameter

		The Variance-Covariance matrix is then standardized to have unitary diagonal

		In case:
		- The vector size is different from the square of the distribution dimensionality
		- The variance-covariance is not symmetric
		- The variance-covariance is not semi-positive definite
		- The variance-covariance has a number of rows different from the Dimension

		The variance covariance matrix of the distribution will not be changed and this function will return false

		\sa GetScaleMatrix()
		*/
		bool SetScaleMatrix(const std::vector<double>& mVect, bool RowWise=true);
		//! Get the standardized scale matrix of the distribution
		/*!
		\return The current scale matrix of the distribution
		\sa SetScaleMatrix(const Eigen::MatrixXd&)
		\sa SetScaleMatrix(const std::vector<double>&,bool)
		*/
		const Eigen::MatrixXd& GetScaleMatrix() const {return LocalVersion->GetScaleMatrix();}
		//! Set the degrees of freedom of the distribution
		/*!
		\param a The number of degrees of freedom of the distribution
		\return A boolean determining if the degrees of freedom of the distribution were changed successfully
		\details This function tries to set the degrees of freedom of the distribution to the new ones.

		In case a is less than 1 the degrees of freedom of the distribution will not be changed and this function will return false

		\sa GetDegreesOfFreedom()
		*/
		bool SetDegreesOfFreedom(unsigned int a);
		//! Get the degrees of freedom of the distribution
		/*!
		\return The current degrees of freedom of the distribution
		\sa SetDegreesOfFreedom()
		*/
		unsigned int GetDegreesOfFreedom() const {return LocalVersion->GetDegreesOfFreedom();}
		//! Generates multiple simulations from the copula
		/*!
		\param NumSamples The number of simulation to run
		\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
		\details This function generates NumSamples simulation from the current copula and returns them in matrix form.
	
		If NumSamples is 0 or the copula is invalid, a null matrix is returned
		*/
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples)const{if(GetDimension()>1U) return LocalVersion->ExtractSamplesCDF(NumSamples); return Eigen::MatrixXd();}
		//! Computes the copula density function in correspondence of the supplied coordinates
		/*!
		\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
		\return The value of the copula density function
		\details This function computes the probability density function of the current copula associated with the given coordinates.
		
		The coordinates must all be in the interval (0,1)

		If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned
		*/
		double GetDensity(const Eigen::VectorXd& Coordinates)const;
		//! Computes the copula function in correspondence of the supplied coordinates
		/*!
		\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
		\return The value of the copula function
		\details This function computes the probability density function of the current copula associated with the given coordinates.
		
		The coordinates must all be in the interval (0,1)

		If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned
		*/
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const;
		//! Computes the inverse copula function in correspondence of the supplied probability
		/*!
		\param Prob The probability for which the corresponding quantile must be found
		\return A vector containing the coordinates of the quantile in the intervall [0;1]
		\details This function computes the inverse cumulative density function of the current distribution associated with the given probability.
	
		The solution is not unique.<br>
		Generally the system of equations \f$ C^{-1}(Coordinates_1 \cdots Coordinates_k)=Prob \f$ has k-1 degrees of freedom, where k is the dimensionality of the distribution.<br>
		The additional restriction imposed to get to an unique solution is that each coordinate has equal distance from it's mean.

		If the coordinates supplied have any component that is greater than 1 or less than 0 or the distribution is invalid, an empty vector is returned.
		*/
		Eigen::VectorXd GetQuantile(double Prob)const;
		using Multivariate::AbstarctCopula::GetDensity;
		using Multivariate::AbstarctCopula::GetCumulativeDesity;
	};
}
#endif // tCopula_h__