#ifndef tDist_h__
#define tDist_h__
#include "NormalDist.h"
namespace Multivariate{
	//! %Multivariate Student's t Distribution
	/*!
	\details This class provides the functionality of calculating the probability density value, cumulative probability density value and generate random samples from a multivariate Student's t distribution.

	Defining:
	- \f$ \Gamma() \f$ as the [Gamma function](http://en.wikipedia.org/wiki/Gamma_function)
	- \f$ k \f$ as the dimensionality of the distribution
	- \f$ v \f$ as the degrees of freedom of the distribution
	- \f$ \boldsymbol{\mu}=[\mu_1 \cdots \mu_k] \f$ as the location vector
	- \f$ \boldsymbol{\Sigma}=\begin{bmatrix}
	\sigma^2_1 & \cdots & \sigma_{1,k}\\
	\vdots  & \ddots & \vdots  \\
	\sigma_{k,1} & \cdots & \sigma^2_k
	\end{bmatrix} \f$ as the scale matrix

	The multivariate student's t distribution funtion is defined as: \f$ f(\textbf{x})= \frac{\Gamma(\frac{v+k}{2})}{\Gamma(\frac{v}{2}) v^{k/2} \pi^{k/2} |\boldsymbol{\Sigma}|^{1/2} (1+ \frac{1}{v} (\textbf{x}-\boldsymbol{\mu})' \boldsymbol{\Sigma}^{-1} (\textbf{x}-\boldsymbol{\mu}))^{\frac{v+k}{2}}} \f$

	The algorithm for cdf calculation is based on [A. Genz & F. Bretz (2002)](http://www.math.wsu.edu/faculty/genz/homepage)

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
	class tDistribution : public AbstarctDistribution{
	private:
		unsigned int DegreesOfFreedom;
		tDistribution(const tDistribution& a);
		tDistribution& operator=(const tDistribution& a);
		Eigen::VectorXd LocatVect;
		Eigen::MatrixXd ScaleMatrix;
		bool CheckValidity();
		double ProbToFind;
		boost::math::tuple<double, double> operator()(double x);
		bool UseGenz;
		unsigned int NumSimul;
		std::pair<double,double> ComputeThat(unsigned int N)const;
	public:
		//! Construct a multivariate student's t with the given parameters
		/*!
		\param Dimension The dimensionality of the distribution (supports also univariate student's t distributions in case this is 1)
		\param DegFreedom The degrees of freedom of the distribution
		\param locVect The location vector
		\param ScaleMatr The scale matrix
		\details Construct a multivariate student's t distribution.
	
		In case:
		- The Dimension is 0
		- The Degrees of freedom are 0
		- The location vector has a number of elements different from the Dimension
		- The ScaleMatr is not square
		- The ScaleMatr is not symmetric
		- The ScaleMatr is not semi-positive definite
		- The ScaleMatr has a number of rows different from the Dimension
	
		The class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
		*/
		tDistribution(unsigned int Dimension,unsigned int DegFreedom,const Eigen::VectorXd& locVect,const Eigen::MatrixXd& ScaleMatr);
		//! Construct a multivariate standardized student's t
		/*!
		\param Dimension The dimensionality of the multivariate normal (supports also univariate Gaussian distributions in case this is 1)
		\param DegFreedom The degrees of freedom of the distribution
		\details Construct a multivariate student's t distribution with all location parameters set to 0 and scale matrix set to the identity.
	
		In case the Dimension or the DegFreedom are 0 the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.

		If no argument is specified, a univariate standardized student's t with 1 degree of freedom, is created
		*/
		tDistribution(unsigned int Dimension=1U,unsigned int DegFreedom=1U);
		//! Set the expected values vector
		/*!
		\param mVect the vector of new values for the location vector
		\return A boolean determining if the location vector was changed successfully
		\details This function attempts to set the location vector to the new values supplied.
		
		If the degrees of freedom are more than 1, this vector is the mean vector
	
		If the dimension of the vector is different from the dimension of the distribution the location vector is not changed and false is returned
		\sa GetLocationVector()
		*/
		bool SetLocationVector(const Eigen::VectorXd& mVect){if(mVect.rows()!=Dim) return false; LocatVect=mVect; CheckValidity(); return true;}
		//! Set the location vector
		/*!
		\param mVect the vector of new values for the location vector
		\details This is an overloaded version of SetLocationVector(const Eigen::VectorXd&)
		*/
		bool SetLocationVector(const std::vector<double>& mVect);
		//! Set the dimensionality of the distribution
		/*!
		\param Dimension the new dimensionality of the distribution 
		\return A boolean determining if the dimensionality was changed successfully
		\details This function will try to change the dimensionality of the distribution (e.g. 2 for bivariate, 3 for trivariate, etc.)
	
		All the components of the location vector will be reset to 0 and the scale matrix will default to an identity matrix, degrees of freedom wil not be changed

		If the argument passed is 0 the dimensionality will not be changed and the function will return false

		\sa GetDimension()
		*/
		bool SetDimension(unsigned int Dimension);
		//! Set the scale matrix of the distribution
		/*!
		\param SclMatr the new scale matrix of the distribution 
		\return A boolean determining if the scale matrix of the distribution was changed successfully
		\details This function tries to set the scale matrix of the distribution to the new one.

		In case:
		- The scale matrix is not square
		- The scale matrix is not symmetric
		- The scale matrix is not semi-positive definite
		- The scale matrix has a number of rows different from the Dimension

		The scale matrix of the distribution will not be changed and this function will return false

		\sa GetScaleMatrix()
		*/
		bool SetScaleMatrix(const Eigen::MatrixXd& SclMatr);
		//! Set the scale matrix of the distribution
		/*!
		\param mVect a vector containing the elements of the new scale matrix of the distribution
		\param RowWise if it's set to true (the default) the matrix will be filled by row. If it's false it will be filled by columns
		\return A boolean determining if the scale matrix of the distribution was changed successfully
		\details This function tries to set the scale matrix of the distribution to the new one.

		Constructs square a matrix with number of rows equal to the dimensionality of the distribution, it is then filled with the values supplied in order according to the RowWise parameter

		In case:
		- The vector size is different from the square of the distribution dimensionality
		- The scale matrix is not symmetric
		- The scale matrix is not semi-positive definite
		- The scale matrix has a number of rows different from the Dimension

		The scale matrix of the distribution will not be changed and this function will return false

		\sa GetScaleMatrix()
		*/
		bool SetScaleMatrix(const std::vector<double>& mVect, bool RowWise=true);
		//! Set the degrees of freedom of the distribution
		/*!
		\param a The number of degrees of freedom of the distribution
		\return A boolean determining if the degrees of freedom of the distribution were changed successfully
		\details This function tries to set the degrees of freedom of the distribution to the new ones.

		In case a is less than 1 the degrees of freedom of the distribution will not be changed and this function will return false

		\sa GetDegreesOfFreedom()
		*/
		bool SetDegreesOfFreedom(unsigned int a);
		//! Get the location vector of the distribution
		/*!
		\return The current location vector of the distribution
		\sa SetLocationVector(const Eigen::VectorXd&)
		*/
		const Eigen::VectorXd& GetLocationVector() const {return LocatVect;}
		//! Get the scale matrix of the distribution
		/*!
		\return The current scale matrix of the distribution
		\sa SetScaleMatrix(const Eigen::MatrixXd&)
		*/
		const Eigen::MatrixXd& GetScaleMatrix() const {return ScaleMatrix;}
		//! Get the degrees of freedom of the distribution
		/*!
		\return The current degrees of freedom of the distribution
		\sa SetDegreesOfFreedom()
		*/
		unsigned int GetDegreesOfFreedom() const {return DegreesOfFreedom;}
		//! Generates multiple simulations from the distribution
		/*!
		\param NumSamples The number of simulation to run
		\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
		\details This function generates NumSamples simulation from the current distribution and returns them in matrix form.
	
		If NumSamples is 0 or the distribution is invalid, a null matrix is returned
		*/
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
		//! Extracts samples from the distribution and returns their marginal CDF
		/*!
		\param NumSamples The number of simulation to run
		\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
		\details This function generates NumSamples simulation from the current distribution, computes the marginal cumulative density function for each of them and returns them in matrix form.
	
		This function simulates extractions from a t copula

		If NumSamples is 0 or the distribution is invalid, a null matrix is returned
		 */
		Eigen::MatrixXd ExtractSamplesCDF(unsigned int NumSamples) const;
		//! Computes the probability density function of the distribution in correspondence of the supplied coordinates
		/*!
		\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
		\return The value of the probability density function
		\details This function computes the probability density function of the current distribution associated with the given coordinates.
	
		If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned
		*/
		double GetDensity(const Eigen::VectorXd& Coordinates)const;
		//! Computes the cumulative density function of the distribution in correspondence of the supplied coordinates
		/*!
		\param Coordinates A vector containing the coordinates of the point for which the cdf should be computed
		\return The value of the cumulative density function
		\details This function computes the cumulative density function of the current distribution associated with the given coordinates.

		If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned.
		*/
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const;
		//! Computes the variance matrix of the distribution
		/*!
		\return The variance matrix
		\details This function computes the variance matrix of the current distribution.

		Defining:
		- \f$ v \f$ as the number of degrees of freedom of the distribution
		- \f$ \boldsymbol{\Sigma}=\begin{bmatrix}
		\sigma^2_1 & \cdots & \sigma_{1,k}\\
		\vdots  & \ddots & \vdots  \\
		\sigma_{k,1} & \cdots & \sigma^2_k
		\end{bmatrix} \f$ as the scale matrix
		
		The variance matrix is defined as \f$ \frac{v}{v-2} \boldsymbol{\Sigma} \forall v>2 \f$

		If the degrees of freedom are less than 3 or the distribution is invalid, a null matrix is returned
		*/
		Eigen::MatrixXd GetVarMatrix() const;
		//! Computes the inverse cumulative density function of the distribution in correspondence of the supplied probability
		/*!
		\param Prob The probability for which the corresponding quantile must be found
		\return A vector containing the coordinates of the quantile
		\details This function computes the inverse cumulative density function of the current distribution associated with the given probability.
	
		The solution is unique only in the univariate case.<br>
		Generally the system of equations \f$ F^{-1}(Coordinates_1 \cdots Coordinates_k)=Prob \f$ has k-1 degrees of freedom, where k is the dimensionality of the distribution.<br>
		The additional restriction imposed to get to an unique solution is that each coordinate has equal distance from it's mean.

		If the probability supplied is greater than 1, less than 0 or the distribution is invalid, an empty vector is returned.
		*/
		Eigen::VectorXd GetQuantile(double Prob)const;

		using Multivariate::AbstarctDistribution::GetDensity;
		using Multivariate::AbstarctDistribution::GetCumulativeDesity;
		template <class F, class T> friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits);
		template <class F, class T>	friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);
		template <class F, class T> friend void boost::math::tools::detail::handle_zero_derivative(F f,T& last_f0,const T& f0,T& delta,T& result,T& guess,const T& min,const T& max);
	};
}
#endif // tDist_h__