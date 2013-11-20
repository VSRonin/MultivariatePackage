#ifndef NormalDist_h__
#define NormalDist_h__
#include "AbstractDistribution.h"
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/roots.hpp>
//! Namespace where all the classes of the library are defined
namespace Multivariate{
//! Multivariate Normal Distribution
/*!
\details This class provides the functionality of calculating the probability density value, cumulative probability density value, inverse cumulative probability density and generate random samples from a multivariate normal.

Defining:
	- \f$ k \f$ as the dimensionality of the distribution
	- \f$ \boldsymbol{\mu}=[\mu_1 \cdots \mu_k] \f$ as the mean vector
	- \f$ \boldsymbol{\Sigma}=\begin{bmatrix}
	\sigma^2_1 & \cdots & \sigma_{1,k}\\
	\vdots  & \ddots & \vdots  \\
	\sigma_{k,1} & \cdots & \sigma^2_k
	\end{bmatrix} \f$ as the variance-covariance matrix

The multivariate normal distribution funtion is defined as: \f$ f(\textbf{x})=((2\pi)^{-\frac{k}{2}} |\boldsymbol{\Sigma}|^{-\frac{1}{2}} e^{(-\frac{1}{2} (\textbf{x}-\boldsymbol{\mu})' \boldsymbol{\Sigma}^{-1} (\textbf{x}-\boldsymbol{\mu}))} \f$

The analytical process for computing pdf and simulate from the distribution is based upon the [mvtnorm package](http://cran.r-project.org/web/packages/mvtnorm/index.html) for [R](http://www.r-project.org/) by Alan Genz, Frank Bretz, Tetsuhisa Miwa, Xuefei Mi, Friedrich Leisch, Fabian Scheipl, Bjoern Bornkamp, Torsten Hothorn
The algorithm for cdf calculation is based on [A. Genz (1992)](http://www.math.wsu.edu/faculty/genz/homepage)

If you construct multiple instances this class, to avoid the generated samples to be the same, you should supply a different seed. To do so, for example, you can call `MyDistribution.SetRandomSeed(MyDistribution.GetCurrentSeed()+1U);`

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
class NormalDistribution : public AbstarctDistribution{
private:
	Eigen::VectorXd meanVect;
	Eigen::MatrixXd VarCovMatrix;
	bool CheckValidity();
	NormalDistribution(const NormalDistribution& a);
	NormalDistribution& operator=(const NormalDistribution& a);
	std::vector<unsigned int> FillOrder(const Eigen::VectorXd& source)const;
	double ProbToFind;
	boost::math::tuple<double, double> operator()(double x);
	bool UseGenz;
	unsigned int NumSimul;
public:
	//! Get either the Genz algorithm is used or not
	/*!
	\return A bool determining if Genz algorithm will be used for calculating the CDF
	\sa SetUseGenz();
	*/
	bool GetUseGenz() const {return UseGenz;}
	//! Get the maximum number of simulations that will be used to compute the CDF
	/*!
	\return The maximum number of simulations
	\sa SetNumSimul();
	*/
	unsigned int GetNumSimul() const {return NumSimul;}
	//! Set if Genz algorithm should be used
	/*!
	\param a Determinses if Genz algorithm will be used
	\details If UseGenz is true, the algorithm used for computing the cumulative density function and the quantiles 
	is the one described in [Alan Genz - "Numerical Computation of Multivariate Normal Probabilities", Journal of Computational and Graphical Statistics, 1(1992), pp. 141-149](http://www.math.wsu.edu/faculty/genz/papers/mvn.pdf)

	If set to false, full monte-carlo simulation will be used (resulting in much slower performance)

	\remark By default Genz algorithm is used in CDF and quantiles calculations.
	\note The GetQuantile() function needs to calculate the CDF several times, using full monte-carlo may lead to very long execution times of that function
	\sa GetUseGenz()
	\sa SetNumSimul()
	*/
	void SetUseGenz(bool a=true){UseGenz=a;}
	//! Set the maximum number of simulations used in the CDF calculation
	/*
	\param a The new maximum number of simulations
	\return A bool determining if the number of simulations was changed successfully
	\details This functions tries to set the maximum number of simulation used when calculating cumulative density function to the new, supplied, value.
	
	If the argument is less than 1 the maximum number of simulations is not changed and false is returned

	If the Genz algorithm is used (the default) this is the maximum number of simulations that the algorithm will use.<br>
	If full monte-carlo is used then this is the number of simulations used when estimating the CDF
	\remark By default this parameter is set to 500000 that proved to give quite stable results for bivariate distributions using full monte-carlo
	\warning Setting this parameters to less than 1000 times the dimensionality of the distribution may lead to very bad and unstable CDF estimates
	\sa SetUseGenz()
	\sa GetNumSimul()
	*/
	bool SetNumSimul(unsigned int a);
	//! Construct a multivariate normal with the given parameters
	/*!
	\param Dimension The dimensionality of the multivariate normal (supports also univariate gaussian distributions in case this is 1)
	\param mVect The column vector of expected values
	\param CovMatr The variance-covariance
	\details Construct a multivariate normal distribution.
	
	In case:
	- The Dimension is 0
	- The mean vector has a number of elements different from the Dimension
	- The variance-covariance is not square
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension
	
	The class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
	*/
	NormalDistribution(unsigned int Dimension,const Eigen::VectorXd& mVect,const Eigen::MatrixXd& CovMatr);
	//! Construct a multivariate standard normal
	/*!
	\param Dimension The dimensionality of the multivariate normal (supports also univariate Gaussian distributions in case this is 1)
	\details Construct a multivariate normal distribution with all mean values set to 0, unitary variances and null covariates.
	
	In case The Dimension is 0 the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.

	If dimension is unspecified, a univariate standard normal is constructed
	*/
	NormalDistribution(unsigned int Dimension=1U);
	//! Set the mean vector
	/*!
	\param mVect the vector of new values for the mean vector
	\return A boolean determining if the mean vector was changed successfully
	\details This function attempts to set the mean vector to the new values supplied.
	
	If the dimension of the vector is different from the dimension of the distribution the mean vector is not changed and false is returned
	\sa GetMeanVector()
	*/
	virtual bool SetMeanVector(const Eigen::VectorXd& mVect);
	//! Set the expected values vector
	/*!
	\param mVect the vector of new values for the mean vector
	\details This is an overloaded version of SetMeanVector(const Eigen::VectorXd&)
	*/
	virtual bool SetMeanVector(const std::vector<double>& mVect);
	//! Set the dimensionality of the distribution
	/*!
	\param Dimension the new dimensionality of the distribution 
	\return A boolean determining if the dimensionality was changed successfully
	\details This function will try to change the dimensionality of the distribution (e.g. 2 for bivariate, 3 for trivariate, etc.)
	
	All the components of the mean vector will be reset to 0 and the variance covariance matrix will default to an identity matrix

	If the argument passed is 0 the dimensionality will not be changed and the function will return false

	\sa GetDimension()
	*/
	virtual bool SetDimension(unsigned int Dimension);
	//! Set the Var-Cov matrix of the distribution
	/*!
	\param CovMatr the new variance covariance matrix of the distribution 
	\return A boolean determining if the variance covariance matrix of the distribution was changed successfully
	\details This function tries to set the variance covariance matrix of the distribution to the new one.

	In case:
	- The variance-covariance is not square
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension

	The variance covariance matrix of the distribution will not be changed and this function will return false

	\sa GetVarCovMatrix()
	*/
	bool SetVarCovMatrix(const Eigen::MatrixXd& CovMatr);
	//! Set the Var-Cov matrix of the distribution
	/*!
	\param mVect a vector containing the elements of the new variance covariance matrix of the distribution
	\param RowWise if it's set to true (the default) the matrix will be filled by row. If it's false it will be filled by columns
	\return A boolean determining if the variance covariance matrix of the distribution was changed successfully
	\details This function tries to set the variance covariance matrix of the distribution to the new one.

	Constructs square a matrix with number of rows equal to the dimensionality of the distribution, it is then filled with the values supplied in order according to the RowWise parameter

	In case:
	- The vector size is different from the square of the distribution dimensionality
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension

	The variance covariance matrix of the distribution will not be changed and this function will return false

	\sa GetVarCovMatrix()
	*/
	bool SetVarCovMatrix(const std::vector<double>& mVect, bool RowWise=true);
	//! Set the Var-Cov matrix of the distribution
	/*!
	\param CorrelationMatrix The correlation coefficients matrix
	\param Variances the vector of variances
	\return A boolean determining if the variance covariance matrix of the distribution was changed successfully
	\details This function tries to set the variance covariance matrix of the distribution according to the correlation matrix supplied.

	Constructs the variance covariance matrix of the distribution using Variances as diagonal elements and covariates corresponding to the linear correlation coefficient supplied

	In case:
	- The size of the vector of variances is different from the distribution dimensionality
	- The correlation matrix is not symmetric
	- The correlation matrix has elements different from 1 on the diagonal
	- The correlation matrix has a number of rows different from the Dimension
	- The correlation matrix is not squared
	- The correlation matrix has off-diagonal elements greater than 1 in absolute value

	The variance covariance matrix of the distribution will not be changed and this function will return false

	\sa GetVarCovMatrix()
	*/
	bool SetVarCovMatrix(const Eigen::MatrixXd& CorrelationMatrix, const Eigen::VectorXd& Variances);
	
	//! Get the mean vector of the distribution
	/*!
	\return The current mean vector of the distribution
	\sa SetMeanVector(const Eigen::VectorXd&)
	*/
	const Eigen::VectorXd& GetMeanVector() const {return meanVect;}
	//! Get the Var-Cov matrix of the distribution
	/*!
	\return The current variance-covariance matrix of the distribution
	\sa SetVarCovMatrix(const Eigen::MatrixXd&)
	\sa SetVarCovMatrix(const std::vector<double>&,bool)
	\sa SetVarCovMatrix(const Eigen::MatrixXd&, const Eigen::VectorXd&)
	*/
	const Eigen::MatrixXd& GetVarCovMatrix() const {return VarCovMatrix;}
	//! Get the linear correlation matrix
	/*!
	\return The linear correlation matrix associated with the current variance-covariance matrix of the distribution
	\sa GetVarCovMatrix()
	*/
	Eigen::MatrixXd GetCorrelationMatrix() const;
	
	//! Generates multiple simulations from the distribution
	/*!
	\param NumSamples The number of simulation to run
	\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
	\details This function generates NumSamples simulation from the current distribution and returns them in matrix form.
	
	If NumSamples is 0 or the distribution is invalid, a null matrix is returned
	*/
	virtual Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
	
	//! Extracts samples from the distribution and returns their marginal CDF
	/*!
	\param NumSamples The number of simulation to run
	\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
	\details This function generates NumSamples simulation from the current distribution, computes the marginal cumulative density function for each of them and returns them in matrix form.
	
	This function simulates extractions from a Gaussian copula

	If NumSamples is 0 or the distribution is invalid, a null matrix is returned
	 */
	Eigen::MatrixXd ExtractSamplesCDF(unsigned int NumSamples) const;
	
	//! Computes the probability density function of the distribution in correspondence of the supplied coordinates
	/*!
	\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
	\param GetLogDensity If set to true the log density is returned instead of the actual density
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
	//! Computes the inverse cumulative density function of the distribution in correspondence of the supplied probability
	/*!
	\param Prob The probability for which the corresponding quantile must be found
	\return A vector containing the coordinates of the quantile
	\details This function computes the inverse cumulative density function of the current distribution associated with the given probability.
	
	The solution is unique only in the univariate case.<br>
	Generally the system of equations \f$ F^{-1}(Coordinates_1 \cdots Coordinates_Dimensions)=Prob \f$ has Dimensions-1 degrees of freedom.<br>
	The additional restriction imposed to get to an unique solution is that each coordinate has equal distance from it's mean.

	If the probability supplied is greater, in absolute value, than 1 or the distribution is invalid, an empty vector is returned.
	*/
	Eigen::VectorXd GetQuantile(double Prob)const;
	
#ifdef mvNormSamplerUnsafeMethods
	/** \name Unsafe Methods
	The methods in this group use unsafe memory access or return arrays allocated on the heap that must be manually deleted.

	These functions are normally not compiled for safety reasons. To use them, the mvNormSamplerUnsafeMethods symbol must be defined at compile time
	\{
	*/
	/**
	\brief Set the expected values vector
	\param mVect an array containing the new values for the expected values vector
	\details This is an overloaded version of SetMeanVector(const Eigen::VectorXd&)
	\warning This function will search for a number of elements equal to the dimensionality of the distribution in the array. This may mean accessing unallocated memory blocks if the supplied array is not big enough
	 */
	void SetMeanVector(double* mVect);
	/**
	\brief Set the Var-Cov matrix of the distribution
	\param mVect a matrix containing the new values for the variance covariance matrix of the distribution
	\details This is an overloaded version of SetVarCovMatrix(const Eigen::MatrixXd&)
	\warning This function will search for a number of elements, both in the row and the column dimension, equal to the dimensionality of the distribution in the matrix. This may mean accessing unallocated memory blocks if the supplied matrix is not big enough
	 */
	bool SetVarCovMatrix(double** mVect);
	/// \}
#endif
	using Multivariate::AbstarctDistribution::GetDensity;
	using Multivariate::AbstarctDistribution::GetCumulativeDesity;
	template <class F, class T> friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits);
	template <class F, class T>	friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);
	template <class F, class T> friend void boost::math::tools::detail::handle_zero_derivative(F f,T& last_f0,const T& f0,T& delta,T& result,T& guess,const T& min,const T& max);
};
//! Gaussian Copula Distribution
/*!
\details This class provides the functionality of calculating the probability density value, cumulative probability density value, inverse cumulative probability density and generate random samples from a Gaussian copula.

Defining:
	- \f$ k \f$ as the dimensionality of the copula
	- \f$ \boldsymbol{\Sigma}=\begin{bmatrix}
	\sigma^2_1 & \cdots & \sigma_{1,k}\\
	\vdots  & \ddots & \vdots  \\
	\sigma_{k,1} & \cdots & \sigma^2_k
	\end{bmatrix} \f$ as the variance-covariance matrix

The gaussian copula distribution funtion is defined as: \f$ f(\textbf{x})=((2\pi)^{-\frac{k}{2}} |\boldsymbol{\Sigma}|^{-\frac{1}{2}} e^{(-\frac{1}{2} \textbf{x}' \boldsymbol{\Sigma}^{-1} \textbf{x})} \f$

The analytical process for computing pdf and simulate from the distribution is based upon the [mvtnorm package](http://cran.r-project.org/web/packages/mvtnorm/index.html) for [R](http://www.r-project.org/) by Alan Genz, Frank Bretz, Tetsuhisa Miwa, Xuefei Mi, Friedrich Leisch, Fabian Scheipl, Bjoern Bornkamp, Torsten Hothorn
The algorithm for cdf calculation is based on [A. Genz (1992)](http://www.math.wsu.edu/faculty/genz/homepage)

If you construct multiple instances this class, to avoid the generated samples to be the same, you should supply a different seed. To do so, for example, you can call `MyDistribution.SetRandomSeed(MyDistribution.GetCurrentSeed()+1U);`

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
class GaussianCopula : public NormalDistribution{
private:
	bool SetMeanVector(const Eigen::VectorXd& mVect){return NormalDistribution::SetMeanVector(mVect);}
	bool SetMeanVector(const std::vector<double>& mVect){return NormalDistribution::SetMeanVector(mVect);}
public:
	//! Constructs a standard Gaussian copula
	/*!
	\param Dimension The dimensionality of the copula
	\details Construct a Gaussian copula with variance-covariance matrix set to the identity matrix.
	
	In case The Dimension less than 2 the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.

	If dimension is unspecified, a bivariate copula is constructed
	*/
	GaussianCopula(unsigned int Dimension=2U) : NormalDistribution(Dimension){AllValid=AllValid && Dimension>1U;}
	//! Construct a Gaussian copula with the given parameters
	/*!
	\param Dimension The dimensionality of the multivariate normal (supports also univariate gaussian distributions in case this is 1)
	\param CovMatr The variance-covariance
	\details Construct a multivariate normal distribution.
	
	In case:
	- The Dimension is less than 2
	- The variance-covariance is not square
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension
	
	The class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
	*/
	GaussianCopula(unsigned int Dimension,const Eigen::MatrixXd& CovMatr) : NormalDistribution(Dimension){AllValid=AllValid && Dimension>1U && SetVarCovMatrix(CovMatr);}
	//! Generates multiple simulations from the copula
	/*!
	\param NumSamples The number of simulation to run
	\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
	\details This function generates NumSamples simulation from the current copula and returns them in matrix form.
	
	If NumSamples is 0 or the distribution is invalid, a null matrix is returned
	*/
	Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const{return ExtractSamplesCDF(NumSamples);}
	//! Set the dimensionality of the distribution
	/*!
	\param Dimension the new dimensionality of the distribution 
	\return A boolean determining if the dimensionality was changed successfully
	\details This function will try to change the dimensionality of the distribution (e.g. 2 for bivariate, 3 for trivariate, etc.)
	
	The variance covariance matrix will default to an identity matrix

	If the argument passed is less than 2 the dimensionality will not be changed and the function will return false

	\sa GetDimension()
	*/
	bool SetDimension(unsigned int Dimension){if(Dimension>1U) return NormalDistribution::SetDimension(Dimension); else return false;}
};
} //namespace Multivariate
#endif // NormalDist_h__
