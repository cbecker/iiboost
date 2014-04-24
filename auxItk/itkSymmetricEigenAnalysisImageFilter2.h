//////////////////////////////////////////////////////////////////////////////////
//																																							//
// Copyright (C) 2010 Engin Turetken																						//
//																																							//
// This program is free software: you can redistribute it and/or modify         //
// it under the terms of the version 3 of the GNU General Public License        //
// as published by the Free Software Foundation.                                //
//                                                                              //
// This program is distributed in the hope that it will be useful, but          //
// WITHOUT ANY WARRANTY; without even the implied warranty of                   //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             //
// General Public License for more details.                                     //
//                                                                              //
// You should have received a copy of the GNU General Public License            //
// along with this program. If not, see <http://www.gnu.org/licenses/>.         //
//                                                                              //
// Contact <engin.turetken@epfl.ch> for comments & bug reports                  //
//////////////////////////////////////////////////////////////////////////////////

#ifndef __itkSymmetricEigenAnalysisImageFilter2_h
#define __itkSymmetricEigenAnalysisImageFilter2_h

#include <itkImageToImageFilter.h>
#include <itkDataObject.h>
#include <itkFixedArray.h>
#include <itkMatrix.h>
#include <itkSymmetricEigenAnalysis.h>


namespace itk {
	
	/** \class SymmetricEigenAnalysisImageFilter2
	 * \brief Computes eigen values and eigen vectors for each pixel  
	 *        of an input image.
	 *
	 *				Unlike the SymmetricEigenAnalysisImageFilter class, which 
	 *				computes only the eigenvalues, this filter also computes 
	 *				the eigenvector matrix for each pixel (of type symmetric 
	 *				matrix) of a given image. Furthermore, instead of the 
	 *				full eigenvector matrix, only the eigenvector corresponding 
	 *				to the first eigenvalue (in the ordered list) can be retained.
	 *
	 *				Input pixel type represents a  symmetric matrix and must 
	 *				provide write access for the [][] operator. Similarly, the output
	 *				eigenvalue, eigenvector and the first eigenvector images must 
	 *				provide write access for the [], [][] and [] operators respectively.
	 *
	 * \author Engin Turetken
	 * 
	 */
	template<class TInputImage, class TEigenValueImage, 
			class TEigenVectorImage = Image<Matrix<typename TInputImage::PixelType::ValueType, 
													TInputImage::ImageDimension, 
													TInputImage::ImageDimension>, 
											TInputImage::ImageDimension>, 
			class TFirstEigenVectorOrientImage = Image<FixedArray<typename TInputImage::PixelType::ValueType, 
																	TInputImage::ImageDimension>, 
														TInputImage::ImageDimension> >
	class ITK_EXPORT SymmetricEigenAnalysisImageFilter2:
		public ImageToImageFilter<TInputImage,TEigenValueImage>
	{
	public:
		/** Standard class typedefs. */
		typedef SymmetricEigenAnalysisImageFilter2									Self;
		typedef ImageToImageFilter<TInputImage,TEigenValueImage>		Superclass;
		typedef SmartPointer<Self>																	Pointer;
		typedef SmartPointer<const Self>														ConstPointer;
		
		/** Smart Pointer type to a DataObject. */
		typedef DataObject::Pointer									DataObjectPointer;
		
		typedef TInputImage																					InputImageType;
		typedef TEigenValueImage																		EigenValueImageType;
		typedef TEigenVectorImage																		EigenVectorImageType;
		typedef TFirstEigenVectorOrientImage												FirstEigenVectorOrientImageType;

		typedef typename InputImageType::PixelType									InputPixelType;
		typedef typename InputPixelType::ValueType									InputValueType;
		typedef typename EigenValueImageType::PixelType							EigenValuePixelType;
		typedef typename EigenVectorImageType::PixelType						EigenVectorPixelType;
		typedef typename FirstEigenVectorOrientImageType::PixelType	EigenVectorOrientPixelType;
		
		typedef SymmetricEigenAnalysis<InputPixelType, 
										EigenValuePixelType, 
										EigenVectorPixelType>												EigenAnalysisModuleType;
		
		/** RegionType only depends on the dimension of the image and not the pixel type. */
		typedef typename EigenValueImageType::RegionType						OutputImageRegionType;
		
		itkStaticConstMacro(InputImageDimension, unsigned int, InputImageType::ImageDimension);
		
		/** Run-time type information (and related methods).   */
		itkTypeMacro( SymmetricEigenAnalysisImageFilter2, ImageToImageFilter );
		
		/** Method for creation through the object factory. */
		itkNewMacro(Self);
		
		
		/** 
		 * Set/Get methods to order the eigen values in ascending order.
		 * This is the default. ie lambda_1 < lambda_2 < ....
		 */
		void SetOrderEigenValues( bool bOrderEigenValues );
		bool GetOrderEigenValues() const;
		
		/** 
		 * Set/Get methods to order the eigen value magnitudes in ascending order.
		 * In other words, |lambda_1| < |lambda_2| < .....
		 */
		void SetOrderEigenMagnitudes( bool bOrderEigenMagnitudes );
		bool GetOrderEigenMagnitudes() const;
		

		/** 
		 * Set the dimension N of the input matrix/tensor. This method does not need 
		 * be called if the dimension of the image is same as the size of the input 
		 * image pixel matrices. 
		 */
		void SetInputPixelMatrixSize( unsigned int unMatrixSize );
		
		/** 
		 * Get the dimension of the input matrix/tensor. If the user does not call 
		 * the SetInputPixelMatrixSize method, the default value is returned which is 
		 * the dimension of the input image.
		 */
		unsigned int GetInputPixelMatrixSize() const;
		
		
		/** 
		 * Methods to turn on/off generation of the eigenvector image with the pixels set 
		 * as the NxN eigenvector matrix, where N is the size of the input pixel matrix. 
		 * By default, the filter generates the eigenvector image. 
		 */
		itkSetMacro(GenerateEigenVectorImage,bool);
		itkGetConstMacro(GenerateEigenVectorImage,bool);
		itkBooleanMacro(GenerateEigenVectorImage);
		
		/** 
		 * Methods to turn on/off generation of the eigenvector (i.e., orientation) 
		 * image with pixels set as the first eigenvector (i.e., the one that 
		 * corresponds to the first eigenvalue in the ordered eigenvalue list), 
		 * where N is the size of the input pixel matrix. By default, the filter 
		 * does not generate this image. 
		 */
		itkSetMacro(GenerateFirstEigenVectorOrientImage,bool);
		itkGetConstMacro(GenerateFirstEigenVectorOrientImage,bool);
		itkBooleanMacro(GenerateFirstEigenVectorOrientImage);
		
		/** Set/Get the input image of this filter. */
		virtual void SetInput( const InputImageType *image);
		const InputImageType* GetInput() const;
						
		/** Get the eigenvalue image. */
		const EigenValueImageType* GetEigenValueImage() const;

		/** Get the eigenvector image. For each pixel matrix, each row represents an 
		 * eigenvector. */
		const EigenVectorImageType* GetEigenVectorImage() const;
		
		/** Get the first eigenvector orientation image. */
		const FirstEigenVectorOrientImageType* GetFirstEigenVectorOrientImage() const;
		
		/**  Overloaded function to create the output images. */
		virtual DataObjectPointer MakeOutput(ProcessObject::DataObjectPointerArraySizeType idx);

		/**  Overloaded function to allocate the output images. */
		virtual void AllocateOutputs();

#ifdef ITK_USE_CONCEPT_CHECKING
		/** Begin concept checking */
		itkConceptMacro(InputHasNumericTraitsCheck,
										(Concept::HasNumericTraits<InputValueType>));
		/** End concept checking */
#endif
		
	protected:
		SymmetricEigenAnalysisImageFilter2();
		virtual ~SymmetricEigenAnalysisImageFilter2(){};
		virtual void PrintSelf(std::ostream& os, Indent indent) const;
		
		/**  Overloaded function for threaded processing of the images. */
		void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
								  ThreadIdType threadId );
		
	private:
		SymmetricEigenAnalysisImageFilter2(const Self&); //purposely not implemented
		void operator=(const Self&);					 //purposely not implemented
		
		
		EigenAnalysisModuleType m_EigenAnalysisModule;
		
		bool m_GenerateEigenVectorImage;
		bool m_GenerateFirstEigenVectorOrientImage;
	};
	
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSymmetricEigenAnalysisImageFilter2.txx"
#endif

#endif
