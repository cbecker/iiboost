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


#ifndef __itkSymmetricEigenAnalysisImageFilter2_txx
#define __itkSymmetricEigenAnalysisImageFilter2_txx

#include "itkSymmetricEigenAnalysisImageFilter2.h"

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkProgressReporter.h"


namespace itk
{
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::SymmetricEigenAnalysisImageFilter2()
	{
		m_GenerateEigenVectorImage = true;
		m_GenerateFirstEigenVectorOrientImage = false;
		
		m_EigenAnalysisModule.SetDimension(InputImageDimension);
		
		this->ProcessObject::SetNumberOfRequiredInputs(1);
		this->ProcessObject::SetNumberOfRequiredOutputs(3);
		this->ProcessObject::SetNthOutput(0, this->MakeOutput(0));
		this->ProcessObject::SetNthOutput(1, this->MakeOutput(1));
		this->ProcessObject::SetNthOutput(2, this->MakeOutput(2));	
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage> 
	typename SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>::DataObjectPointer 
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::MakeOutput(ProcessObject::DataObjectPointerArraySizeType idx)
	{
		switch( idx )
		{
			case 1:
				return (EigenVectorImageType::New()).GetPointer();
			case 2:
				return (FirstEigenVectorOrientImageType::New()).GetPointer();
			default:
				return (EigenValueImageType::New()).GetPointer();
		}
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage> 
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::AllocateOutputs()
	{
		typename EigenValueImageType::Pointer EigenValueImagePtr = 
			dynamic_cast<EigenValueImageType*>(this->ProcessObject::GetOutput(0));
		if( EigenValueImagePtr )
		{
			EigenValueImagePtr->SetBufferedRegion( EigenValueImagePtr->GetRequestedRegion() );
			EigenValueImagePtr->Allocate();
		}
		
		if( m_GenerateEigenVectorImage )
		{
			typename EigenVectorImageType::Pointer EigenVectorImagePtr = 
				dynamic_cast<EigenVectorImageType*>(this->ProcessObject::GetOutput(1));
			if( EigenVectorImagePtr )
			{
				EigenVectorImagePtr->SetBufferedRegion( EigenVectorImagePtr->GetRequestedRegion() );
				EigenVectorImagePtr->Allocate();
			}
		}
		
		if( m_GenerateFirstEigenVectorOrientImage )
		{
			typename FirstEigenVectorOrientImageType::Pointer FirstEigenVectorOrientImagePtr = 
				dynamic_cast<FirstEigenVectorOrientImageType*>(this->ProcessObject::GetOutput(2));
			if( FirstEigenVectorOrientImagePtr )
			{
				FirstEigenVectorOrientImagePtr->SetBufferedRegion( FirstEigenVectorOrientImagePtr->GetRequestedRegion() );
				FirstEigenVectorOrientImagePtr->Allocate();
			}
		}
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage> 
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf(os, indent);
		
		os << indent << "EigenAnalysisModuleType:  " << m_EigenAnalysisModule << std::endl;
		os << indent << "GenerateEigenVectorImage:  " << m_GenerateEigenVectorImage  << std::endl;
		os << indent << "GenerateFirstEigenVectorOrientImage:  " << m_GenerateFirstEigenVectorOrientImage  << std::endl;
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage> 
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::SetInput(const typename 
						SymmetricEigenAnalysisImageFilter2<TInputImage,
															TEigenValueImage,
															TEigenVectorImage,
															TFirstEigenVectorOrientImage>::InputImageType* pInputImage)
	{
		// Process object is not const-correct so the const_cast is required here.
		this->ProcessObject::SetNthInput(0, const_cast<InputImageType*>(pInputImage));
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage> 
	const typename SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>::InputImageType*
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetInput() const
	{		
		return static_cast<const InputImageType*>(this->ProcessObject::GetInput(0) );
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage> 
	const typename SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>::EigenValueImageType*
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetEigenValueImage() const
	{
		return static_cast<const EigenValueImageType*>(this->ProcessObject::GetOutput(0));
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	const typename SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>::EigenVectorImageType*
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetEigenVectorImage() const
	{
		return static_cast<const EigenVectorImageType*>(this->ProcessObject::GetOutput(1));
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	const typename SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>::FirstEigenVectorOrientImageType*
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetFirstEigenVectorOrientImage() const
	{
		return static_cast<const FirstEigenVectorOrientImageType*>(this->ProcessObject::GetOutput(2));
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::SetOrderEigenValues( bool bOrderEigenValues )
	{
		if( bOrderEigenValues != this->m_EigenAnalysisModule.GetOrderEigenValues() )
		{
			m_EigenAnalysisModule.SetOrderEigenValues(bOrderEigenValues);
			this->Modified();
		}
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	bool
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetOrderEigenValues() const
	{
		return m_EigenAnalysisModule.GetOrderEigenValues();
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::SetOrderEigenMagnitudes( bool bOrderEigenMagnitudes )
	{
		if( bOrderEigenMagnitudes != m_EigenAnalysisModule.GetOrderEigenMagnitudes() )
		{
			m_EigenAnalysisModule.SetOrderEigenMagnitudes(bOrderEigenMagnitudes);
			this->Modified();
		}
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	bool
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetOrderEigenMagnitudes() const
	{
		return m_EigenAnalysisModule.GetOrderEigenMagnitudes();
	}
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::SetInputPixelMatrixSize( unsigned int unMatrixSize )
	{
		if( unMatrixSize != this->m_EigenAnalysisModule.GetDimension() )
		{
			m_EigenAnalysisModule.SetDimension(unMatrixSize);
			this->Modified();
		}
	}

	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	unsigned int
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::GetInputPixelMatrixSize() const
	{
		return m_EigenAnalysisModule.GetDimension();
	}	
	
	
	template <class TInputImage, class TEigenValueImage, class TEigenVectorImage, class TFirstEigenVectorOrientImage>
	void
	SymmetricEigenAnalysisImageFilter2<TInputImage,TEigenValueImage,TEigenVectorImage,TFirstEigenVectorOrientImage>
	::ThreadedGenerateData(const typename 
								SymmetricEigenAnalysisImageFilter2<TInputImage,
								TEigenValueImage,
								TEigenVectorImage,
								TFirstEigenVectorOrientImage>::OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
	{
		// Image pointer and iterator definitions.
		typedef ImageRegionConstIterator<InputImageType>				InputImageIterator;
		typedef ImageRegionIterator<EigenValueImageType>				EigenValueImageIterator;
		typedef ImageRegionIterator<EigenVectorImageType>				EigenVectorImageIterator;
		typedef ImageRegionIterator<FirstEigenVectorOrientImageType>	FirstEigenVectorOrientImageIterator;
		typename InputImageType::Pointer InputImagePtr;
		typename EigenValueImageType::Pointer EigenValueImagePtr;
		typename EigenVectorImageType::Pointer EigenVectorImagePtr;
		typename FirstEigenVectorOrientImageType::Pointer FirstEigenVectorOrientImagePtr;
		const unsigned int nMatrixDim = this->GetInputPixelMatrixSize();
		
		// Get the input and output pointers.
		InputImagePtr = 
			static_cast<InputImageType*>(this->ProcessObject::GetInput(0));
		EigenValueImagePtr = 
			static_cast<EigenValueImageType*>(this->ProcessObject::GetOutput(0));
		
		if( m_GenerateEigenVectorImage )
		{
			EigenVectorImagePtr = 
				static_cast<EigenVectorImageType*>(this->ProcessObject::GetOutput(1));
		}
		
		if( m_GenerateFirstEigenVectorOrientImage )
		{
			FirstEigenVectorOrientImagePtr = 
				static_cast<FirstEigenVectorOrientImageType*>(this->ProcessObject::GetOutput(2));
		}

		// Create iterators for this thread's region.
		InputImageIterator		inIt(InputImagePtr, outputRegionForThread);
		EigenValueImageIterator	evalIt(EigenValueImagePtr, outputRegionForThread);
		EigenVectorImageIterator  evecIt;
		FirstEigenVectorOrientImageIterator fevecIt;
		if( m_GenerateEigenVectorImage )
		{
			evecIt = EigenVectorImageIterator(EigenVectorImagePtr, outputRegionForThread);
		}
		if( m_GenerateFirstEigenVectorOrientImage )
		{
			fevecIt = FirstEigenVectorOrientImageIterator(FirstEigenVectorOrientImagePtr, outputRegionForThread);
		}
		
		// Support progress methods/callbacks.
		ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
		
		
		inIt.GoToBegin();
		evalIt.GoToBegin();
		if( m_GenerateEigenVectorImage )
		{
			evecIt.GoToBegin();
		}
		if( m_GenerateFirstEigenVectorOrientImage )
		{
			fevecIt.GoToBegin();
		}
		
		// Traverse the image region.
		while( !evalIt.IsAtEnd() )
		{
			if( m_GenerateEigenVectorImage || m_GenerateFirstEigenVectorOrientImage )
			{
				EigenVectorPixelType EigenVector;
					m_EigenAnalysisModule.ComputeEigenValuesAndVectors(inIt.Value(), evalIt.Value(), EigenVector);

				// what to do if NaN or Inf is found
				if ( (!evalIt.Value().GetVnlVector().is_finite())
				      || ( !EigenVector.GetVnlMatrix().is_finite() ) )
				{
					evalIt.Value().Fill(0);
					EigenVector.SetIdentity();
				}
				
				if( m_GenerateEigenVectorImage )
				{					
					evecIt.Set(EigenVector);
				}
				
				if( m_GenerateFirstEigenVectorOrientImage )
				{
					EigenVectorOrientPixelType FirstEigenVector;
						
					for(unsigned int col = 0; col < nMatrixDim; col++ )
					{
						FirstEigenVector[col] = EigenVector[0][col];
					}
					
					fevecIt.Set(FirstEigenVector);
				}				
			}
			else
			{
				m_EigenAnalysisModule.ComputeEigenValues(inIt.Value(), evalIt.Value());

				// what to do if NaN or Inf is found
				if ( !evalIt.Value().GetVnlVector().is_finite() )
					evalIt.Value().Fill(0);
			}
			
			// Advance the iterators.
			++inIt;
			++evalIt;
			if( m_GenerateEigenVectorImage )
			{
				++evecIt;
			}
			if( m_GenerateFirstEigenVectorOrientImage )
			{
				++fevecIt;
			}
			
			progress.CompletedPixel();
		}
	}
} // end namespace itk

#endif
