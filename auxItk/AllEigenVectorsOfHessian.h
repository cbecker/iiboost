//////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013 Carlos Becker                                             //
// Ecole Polytechnique Federale de Lausanne                                     //
// Contact <carlos.becker@epfl.ch> for comments & bug reports                   //
//                                                                              //
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
//////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <itkImage.h>
#include <itkVectorImage.h>
#include <itkIndex.h>
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"

#include "itkNthElementImageAdaptor.h"

#include "SymmetricEigenAnalysisTest.h"

#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricEigenAnalysisImageFilter2.h"

#include "itkComposeImageFilter.h"
#include "itkMatrixIndexSelectionImageFilter.h"

// Macro to avoid re-typing with itk
// --> For instance: 
//			makeNew( random, itk::RandomImageSource<FloatImage2DType> );
// is equivalent to:
//			itk::RandomImageSource<FloatImage2DType>::Pointer	random;
//			random = itk::RandomImageSource<FloatImage2DType>::New();
// The __VA_ARGS__ is there so that it can handle commas within the argument list 
//   in a natural way
#define makeNew(instanceName, ...)	\
    __VA_ARGS__::Pointer instanceName = __VA_ARGS__::New()

namespace AllEigenVectorsOfHessian
{
    //Image
    const int Dimension = 3;
    typedef float                                     PixelType;
    typedef itk::Image<PixelType, Dimension>          ImageType;
    typedef ImageType::IndexType                      IndexType;
    typedef itk::ImageFileReader< ImageType >         ReaderType;

    typedef itk::FixedArray<float, Dimension>         OrientationPixelType;
    typedef itk::Image<OrientationPixelType, Dimension>
    OrientationImageType;
    typedef itk::ImageFileReader< OrientationImageType >     OrientationImageReaderType;
    typedef itk::ImageFileWriter< ImageType >     FloatImageWriterType;


    typedef itk::ImageRegionConstIterator< OrientationImageType > ConstOrientationIteratorType;
    typedef itk::ImageRegionIterator< OrientationImageType>       OrientationIteratorType;

    typedef itk::ImageRegionConstIterator< ImageType > ConstFloatIteratorType;
    typedef itk::ImageRegionIterator< ImageType>       FloatIteratorType;


    /** Hessian & utils **/
    typedef itk::SymmetricSecondRankTensor<float,Dimension>                       HessianPixelType;
    typedef itk::Image< HessianPixelType, Dimension >                             HessianImageType;
    typedef itk::HessianRecursiveGaussianImageFilter<ImageType, HessianImageType> HessianFilterType;

    typedef itk::Vector<float, Dimension>            VectorPixelType;

    typedef itk::Vector<float, Dimension>          EigenValuePixelType;
    typedef itk::Matrix<float, Dimension, Dimension>	EigenVectorPixelType;

    typedef itk::VectorImage<float, Dimension> VectorImageType;

    typedef itk::Image<EigenValuePixelType, Dimension> EigenValueImageType;
    typedef itk::Image<EigenVectorPixelType, Dimension> EigenVectorImageType;
    typedef EigenValueImageType 		FirstEigenVectorOrientImageType;


    typedef itk::ComposeImageFilter< ImageType >	ComposeFilterType;
    typedef itk::MatrixIndexSelectionImageFilter< EigenVectorImageType, ImageType >	MatrixIndexSelectionFilterType;


    typedef itk::SymmetricEigenAnalysisImageFilter2< HessianImageType, EigenValueImageType, EigenVectorImageType, FirstEigenVectorOrientImageType >        HessianToEigenFilter;

    using namespace std;

    #define showMsg(args...) \
        do { \
            printf("\x1b[32m" "\x1b[1m["); \
            printf(args); \
            printf("]\x1b[0m\n" ); \
        } while(0)

    enum WhichEigVec
    {
        EByMagnitude,
        EByValue
    };

    template<typename InputImageScalarType>
    EigenVectorImageType::Pointer allEigenVectorsOfHessian( float sigma, 
                                                       float zAnisotropyFactor,
                                                       const typename itk::Image<InputImageScalarType, 3>::Pointer origInpImg,
                                                       WhichEigVec whichEig )
    {
        typedef typename itk::Image<InputImageScalarType, 3>  OrigInputImageType;
        typedef typename itk::Image<float, 3>  InputImageType;

        typedef typename itk::CastImageFilter< OrigInputImageType, InputImageType > CastFilterType;

        typename CastFilterType::Pointer caster = CastFilterType::New();
        caster->SetInput( origInpImg );

        caster->Update();
        typename InputImageType::Pointer inpImg = caster->GetOutput();

        typename InputImageType::SpacingType spacing = inpImg->GetSpacing();
        typename InputImageType::SpacingType oldSpacing = spacing;

        spacing[2] *= zAnisotropyFactor;
        
        std::cout << "Using spacing: " << spacing << ", anisotr factor = " << zAnisotropyFactor << std::endl;
        inpImg->SetSpacing(spacing);
        
        showMsg("Hessian filtering");
        makeNew( hessianFilt, HessianFilterType );
        hessianFilt->SetSigma( sigma );
        hessianFilt->SetInput( inpImg );

        showMsg("Computing eigenvalues");
        // now compute eigenvalues/main eigenvector
        makeNew( eigenFilt, HessianToEigenFilter );
        
        eigenFilt->SetGenerateEigenVectorImage(true);
        eigenFilt->SetGenerateFirstEigenVectorOrientImage(false);
        
        // only sort by magnitude if highest magnitude is required
        eigenFilt->SetOrderEigenValues( whichEig == EByValue );
        eigenFilt->SetOrderEigenMagnitudes( whichEig == EByMagnitude );
        
        eigenFilt->SetInput( hessianFilt->GetOutput() );


        showMsg("Generating");
        {
            eigenFilt->Update();
            EigenVectorImageType::Pointer outImg = (EigenVectorImageType *) eigenFilt->GetEigenVectorImage();
                    
            // reset spacing
            spacing[0] = spacing[1] = spacing[2] = 1.0;
            
            outImg->SetSpacing( spacing );
            
            // restore (this is from old code)
            inpImg->SetSpacing(oldSpacing);

            return outImg;
        }
    }



    // int main(int argc, char **argv)
    // {
    //     cjbCheckSymmetricEigenAnalysisEigMagnitudeOrdering();

    //     if(argc != 6){
    //         printf("Usage: %s image sigma zAnisotropyFactor outputFile ordering\n", argv[0]);
    //         printf("ordering:\n");
    //         printf("\tBy magnitude: 1\n");
    //         printf("\tBy value: 0\n");
    //         exit(0);
    //     }

    //     string imageName(argv[1]);

    //     float sigma  = atof(argv[2]);
    //     float zAnisotropyFactor = atof(argv[3]);
        
    //     string outputFile(argv[4]);
        
    //     const int whichEigInt = atoi(argv[5]);
    //     WhichEigVec whichEig;
    //     switch(whichEigInt)
    //     {
    //         case 1:
    //             whichEig = EByMagnitude;
    // 			break;
    //         case 0:
    //             whichEig = EByValue;
    // 			break;
    // 		default:
    //             printf("\tError in 'order' parameter\n");
    // 			return -1;
    // 			break;
    // 	}


    // 	return execute(sigma, imageName, outputFile, whichEig, zAnisotropyFactor);
    // }
}
