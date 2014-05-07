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

#ifndef _ROI_DATA_H_
#define _ROI_DATA_H_

#include <Eigen/Dense>
#include <vector>

#include "globaldefs.h"

#include <Matrix3D.h>
#include "IntegralImage.h"

#if USE_SUPERVOXELS
    #include <slic/SuperVoxeler.h>
#endif

#include "auxItk/AllEigenVectorsOfHessian.h"

/**
 * Holds information for each Region of Interest, namely:
 *      - Rotation matrices
 *      - Raw image
 *      - Ground truth image
 *      - Integral images / channels
 */
struct ROIData
{
private:
    // rotation matrices, as Itk matrices, one per channel
    typedef AllEigenVectorsOfHessian::EigenVectorImageType  ItkEigenVectorImageType;
    ItkEigenVectorImageType::Pointer  rotMatricesImg;

public:

#if USE_SUPERVOXELS
    SuperVoxeler<ImagePixelType>     SVox;
    std::vector<UIntPoint3D>         svCentroids;
#endif

    typedef IntegralImage<IntegralImagePixelType> IntegralImageType;

    // each integral image channel
    std::vector< IntegralImageType > integralImages;

    // for now we need this to compute rotMatrices
    Matrix3D<ImagePixelType>  rawImage;

    // ground truth
    // see GTIgnoreLabel, etc in globaldefs.h
    Matrix3D<GTPixelType>     gtImage;


    // ITK stores matrices in row major
    typedef Eigen::Matrix<float,3,3,Eigen::RowMajor>     RotationMatrixType;

    // this will point to rotMatricesImg's data
    const RotationMatrixType *rotMatrices;


//////////////////////////////////////////////////////////////////////////////////////

    // computes rotation matrices from raw image
    void updateRotationMatrices( const float rotHessianSigma, const float zAnisotropyFactor )
    {
        // compute rotation matrix
        rotMatricesImg = 
            AllEigenVectorsOfHessian::allEigenVectorsOfHessian<ImagePixelType>( 
                rotHessianSigma, zAnisotropyFactor, rawImage.asItkImage(), 
                AllEigenVectorsOfHessian::EByMagnitude );

        // convert pointers
        rotMatrices = (const RotationMatrixType *) rotMatricesImg->GetPixelContainer()->GetImportPointer();
    }

    // initialize from images, using move semantics
    void init( Matrix3D<ImagePixelType> &&rawImg, 
               Matrix3D<ImagePixelType> &&gtImg, 
               const float rotHessianSigma = 3.5,
               const float zAnisotropyFactor = 1.0 )
    {
        // in case we had other info before
        freeIntegralImages();

        rawImage = std::move(rawImg);
        gtImage = std::move(gtImg);

        updateRotationMatrices( rotHessianSigma, zAnisotropyFactor );
    }

    // initialize from pointers, which must remain valid
    void init( ImagePixelType *rawImgPtr, 
               ImagePixelType *gtImgPtr, 
               IntegralImagePixelType **intImgPtr,
               unsigned numII,
               unsigned width, unsigned height, unsigned depth,
               const float rotHessianSigma = 3.5,
               const float zAnisotropyFactor = 1.0 )
    {
        // in case we had other info before
        freeIntegralImages();

        rawImage.fromSharedData( rawImgPtr, width, height, depth );
        gtImage.fromSharedData( gtImgPtr, width, height, depth );

        updateRotationMatrices( rotHessianSigma, zAnisotropyFactor );

        for (unsigned i=0; i < numII; i++)
        {
            integralImages.push_back( IntegralImageType() );
            integralImages.back().fromSharedData( intImgPtr[i],
                                                   width, 
                                                   height,
                                                   depth );
        }
    }

    // it won't free pointer data
    void addII( IntegralImagePixelType *iiDataPtr )
    {
        integralImages.push_back( IntegralImageType() );
        integralImages.back().fromSharedData( iiDataPtr,
                                               rawImage.width(),
                                               rawImage.height(),
                                               rawImage.depth() );
    }

    // to be used with std::move(), we love move semantics
    void addII( IntegralImageType &&iiDataPtr )
    {
        integralImages.push_back( std::move(iiDataPtr) );
    }

    void freeIntegralImages()
    {
        integralImages.clear();
    }

    ~ROIData() 
    {
        freeIntegralImages();
    }
    
};

#endif