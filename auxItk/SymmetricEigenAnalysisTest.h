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

#ifndef SYMMETRICEIGENANALYSISTEST_H
#define SYMMETRICEIGENANALYSISTEST_H

// this is a very simple and stupid test
//  to check whether there is a bug in SymmetricEigenAnalysis
//  affecting our results.

#include <itkSymmetricEigenAnalysis.h>
#include <itkMatrix.h>
#include <itkVector.h>
#include <iostream>

static bool cjbCheckSymmetricEigenAnalysisEigMagnitudeOrdering( bool displayResults = false, bool exitAppOnError = true )
{
    const unsigned int Dim = 3;

    typedef float   ScalarType;
    typedef itk::Matrix<ScalarType, Dim, Dim>  MatrixType;
    typedef itk::Vector<ScalarType, Dim>       VectorType;

    typedef itk::SymmetricEigenAnalysis<MatrixType, VectorType, MatrixType> EigenAnalysisType;

    MatrixType M;

    /*
       1     2     3
       2     1     4
       3     4     4
        Has 1 pos and 2 neg eigvalues
     */

    M(0,0) = 1; M(0,1) = 2; M(0,2) = 3;
    M(1,0) = 2; M(1,1) = 1; M(1,2) = 4;
    M(2,0) = 3; M(2,1) = 4; M(2,2) = 4;

    EigenAnalysisType eig;
    eig.SetDimension(Dim);

    VectorType eigVals[2];

    MatrixType eigVecs;

    eig.SetOrderEigenValues(false);
    eig.SetOrderEigenMagnitudes(true);

    eig.ComputeEigenValues( M, eigVals[0] );
    eig.ComputeEigenValuesAndVectors( M, eigVals[1], eigVecs );

    if (displayResults)
    {
        std::cout << "CheckSymmetricEigenAnalysis: " << eigVals[0] << std::endl;
        std::cout << "CheckSymmetricEigenAnalysis: " << eigVals[1] << std::endl;
    }

    for (unsigned i=0; i < 2; i++)
    {
        if ( (eigVals[i][0] > 0) || (eigVals[i][1] > 0) || ((eigVals[i][2] < 0)) )
        {
            std::cout << "Something is wrong at SymmetricAnalysis check. i = " << i << std::endl;
            std::cout << "CheckSymmetricEigenAnalysis: " << eigVals[0] << std::endl;
            std::cout << "CheckSymmetricEigenAnalysis: " << eigVals[1] << std::endl;

            if (exitAppOnError)
            {
                exit(-1);
                return false;
            }
        }
    }

    return true;
}

#endif // SYMMETRICEIGENANALYSISTEST_H
