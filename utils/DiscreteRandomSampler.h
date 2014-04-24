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

#ifndef DISCRETERANDOMSAMPLER_H
#define DISCRETERANDOMSAMPLER_H

#include <cstdlib>
#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include <RandomLib/Random.hpp>
#include <RandomLib/RandomSelect.hpp>

// class that implements weighted random sampling, with and without replacement
// (the version without replacement is a bit too expensive computationally)
template<typename VecType>
class DiscreteRandomSampler
{
private:
    RandomLib::Random               mRandGen;

public:
    DiscreteRandomSampler()
    {
        mRandGen.Reseed(87256);
    }

    DiscreteRandomSampler(unsigned int seed)
    {
        mRandGen.Reseed(seed);
    }

    void reSeed(unsigned int seed)
    {
        mRandGen.Reseed(seed);
    }
    
    void sampleWithReplacement( const VecType &weightVector, std::vector<unsigned int> &idxs, unsigned int N )
    {
		RandomLib::RandomSelect<double> mRandSelect( weightVector.data(), weightVector.data() + weightVector.rows() );
		
		idxs.resize( N );
		
		for (unsigned int i=0; i < N; i++)
			idxs[i] = mRandSelect(mRandGen);
	}

    void sampleWithoutReplacement( const VecType &weightVector, std::vector<unsigned int> &idxs, unsigned int N )
    {
        RandomLib::RandomSelect<double> mRandSelect( weightVector.data(), weightVector.data() + weightVector.rows() );

        std::vector<unsigned int>  alreadyFound( weightVector.rows(), false );

        idxs.resize( N );

        for (unsigned int i=0; i < N; i++)
        {
            unsigned int idx = 0;
            while (true)
            {
                idx = mRandSelect(mRandGen);

                if ( !alreadyFound.at(idx) )
                    break;
            }

            alreadyFound[idx] = true;

            idxs[i] = idx;
        }
    }
};

#endif // DISCRETERANDOMSAMPLER_H
