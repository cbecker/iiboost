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

#ifndef _MISC_UTILS_H_
#define _MISC_UTILS_H_

#include <vector>
#include <sstream>


// some macros to print float/double
template<typename T>
inline const char *  typeToString()
{
    return "UNKNOWN";
}

template<>
inline const char * typeToString<float>()
{
    return "Float";
}

template<>
inline const char * typeToString<double>()
{
    return "Double";
}


// sample M indexes from 0..(N-1)
static void sampleWithoutReplacement( unsigned M, unsigned N, std::vector<unsigned> *idxs )
{
    if (M > N)  M = N;

    unsigned  max = N-1;

    std::vector<unsigned> toSample(N);
    for (unsigned i=0; i < N; i++)
        toSample[i] = i;

    idxs->resize(M);

    for (unsigned i=0; i < M; i++)
    {
        const unsigned idx = (((unsigned long)rand()) * max) / RAND_MAX;
        (*idxs)[i] = toSample[idx];

        //printf("Idx: %d / %d\n", idx, toSample[idx]);

        toSample[idx] = toSample[max];
        max = max - 1;
    }
}


// dummy func to convert something to string
template<typename T>
inline static std::string xToString( const T &val )
{
    std::stringstream strStream;
    strStream << val;
    return strStream.str();
}

#endif