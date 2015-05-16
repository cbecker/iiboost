project (ContextCueBoosting)

# we need v2.8 for ExternalProject
cmake_minimum_required (VERSION 2.8) 
include(ExternalProject)

## Needs ITK at least v4
find_package(ITK REQUIRED NO_DEFAULT_PATH)
include(${ITK_USE_FILE})

if( "${ITK_VERSION_MAJOR}" LESS 4 )
  message(FATAL_ERROR "ITK version 4 required")
endif()

link_directories( ${ITK_LIBRARY_DIRS} )

## RANDOMLIB
set(RANDOMLIB_DESTPATH "${CMAKE_BINARY_DIR}/randomlib")
if(MSVC)
    set(RANDOMLIB_LIB "${RANDOMLIB_DESTPATH}/lib/Random.lib")
else()
    set(RANDOMLIB_LIB "${RANDOMLIB_DESTPATH}/lib/libRandom.a")
endif()
set(RANDOMLIB_INCPATH "${RANDOMLIB_DESTPATH}/include")


externalproject_add( 
    randomlibBuild
    GIT_REPOSITORY http://git.code.sf.net/p/randomlib/code
    GIT_TAG "r1.6"
    CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
               "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
               "-DCMAKE_INSTALL_PREFIX=${RANDOMLIB_DESTPATH}"
               "-DCMAKE_BUILD_TYPE=RELEASE"
               "-DMAINTAINER=OFF"
               "-DRANDOM_SHARED_LIB=OFF"
               "-DCMAKE_C_FLAGS=-fpic"
               "-DCMAKE_CXX_FLAGS=-fpic"
               "-DDISABLE_BOOST=ON"
               "-DRANDOMLIB_DOCUMENTATION=OFF"
    UPDATE_COMMAND "" )

add_library(randomlib STATIC IMPORTED DEPENDS)
add_dependencies( randomlib randomlibBuild)
set_property(TARGET randomlib PROPERTY IMPORTED_LOCATION ${RANDOMLIB_LIB})


# Needed for log loss
#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /home/cjbecker/filer/bitbucket/liblbfgs-1.10/lib/.libs)

include_directories(${RANDOMLIB_INCPATH})

set(IIBOOST_USE_OPENMP TRUE CACHE BOOL "Enable and compile IIBoost with OpenMP")
set(OPENMP_FLAG "")
if(IIBOOST_USE_OPENMP)
    # Merely adding this flag causes gcc to define the _OPENMP preprocessor definition.
    if(MSVC)
        # FIXME: support for OpenMP on Windows needs some fixes in the code
        message(STATUS "OpenMP support on Windows not yet implemented, ignoring IIBOOST_USE_OPENMP.")
        # set(OPENMP_FLAG "-openmp")
    else()
        set(OPENMP_FLAG "-fopenmp")
    endif()
endif()

# Flags
# static release is pretty generic, possibly not the fastest one though
set(CMAKE_CXX_FLAGS_RELEASE_ALLSTATIC "-std=c++11 -O3 ${OPENMP_FLAG}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE_ALLSTATIC "-static-libgcc -static-libstdc++ -static")


# C++11 with ITK
add_definitions(-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0 -DVCL_CAN_STATIC_CONST_INIT_INT=0)

if(MSVC)
    # NOTE: The slic library uses a preprocessor flag "WINDOWS". Do NOT set
    #       this flag for Visual Studio because it breaks everything
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_DEPRECATE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG}")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3 ${OPENMP_FLAG}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -msse2 -msse3 -mssse3 -g ${OPENMP_FLAG}")
endif()

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

include_directories( ${ContextCueBoosting_SOURCE_DIR} )
include_directories( ${ContextCueBoosting_SOURCE_DIR}/extra )
include_directories( ${ContextCueBoosting_SOURCE_DIR}/deps/eigen )

# RAPIDJSON, and make assert throw exception rather than raw assert
#  TODO: may need correction to include <stdexcept>
include_directories( ${ContextCueBoosting_SOURCE_DIR}/deps/rapidjson/include )
add_definitions( "-D\"RAPIDJSON_ASSERT(x)=if(!(x)) throw std::logic_error(\\\"rapidjson exception\\\");\"" )

# Common libs needed for linking
set(COMMON_LIBS ${ITK_LIBRARIES} randomlib)

if(NOT MSVC)
    set(COMMON_LIBS ${COMMON_LIBS} stdc++)  # not sure why we need stdc++ here
endif()

add_executable( test test.cpp slic/LKM.cpp slic/utils.cpp )
if (APPLE OR MSVC)
    target_link_libraries( test ${COMMON_LIBS} )
else()
    target_link_libraries( test ${COMMON_LIBS} rt)
endif()

#
# Set a variable with CMake code which:
# Creates a symlink from src to dest (if possible) or alternatively
# copies if different.
macro(_create_symlink TARGET SRC DEST)
  if(WIN32 AND NOT CYGWIN)
    message(STATUS "Will copy ${SRC} to ${DEST} if needed")
    add_custom_command( TARGET ${TARGET} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${SRC}\" \"${DEST}\")
  else()
    message(STATUS "Will symlink ${SRC} to ${DEST}")
    #if you don't want silent replacement, comment out the remove line
    add_custom_command( TARGET ${TARGET} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E remove \"${DEST}\"
                        COMMAND ${CMAKE_COMMAND} -E create_symlink \"${SRC}\" \"${DEST}\")
  endif()

endmacro()


# If you want to comile the python wrapper, for Ilastik point PYTHON_BASE_PATH it to where Ilastik is
option(BUILD_PYTHON_WRAPPER "Build the python wrapper" ON)
if (BUILD_PYTHON_WRAPPER)
    add_subdirectory(python)
endif()

## extra tests
if(EXISTS "${ContextCueBoosting_SOURCE_DIR}/mytest.cpp")
    add_executable( mytest mytest.cpp slic/LKM.cpp slic/utils.cpp )
    target_link_libraries( mytest ${COMMON_LIBS} )
endif()
