import distutils, ctypes.util, shutil, os, sys
from distutils.core import setup

# copy the required DLLs to the directory $vigranumpy_tmp_dir/dlls
# if additional libraries are linked dynamically (e.g. tiff, png)
# they must be added to the list as well
dlls = ['@Boost_PYTHON_LIBRARY_RELEASE@',
        '@FFTW3_LIBRARY@',
        '@HDF5_Z_LIBRARY@',
        '@HDF5_SZ_LIBRARY@',
        '@HDF5_CORE_LIBRARY@',
        '@HDF5_HL_LIBRARY@']

for d in dlls:
    if not d:
        continue
    dll = ctypes.util.find_library(os.path.splitext(os.path.basename(d))[0])
    shutil.copy(dll, '@vigranumpy_tmp_dir@/dlls')
vigraimpex_dll='@VIGRAIMPEX_LOCATION@'.replace('$(OutDir)', 'release').replace('$(Configuration)', 'release')
shutil.copy(vigraimpex_dll, '@vigranumpy_tmp_dir@/dlls')
msvc_runtime = ctypes.util.find_library(ctypes.util.find_msvcrt())
shutil.copy(msvc_runtime, '@vigranumpy_tmp_dir@/dlls')

setup(name = 'iiboost',
      description = 'IIBoost',
      author = 'Carlos Becker',
      author_email = 'carlosbecker@gmail.com',
      url = 'https://github.com/cbecker/iiboost',
      license = 'GPLv3',
      version = '0.1',
      packages = ['iiboost'],
      package_dir = {'iiboost': 'iiboost'},
      package_data = {'iiboost': ['*.pyd', 'dlls/*.dll', 
                  'doc/vigra/*.*', 'doc/vigra/documents/*.*', 
                  'doc/vigranumpy/*.*', 'doc/vigranumpy/_static/*.*']})
