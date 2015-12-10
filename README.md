# CS205-finalproject

Authors: Sami Ghoche, George Lok, Devvret Rishi

## Quick Start

Assuming all requirements are met, to run our application, simply run "$python TFIDF/run_cython.py". This will automatically compile our main application code and run it.

Our application will run through our algorithm on 20,000 Stack Overflow questions, using configurations options specified by the prompts. At the end, it will then open a web browser to display a cluster our algorithm generated.

## Files

* util
  * omp_defs.pxd
    * Borrowed from CS205 PSET2 util code
    * Provides cython header for openmp locks
  * omp_defs.pyx
    * Borrowed from CS205 PSET2 util code
    * Provides cython implementation for openmp locks
  * omp_defs.pyxbld
    * Borrowed from CS205 PSET2 util code
    * Modified to explicitly use c compilation.
  * pyxbld_omp.py
    * Borrowed from CS205 PSET2 util code
    * Modifed to support C++ compilation
    * Provides functions for cython compilation.  Interfaces with pyxbld files.
  * set_compiler.py
    * Borrowed from CS205 PSET2 util code
    * See note below on compilers
  * timer.py
    * Borrowed from CS205 PSET2 util code
    * Provides timing wrapper for benchmarking
* TFIDF
  * AVX_cpp.h
    * based off AVX.h from PSET2
    * C++ shim for AVX, because of cython limitation with C++ compilation.
  * AVX_cpp.pxd
    * based off AVX.pxd
    * cython wrapper for AVX_cpp
  * avxintrin-emu.h
    * AVX emulation header provided by Intel.
  * run_cython.py
    * Main Application, runs our cython code to provide clusters of similar stack overflow questions and displays them in a browser.
  * run_numpy.py
    * Provided for benchmarking purposes, runs numpy version of algorithm
  * Serial.ipynb
    * Python notebook.  Contains miscellaneous code.  
    * Provided only for convenience, not necessary for application.
  * tfidf_cython.pyx
    * cython implementation of our application.
  * tfidf_cython.pyxbld
    * build file for our cython application
  * tfidf_cython_wrapper.py
    * Python wrapper which runs our cython implementation.
  * tfidf_numpy.py
    * Numpy/Python implementation of our application
  * xxhash.c
    * Fast hash algorithm by Yann Collet
    * Modified to allow easy compilation via cython
  * xxhash.h
    * Fast hash algorithm by Yann Collet
    * Modified to allow easy compilation via cython
* TFIDF/data
  * Preprocessed pickle files for our application.  Contains data for 20000 stackoverflow questions.
  * questions_sm.pkl
  * questionTexts_sm.pkl
  * wordIndices_sm.pkl

  
## Requirements

### Required Packages
* cython-0.23.x (lower versions may work if they support c++)

### Compiling
Our particular application uses C++ with cython.  The options set in util/pyxbld_omp.py:make_ext_cpp are tested to work with clang-omp on Mac OSX.  gcc is not tested.  

## Benchmarking

Running "$python TFIDF/run_cython.py" will provide benchmarks for the various stages of our algorithm.

As comparison, you can run "$python TFIDF/run_numpy.py" to run the numpy version.  Note the numpy version does not provide options for threading, AVX, or locking (as it doesn't use them) and does not provide clusters at the end.

