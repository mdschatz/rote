Introduction
************

Overview
========
|projectName| is an application programming interface (API) for distributed-memory 
computations relying on an elemental cyclic distributeion that can be viewed as an 
implementation of the ideas found here `<>`_.  These ideas can be viewed as generalizations 
of those behind the Elemental library for dense linear algebra extended to tensor computations.  
The Elemental library itself can be viewed as a careful combination of the following:

* A `PLAPACK <http://cs.utexas.edu/users/plapack>`_-like framework of matrix 
  distributions that are trivial for users to redistribute between.
* A `FLAME <http://cs.utexas.edu/users/flame>`_ approach to tracking 
  submatrices within (blocked) algorithms. 
* Element-wise distribution of matrices. One of the major benefits to this 
  approach is the much more convenient handling of submatrices, relative to 
  block distribution schemes.

The goal of |projectName| is to provide users an interface for quickly prototyping algorithms 
based on the above mentioned language for distributed-memory environments.

In conjunction with an algorithm derivation process, this work can be 
considered similar to that of RRR and the Cyclops Tensor Framework (CTF) `<>`_.

Dependencies
============
* Functioning C++ and ANSI C compilers.
* A working MPI2 implementation.
* BLAS implementations.
* `CMake <http://www.cmake.org>`_ (version 2.8.5 or later).

|projectName| should successfully build on nearly every platform, as it has been
verified to build on most major desktop platforms (including Linux, Mac OS X, 
Microsoft Windows, and Cygwin), as well as a wide variety of Linux clusters (including Blue Gene/Q).

License and copyright
=====================
Place copyright here
