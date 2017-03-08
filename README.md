[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.375673.svg)](https://doi.org/10.5281/zenodo.375673)

CUDA by practice
===================

Introduction
-------------
This is an introduction to learn CUDA. I used a lot of references to learn the basics about CUDA, all of them are included at the end. There is a pdf file that contains the basic theory to start programming in CUDA, as well as a source code to practice the theory explained and its solution.

-------------

Details
-------------

####Description of the exercise

During the presentation, there are some examples and practices. For the examples, it is just necessary to do what is described in the **Running the scripts** section. For the practices, it is required to implement some sections of the code.

The **_P** and the **_S** in the scripts' name mean *practice* and *solution* respectively. Try to complete the practice and compare it with the solution at the end.

It is not necessary to create the source code from scratch, the file contains certain sections that need to be completed. The following example shows a line with the sentence **// -:YOUR CODE HERE:- **. This indicates that you have to complete the code *just in that section* avoiding removing the other parts of the code.


```
//Do the add vector operation
int* add(int *a, int *b, int *result, int N){

	// -:YOUR CODE HERE:-
}
```

####Running the scripts


**NOTE**: All the codes have been tested in linux environments. The command **nvcc** is used to compile CUDA source files, it is similar to the command **gcc** used to compile *C* source codes.

```
// The '$' indicates the prompt in the command window in linux.

//1. Compile. The flag '-o' is used to create a execution file with the name "exe", the name can be changed.
$ nvcc 00add_S.cu -o exe

//2. Execute. './' is used to execute the program in the current directory.
$./exe

//3. Result.
-: successful execution :-
```
-------------

Installation of CUDA
-------------


>* [Windows](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html)
>* [OS X](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/index.html)
>* [Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html)

-------------

References
-------------

####Books
>* **CUDA by Example: An Introduction to General-Purpose GPU Programming**. Jason Sanders, Edward Kandrot
>* **CUDA Application Design and Development**. Rob Farber
>* **CUDA Programming: A Developer's Guide to Parallel Computing with GPUs (Applications of GPU Computing Series)**. Shane Cook
>* **Programming Massively Parallel Processors, Second Edition: A Hands- on Approach**. David B. Kirk , Wen-mei W. Hwu

####Courses
>* [Udacity:](https://www.udacity.com/course/cs344) Introduction to Parallel Programming
>* [Coursera:](https://www.coursera.org/course/hetero) Heterogeneous Parallel Programming

####Websites

>* [Dr. Dobbs: ](http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/207200659) CUDA, Supercomputing for the Masses
>* [Livermore Computing:](https://computing.llnl.gov/?set=training&page=index) High performance computing training
>* [Parallel for all:](http://devblogs.nvidia.com/parallelforall/) Nvidia developer zone

-------------

This document was last modified on : March. 7, 2017.
