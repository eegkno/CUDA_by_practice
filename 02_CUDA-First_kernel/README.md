02_CUDA-First_kernel
===================

Introduction
-------------
This is an introduction to learn CUDA. I used a lot of references to learn the basics about CUDA, all of them are included at the end. There is a pdf file that contains the basic theory to start programming in CUDA, as well as a source code to practice the theory explained and its solution.

-------------

List of files
-------------
> * **00_References.pdf** list of references used for the presentation.
> * **02_First kernel.pdf** theory to solve the practice.
> * **Source_code/01simple_kernel/01simple_kernel.cu**  example.
> * **Source_code/01simple_kernel/makefile**  to compile and to execute.
> * **Source_code/02simple_kernel2/01simple_kernel2.cu**  example.
> * **Source_code/02simple_kernel/makefile**  to compile and to execute..

-------------

Running the scripts
-------------

**NOTE**: All the codes have been tested in linux environments. A makefile is used to generate the execution file, you can see the makefile's description here [Wiki](http://en.wikipedia.org/wiki/Makefile) and here [GNU make](https://www.gnu.org/software/make/manual/make.html#Introduction).

```
// The '$' indicates the prompt in the command window in linux.

//1. Compile. 
$ make

//2. Execute. 
$make run

//3. Result.
./exe
Hello, World!
```
-------------

Installation
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
>* [Udacity:](https://www.udacity.com/course/cs344) Introduction to Parallel Programming.
>* [Coursera:](https://www.coursera.org/course/hetero) Heterogeneous Parallel Programming

####Websites

>* [Dr. Dobbs: ](http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/207200659) CUDA, Supercomputing for the Masses.
>* [Livermore Computing:](https://computing.llnl.gov/?set=training&page=index) High performance computing training
>* [Parallel for all:](http://devblogs.nvidia.com/parallelforall/) Nvidia developer zone

-------------


> Written with [StackEdit](https://stackedit.io/).
