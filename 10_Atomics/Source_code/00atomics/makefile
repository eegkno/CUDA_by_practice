#Makefile


SOURCE =  00atomics.cu
FLAGS = -arch=sm_11
CC = nvcc  

exe: $(SOURCE)
	$(CC) $(FLAGS) $(SOURCE)  -o exe 


clean:
	$(RM) -rf exe 

run:
	./exe




