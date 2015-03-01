#Makefile

SOURCE = 08multidevice_openMP.cu

OPENMP_FLAGS = -Xcompiler -fopenmp

CC = nvcc  


exe: $(SOURCE) 
	$(CC) $(OPENMP_FLAGS) $(SOURCE)   -o exe


clean:
	$(RM) -rf $ exe 

run:
	./exe

