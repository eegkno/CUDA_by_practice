#Makefile
# -L/usr/local/cuda/lib -lcurand	: es utilizado para enlazar la biblioteca de curand
# -L/usr/local/cuda/lib64 -lcurand	: en linux y el anterior en mac


SOURCE =  03random_curand.cu

CC = nvcc

exe: $(SOURCE)
	$(CC)  -lcurand  $(SOURCE)  -o exe


clean:
	$(RM) -rf exe 

run:
	./exe
