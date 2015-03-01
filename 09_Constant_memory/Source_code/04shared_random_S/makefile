#Makefile

	
SOURCE =  04shared_random.cu

CURAND_LIB_PATH = -L/usr/local/cuda/lib -lcurand 

CC = nvcc

exe: $(SOURCE)
	$(CC)  $(CURAND_LIB_PATH)  $(SOURCE)  -o exe


clean:
	$(RM) -rf exe 

run:
	./exe

