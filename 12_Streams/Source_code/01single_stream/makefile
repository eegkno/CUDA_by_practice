#Makefile

SOURCE =  01single_stream.cu

CC = nvcc  

exe: $(SOURCE)
	$(CC) $(SOURCE)  -o exe 


clean:
	$(RM) -rf exe 

run:
	./exe