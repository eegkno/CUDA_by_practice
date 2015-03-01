#Makefile

SOURCE =  03single_stream_chunks.cu

CC = nvcc  

exe: $(SOURCE)
	$(CC) $(SOURCE)  -o exe 


clean:
	$(RM) -rf exe 

run:
	./exe
	
