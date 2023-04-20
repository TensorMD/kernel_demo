CC=mpiicc
CFLAGS=-O2 -qopenmp -march=native -std=c99 -g
LDFLAGS=-qmkl=sequential
TARGET=kernel1 kernel2 kernel3 kernel4

.PHONY: all
all: $(TARGET)
	@echo built $(TARGET)

kernel1: kernel1.o utils.o
	$(CC) kernel1.o utils.o -o kernel1 $(CFLAGS) $(LDFLAGS)

kernel1.o: kernel1.c
	$(CC) -c kernel1.c $(CFLAGS) 

kernel2: kernel2.o utils.o
	$(CC) kernel2.o utils.o -o kernel2 $(CFLAGS) $(LDFLAGS)

kernel2.o: kernel2.c
	$(CC) -c kernel2.c $(CFLAGS) 

kernel3: kernel3.o calculate_dM.o utils.o
	$(CC) kernel3.o calculate_dM.o utils.o -o kernel3 $(CFLAGS) $(LDFLAGS)

kernel3.o: kernel3.c calculate_dM.c
	$(CC) -c kernel3.c $(CFLAGS) 
	$(CC) -c calculate_dM.c $(CFLAGS)

kernel4: kernel4.o utils.o
	$(CC) kernel4.o utils.o -o kernel4 $(CFLAGS) $(LDFLAGS)

kernel4.o: kernel4.c
	$(CC) -c kernel4.c $(CFLAGS)

utils.o: utils.c
	$(CC) -c utils.c $(CFLAGS)

.PHONY: clean
clean:
	rm ./*.o
	rm -f $(TARGET)