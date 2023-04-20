#CXX=mpiicpc
CXX=mpicxx
CXXFLAGS=-O2 -std=c++14
CFLAGS=-O2
#LDFLAGS=-L/opt/homebrew/opt/openblas/lib -lopenblas -I/opt/homebrew/opt/openblas/include -lz
#LDFLAGS=-qmkl=sequential -lz
LDFLAGS=-lopenblas -lz
TARGET=nnp.x

.PHONY: all
all: $(TARGET)
	@echo built $(TARGET)

nnp.x: main.o nnp.o cnpy.o nnp_pseudo_athread.o
	$(CXX) nnp_pseudo_athread.o main.o nnp.o cnpy.o -o $(TARGET) $(LDFLAGS)

nnp.o: nnp.cpp
	$(CXX) -c nnp.cpp -o nnp.o $(CXXFLAGS) $(LDFLAGS)

cnpy.o: cnpy.cpp
	$(CXX) -c cnpy.cpp -o cnpy.o $(CXXFLAGS) $(LDFLAGS)

main.o: main.cpp
	$(CXX) -c main.cpp -o main.o $(CXXFLAGS) $(LDFLAGS)

nnp_pseudo_athread.o: nnp_pseudo_athread.c
	$(CXX) -c nnp_pseudo_athread.c -o nnp_pseudo_athread.o $(CFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm *.o
	rm -f $(TARGET)
