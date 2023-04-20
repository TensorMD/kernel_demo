CXX=mpiicpc
CXXFLAGS=-O2
LDFLAGS=-qmkl=sequential -lm
TARGET=main

.PHONY: all
all: $(TARGET)
	@echo built $(TARGET)

main: main.o
	$(CXX) main.o -o main $(CXXFLAGS) $(LDFLAGS)

kernel1.o: main.cpp ration.hpp
	$(CXX) -c main.cpp $(CXXFLAGS) 

.PHONY: clean
clean:
	rm ./*.o
	rm -f $(TARGET)