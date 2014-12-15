CC=gcc
CXX=g++
CFLAGS=-framework OpenCL

all:
	$(CC) clInfo.c -o clInfo.o $(CFLAGS)
	$(CC) clKernel.c -o clKernel.o $(CFLAGS)
	$(CXX) ocl_test.cpp -o ocl_test.o $(CFLAGS)
