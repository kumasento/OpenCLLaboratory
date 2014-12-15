SRC=src
INC=include
LIB=lib

BIN=bin
CC=gcc
CXX=g++

SRC=src

AMDAPPDIR=/opt/AMDAPPSDK-2.9-1
INTELINC=/opt/intel/opencl-sdk/include
INTELLIB=/opt/intel/opencl/lib64

ICPC=icpc

intel:
	$(ICPC) \
		$(SRC)/oclcommon.cpp \
		$(SRC)/saxpy_naive.cpp -o \
		$(BIN)/saxpy_naive.o \
		-I$(INC) -I$(INTELINC) \
		-L$(INTELLIB) \
		-lOpenCL

osx-dim0:
	$(CXX) \
		$(SRC)/oclcommon.cpp \
		$(SRC)/saxpy_naive.cpp -o \
		$(BIN)/saxpy_naive.o \
		-I$(INC) \
		-framework OpenCL
osx-dim4:
	$(CXX) \
		$(SRC)/oclcommon.cpp \
		$(SRC)/saxpy_dim4.cpp -o \
		$(BIN)/saxpy_dim4.o \
		-I$(INC) \
		-framework OpenCL
osx-dim8:
	$(CXX) \
		$(SRC)/oclcommon.cpp \
		$(SRC)/saxpy_dim8.cpp -o \
		$(BIN)/saxpy_dim8.o \
		-I$(INC) \
		-framework OpenCL
osx-dim16:
	$(CXX) \
		$(SRC)/oclcommon.cpp \
		$(SRC)/saxpy_dim16.cpp -o \
		$(BIN)/saxpy_dim16.o \
		-I$(INC) \
		-framework OpenCL
