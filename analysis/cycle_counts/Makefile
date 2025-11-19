ARCH := sm_90
NVCC := nvcc

INC := -I$(CUDA_HOME)/include
LIB := -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64
NVCCFLAGS := -arch=$(ARCH) -std=c++17

default: all

all: main

main: main.cu Makefile
	$(NVCC) main.cu -o main $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f main

run: main
	./main
