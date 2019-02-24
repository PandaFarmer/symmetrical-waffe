INCLUDE += -I /usr/local/cuda-7.5/include
INCLUDE += -I $(shell python -c \
    'import tensorflow as tf; print(tf.sysconfig.get_include())')

CXX = gcc -std=c++11
CXXFLAGS =                          \
    -D_MWAITXINTRIN_H_INCLUDED  \
    -D_FORCE_INLINES            \
    $(INCLUDE) -fPIC -lcudart   \

NVCC = nvcc -std=c++11 -c
NVCCFLAGS =                         \
    -D_MWAITXINTRIN_H_INCLUDED  \
    -D_FORCE_INLINES            \
    $(INCLUDE) -x cu -Xcompiler -fPIC

LDFLAGS = -shared
CUDA_SRCS = zero_out_gpu.cu.cc
SRCS = zero_out.cc
RM = rm -f
TARGET_LIB = zero_out.so
CUDA_OBJ = zero_out.cu.o

all: $(TARGET_LIB)

# This target (CPU and GPU) does not find the right symbols
$(TARGET_LIB): $(SRCS) $(CUDA_OBJ) 
    $(CXX) $(LDFLAGS) -o $@ $^ $(CXXFLAGS) -DGOOGLE_CUDA=1

# This target (CPU only) is fine
# $(TARGET_LIB): $(SRCS) 
#   $(CXX) $(LDFLAGS) -o $@ $^ $(CXXFLAGS) -DGOOGLE_CUDA=0

$(CUDA_OBJ): $(CUDA_SRCS)
    $(NVCC) -o $@ $^ $(NVCCFLAGS) -DGOOGLE_CUDA=1

.PHONY: clean
clean:
    -$(RM) $(TARGET_LIB)
    -$(RM) *~
    -$(RM) *.o