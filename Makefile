CC = g++
CXXFLAGS = -Wall -std=c++17
SRC_DIR = src
LIB_DIR = lib
INCLUDE_DIRS = $(foreach dir,$(wildcard $(LIB_DIR)/*),-I$(dir))

SRCS = $(wildcard $(SRC_DIR)/**/*.cpp $(SRC_DIR)/*.cpp)
OBJS = $(patsubst %.cpp,%.o,$(SRCS))
LIBS = $(wildcard $(LIB_DIR)/**/*.cpp $(LIB_DIR)/*.cpp)
LIBOBJS = $(patsubst %.cpp,%.o,$(LIBS))
LIB_TARGET = my_library
TARGET = main.exe

all: $(TARGET) clean

$(TARGET): $(OBJS) $(LIBOBJS)
	$(CC) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

clean:
	del /Q .\$(SRC_DIR)\*.o
	del /Q /S .\$(LIB_DIR)\*.o

.PHONY: clean