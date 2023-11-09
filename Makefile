CC = g++
CXXFLAGS = -Wall -std=c++17
SRC_DIR = ./src
LIB_DIR = ./lib
SRCS = $(wildcard $(SRC_DIR)/**/*.cpp $(SRC_DIR)/*.cpp)
OBJS = $(patsubst %.cpp,%.o,$(SRCS))
TARGET = main.exe
LIBS = $(wildcard $(LIB_DIR)/**/*.cpp $(LIB_DIR)/*.cpp)
LIBOBJS = $(patsubst %.cpp,%.o,$(LIBS))
LIB_TARGET = my_library

all: $(TARGET) $(LIB_TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $^

$(LIB_TARGET): $(LIBOBJS)
	$(CC) $(CXXFLAGS) -shared -o $@.so $^

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $<

clean:
	find $(SRC_DIR) -name "*.o" -type f -delete
	find $(LIB_DIR) -name "*.o" -type f -delete
	rm -f $(TARGET) $(LIB_TARGET).so