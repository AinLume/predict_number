CXX     	:= g++

BIN     	:= bin
SRC     	:= src
INCLUDE 	:= include
LIB     	:= lib
LIBRARIES   := -lsfml-graphics -lsfml-window -lsfml-system
EXECUTABLE  := main


all: $(BIN)/$(EXECUTABLE)

run: clean all
	clear
	@echo "🚀 Executing..."
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	@echo "🚧 Building..."
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

clean:
	@echo "🧹 Clearing..."
	-rm $(BIN)/*