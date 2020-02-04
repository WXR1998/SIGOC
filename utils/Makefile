CXX = g++
CXXFLAG =  -O2 -w 

TARGET = extract

all: 
	rm -f $(TARGET)
	$(CXX) $(TARGET).cpp CJsonObject/CJsonObject.cpp CJsonObject/cJSON.c lodepng/lodepng.cpp -o $(TARGET) $(CXXFLAG)
	./$(TARGET)

clean:
	rm -f $(TARGET)