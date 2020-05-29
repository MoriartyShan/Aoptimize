cmake_minimum_required(VERSION 3.3)
project(CameraAndHolderCalibrate)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB HEADER_FILES include/*.h)
file(GLOB SOURCE_FILES src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cpp)
target_link_libraries(${PROJECT_NAME}
      ceres
	    ${OpenCV_LIBS}
)
project(CameraAndHolderCalibrateOneImage)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} plain.cpp)
target_link_libraries(${PROJECT_NAME}
     ceres
		 common
)
