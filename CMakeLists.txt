cmake_minimum_required(VERSION 3.3)
project(CameraAndHolderCalibrate)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

project(CameraAndHolderCalibrateOneImage)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} plain.cpp)
target_link_libraries(${PROJECT_NAME}
     ceres
	 common
	 cv_plot
)
project(TwoCameraExtrinsics)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} two_camera.cpp)
target_link_libraries(${PROJECT_NAME}
		ceres
		common
		)