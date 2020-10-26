set(the_description "Plot function for Mat data. Copied from opencv contrib-3.2.0")
#ocv_define_module(plot opencv_core opencv_highgui WRAP python)
project(cv_plot)

find_package(OpenCV REQUIRED COMPONENTS core highgui)
add_library(${PROJECT_NAME}
	plot.cpp
	precomp.cpp
)


target_link_libraries(${PROJECT_NAME}
  opencv_core
  opencv_highgui
)
