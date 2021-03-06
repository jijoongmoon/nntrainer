cmake_minimum_required(VERSION 2.8.3)
project(TransferLearning)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -std=c++11 -pthread")
find_package(PkgConfig REQUIRED)

pkg_check_modules(TFLITE tensorflow-lite)
link_libraries(${TFLITE_LIBRARIES})

pkg_check_modules(NNTRAINER nntrainer)
link_libraries(${NNTRAINER_LIBRARIES})

include_directories( ${include_directories}
		     ${CMAKE_CURRENT_SOURCE_DIR}
		     ${TFLITE_INCLUDE_DIRS}
		     ${NNTRAINER_INCLUDE_DIRS}		     
		     )
set(SRCS
	main.cpp
	bitmap_helpers.cpp
	)

add_executable(TransferLearning ${SRCS})
target_link_libraries( TransferLearning ${TFLITE_LIBRARIES} ${CMAKE_DL_LIBS} ${NNTRAINER_LIBRARIES}) 

install (TARGETS TransferLearning
	RUNTIME DESTINATION bin
	)
