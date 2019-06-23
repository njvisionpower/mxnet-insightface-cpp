cmake_minimum_required(VERSION 3.2)
ADD_DEFINITIONS(-std=c++11)
OPTION(LINUX "Option description" OFF)
IF(LINUX)
    ADD_DEFINITIONS(-DLINUX)
ENDIF(LINUX)

PROJECT(mxnet-insightface-cpp)

FIND_PACKAGE(OpenCV REQUIRED)

MESSAGE(STATUS "    OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( 
                     include/include
                     include/include/cpp-package
)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

LINK_DIRECTORIES(./lib)

ADD_EXECUTABLE(mxnet-insightface-cpp ${DIR_SRCS})

TARGET_LINK_LIBRARIES(mxnet-insightface-cpp ${OpenCV_LIBS} mxnet)
