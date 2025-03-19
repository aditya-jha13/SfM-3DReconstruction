cmake_minimum_required(VERSION 3.5)
project(sfm)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

enable_testing()

# Find GTSAM components
# find_package(GTSAM REQUIRED) # Uses installed package

# set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs

# find_package(GTSAMCMakeTools)
# include(GtsamMakeConfigFile)
# include(GtsamBuildTypes)
# include(GtsamTesting)

# for unittest scripts
# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

# Boost - same requirement as gtsam

find_package(Boost REQUIRED COMPONENTS thread system chrono timer date_time filesystem serialization)
include_directories(${Boost_INCLUDE_DIR})


# Process source subdirs

# set(SOURCE_FILES sfm.cpp)
# add_executable(featureMatch src/featurematching.cpp)
add_executable(sfm src/mysfm.cpp)
# add_executable(main src/main.cpp)
# link_directories(
#     ${OpenCV_LIBRARY_DIRS}
#     #${GTSAM_LIBRARY_DIRS}
# )

add_definitions(${OpenCV_DEFINITIONS} ${GTSAM_DEFINITIONS})

# target_link_libraries (featureMatch ${OpenCV_LIBRARIES})
target_link_libraries (sfm  ${OpenCV_LIBRARIES} )#gtsam )
target_link_libraries (sfm Eigen3::Eigen)
target_link_libraries (sfm Boost::chrono Boost::timer Boost::date_time Boost::filesystem Boost::serialization)
target_link_libraries (sfm ${OpenCV_LIBRARIES})