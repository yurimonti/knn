cmake_minimum_required (VERSION 3.20)

project ("knnmpi")
find_package(MPI REQUIRED)

add_executable (knnmpi "main.c")

target_link_libraries(knnmpi
 PRIVATE
 ${MPI_C_LIBRARIES})

target_include_directories(knnmpi
 PRIVATE
 ${MPI_C_INCLUDE_PATH}) 

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET knnmpi PROPERTY CXX_STANDARD 20)
endif()