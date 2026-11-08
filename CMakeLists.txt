cmake_minimum_required(VERSION 3.21)
project(ocean CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(ocean src/main.cpp)
target_link_libraries(ocean PRIVATE OpenMP::OpenMP_CXX glfw GLEW::GLEW glm::glm)

add_executable(test_fft src/test_fft.cpp)
add_executable(test_fft2d src/test_fft2d.cpp)
add_executable(test_ocean src/test_ocean.cpp)
add_executable(test_fields src/test_fields.cpp)

add_executable(ocean_viewer src/viewer.cpp)
target_link_libraries(ocean_viewer PRIVATE glfw GLEW::GLEW glm::glm OpenMP::OpenMP_CXX)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)