add_library(nn STATIC nn/tensor.cpp)

target_include_directories(nn PUBLIC ${CMAKE_SOURCE_DIR}/include)
