add_executable(testTensor testTensor.cpp)
target_link_libraries(testTensor PRIVATE nn)

add_test(NAME testTensor COMMAND testTensor)