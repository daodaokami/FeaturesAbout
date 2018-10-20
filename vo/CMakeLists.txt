add_executable(main main.cpp)
add_executable(testgb testgaussblur.cpp)

target_link_libraries(main suo15features)
target_link_libraries(testgb suo15features)