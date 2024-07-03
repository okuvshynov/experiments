cmake_minimum_required(VERSION 3.16)

project(beam_width)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b3269
)
FetchContent_MakeAvailable(llama.cpp)

add_executable(gen_main gen_main.cpp)
add_executable(ranks ranks.cpp)

target_link_libraries(gen_main  PRIVATE common) # from llama.cpp
target_compile_options(gen_main PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(ranks  PRIVATE common) # from llama.cpp
target_compile_options(ranks PRIVATE -Wall -Wextra -Wpedantic)
