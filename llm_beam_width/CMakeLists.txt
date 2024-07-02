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

add_executable(beam_width beam_width.cpp)

target_link_libraries(beam_width  PRIVATE common) # from llama.cpp
target_compile_options(beam_width PRIVATE -Wall -Wextra -Wpedantic)
