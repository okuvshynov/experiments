cmake_minimum_required(VERSION 3.16)

project(dlm)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b2865
)
FetchContent_MakeAvailable(llama.cpp)

add_executable(dlm dlm.cpp)
add_executable(speculator speculator.cpp)

target_link_libraries(dlm PRIVATE common) # from llama.cpp
target_link_libraries(dlm PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(dlm PRIVATE httplib)

target_link_libraries(speculator PRIVATE common) # from llama.cpp
target_link_libraries(speculator PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(speculator PRIVATE httplib)

configure_file(${llama.cpp_SOURCE_DIR}/ggml-metal.metal ggml-metal.metal COPYONLY)
configure_file(${llama.cpp_SOURCE_DIR}/ggml-common.h ggml-common.h COPYONLY)
