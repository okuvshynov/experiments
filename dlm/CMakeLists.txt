cmake_minimum_required(VERSION 3.16)

project(dlm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

option(ZMQ_BUILD_TESTS    "Build the zmq tests."    OFF)
option(CPPZMQ_BUILD_TESTS "Build the cppzmq tests." OFF)

include(FetchContent)

FetchContent_Declare(
    zmq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG        v4.3.2
)
FetchContent_MakeAvailable(zmq)

FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG        v4.6.0
)
FetchContent_MakeAvailable(cppzmq)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b2781
)
FetchContent_MakeAvailable(llama.cpp)

add_executable(dlm dlm.cpp)
add_executable(speculator speculator.cpp)

target_link_libraries(dlm PRIVATE cppzmq)
target_link_libraries(dlm PRIVATE common) # from llama.cpp
target_link_libraries(dlm PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(speculator PRIVATE cppzmq)
target_link_libraries(speculator PRIVATE common) # from llama.cpp
target_link_libraries(speculator PRIVATE nlohmann_json::nlohmann_json)

configure_file(${llama.cpp_SOURCE_DIR}/ggml-metal.metal ggml-metal.metal COPYONLY)
configure_file(${llama.cpp_SOURCE_DIR}/ggml-common.h ggml-common.h COPYONLY)
