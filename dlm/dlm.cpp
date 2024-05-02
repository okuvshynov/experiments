// ZeroMQ
#include <zmq.hpp>

// json parsing
#include <nlohmann/json.hpp>

// llama.cpp
#include <llama.h>
#include <common.h>

using json = nlohmann::json;

int main()
{
    gpt_params params;

    llama_backend_init();
    llama_numa_init(params.numa);

    zmq::context_t context{1};
    zmq::socket_t socket{context, zmq::socket_type::rep};
    socket.bind("tcp://*:5555");

    json ex1 = json::parse(R"(
      {
        "pi": 3.141,
        "happy": true
      }
    )");
    return 0;
}
