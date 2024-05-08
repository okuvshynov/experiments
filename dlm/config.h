#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <sstream>

struct config
{
    std::string bind_address;
    std::string attach_to;

    std::string model_path;
    uint32_t n_batch;
    uint32_t n_ctx;
    uint32_t n_threads;
    uint32_t n_gpu_layers;
};

// simple and not very efficient option parser
struct parser
{
    parser()
    {
        add_option("--addr",       &config::bind_address);
        add_option("-a",           &config::attach_to);
        add_option("-m",           &config::model_path);
        add_option("--model",      &config::model_path);
        add_option("-ngl",         &config::n_gpu_layers);
        add_option("--gpu_layers", &config::n_gpu_layers);
        add_option("-t",           &config::n_threads);
        add_option("--threads",    &config::n_threads);
        add_option("--batch_size", &config::n_batch);
    }

    int parse_options(int argc, char ** argv, config & conf)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string key(argv[i]);
            auto it = setters_.find(key);
            if (it != setters_.end())
            {
                if (++i < argc)
                {
                    it->second(argv[i], conf);
                }
                else
                {
                    fprintf(stderr, "No argument value provided for %s\n", argv[i]);
                    return 1;
                }
            }
            else
            {
                fprintf(stderr, "Unknown argument %s\n", argv[i]);
                return 1;
            }
        }
        return 0;
    }
  private:
    std::map<std::string, std::function<void(const char*, config&)>> setters_;

    template<typename T>
    void add_option(const std::string& key, T config::* field)
    {
        setters_[key] = [field](const char * value, config & conf)
        {
            parse_value(value, conf.*field);
        };
    }

    template<typename T>
    static void parse_value(const char * value, T & field)
    {
        std::istringstream iss(value);
        iss >> field;
    }
};

template<>
void parser::parse_value<std::string>(const char * value, std::string & field)
{
    field = value;
}