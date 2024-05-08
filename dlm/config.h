#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <sstream>


struct value_parser
{
    template<typename value_t>
    static void parse(const char * value, value_t & field)
    {
        std::istringstream iss(value);
        iss >> field;
    }
};

template<>
void value_parser::parse<std::string>(const char * value, std::string & field)
{
    field = value;
}

// simple and not very efficient option parser
template<typename config_t>
struct parser
{
    int parse_options(int argc, char ** argv, config_t & conf)
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
                    fprintf(stderr, "No argument value provided for %s\n", argv[i - 1]);
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

    template<typename T>
    void add_option(const std::string& key, T config_t::* field)
    {
        setters_[key] = [field](const char * value, config_t & conf)
        {
            value_parser::parse(value, conf.*field);
        };
    }

    template<typename T>
    void add_option(const std::initializer_list<std::string>& keys, T config_t::* field)
    {
        for (const auto& key : keys)
        {
            add_option(key, field);
        }
    }

  private:
    std::map<std::string, std::function<void(const char*, config_t&)>> setters_;
};

