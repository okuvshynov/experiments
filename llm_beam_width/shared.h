#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

// llama.cpp
#include <common.h>
#include <llama.h>

std::string replace(const std::string& templ, const std::string& question) {
    std::string res = templ;
    size_t i = 0;
    const std::string placeholder = "{question}";
    while ((i = res.find(placeholder, i)) != std::string::npos)
    {
        res.replace(i, placeholder.length(), question);
        i += question.length();
    }
    return res;
}

std::string q_filename(size_t index)
{
    return "data/q" + std::to_string(index) + ".txt"; 
}

std::string a_filename(size_t index)
{
    return "data/a" + std::to_string(index) + ".txt"; 
}

std::string p_filename(size_t /* index */)
{
    return "data/prompt.txt"; 
}

int r_file(const std::string& name, std::string * out_content)
{
    std::ifstream file(name);
    if (!file)
    {
        fprintf(stderr, "Unable to open file: %s\n", name.c_str());
        return 1;
    }
    * out_content = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return 0;
}

int r_tokens(const std::string& name, std::vector<llama_token> * out_tokens)
{
    std::ifstream file(name);
    if (!file)
    {
        fprintf(stderr, "Unable to open file: %s\n", name.c_str());
        return 1;
    }
    llama_token id;

    while (file >> id)
    {
        out_tokens->push_back(id);
    }
    return 0;
}

int w_file(const std::string& name, const std::vector<llama_token>& tokens)
{
    std::ofstream file(name);
    if (!file)
    {
        fprintf(stderr, "Unable to open file: %s\n", name.c_str());
        return 1;
    }
    for (llama_token id : tokens)
    {
        file << id << std::endl;
        if (!file)
        {
            fprintf(stderr, "Error writing to file: %s\n", name.c_str());
            return 1;
        }
    }
    return 0;
}

llama_token greedy(float * logits, llama_token n_vocab)
{
    llama_token res = 0;
    for (llama_token tok = 1; tok < n_vocab; tok++)
    {
        if (logits[tok] > logits[res])
        {
            res = tok;
        }
    }
    return res;
}

