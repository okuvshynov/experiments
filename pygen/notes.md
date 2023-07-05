https://huggingface.co/mosaicml/mpt-30b
https://github.com/mosaicml/llm-foundry/


Extra steps:
apt-get install python3-pybind11
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install pyopenssl --upgrade

Testing on h100:

batch_size = 1, total time = 14.3
batch_size = 2, total time = 20.7
batch_size = 4, total time = 35.5
batch_size = 8, total time = 66
batch_size = 16, total time = 126
batch_size = 32, total time = 245

Let's count in $$$.

7.6 seconds per output program.

Let's say we want to get 1 million programs.

7.6 millions == 2000 hours == $4k