test:
	mkdir -p _build
	 $(CXX) -Wall tests/rtdigest_tests.cpp -g -O2 -o _build/unit_tests -I . -std=c++17 -lgtest
	_build/unit_tests

precision:
	mkdir -p _build
	$(CXX) -Wall tests/rtdigest_precision.cpp -O2 -o _build/precision -I . -std=c++17
	_build/precision > /tmp/1
	Rscript tests/precision_plot.r /tmp/1 

compare:
	mkdir -p _build
	$(CXX) -Wall tests/rtdigest_compare.cpp -O2 -o _build/compare -I . -std=c++17
	_build/compare > tests/logs/rtdigest.log

perf:
	mkdir -p _build
	$(CXX) -Wall tests/rtdigest_performance.cpp -O2 -o _build/performance -I . -std=c++17
	_build/performance

format:
	find . -iname *.h -o -iname *.c -o -iname *.cpp | xargs clang-format -i

