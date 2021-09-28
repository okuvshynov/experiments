bm:
	mkdir -p _build/
	clang++ -Wall -O3 bm_bswaps.cpp -o _build/bm_bswaps -std=c++17
	./_build/bm_bswaps -i

test:
	mkdir -p _build/
	clang++ -Wall -O3 test_bswaps.cpp -o _build/test_bswaps -std=c++17 -lgtest
	_build/test_bswaps
