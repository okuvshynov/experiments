run:
	mkdir -p _out/
	clang++ m1cores.cpp -O3 -std=c++17 -o _out/m1cores
	sudo ./_out/m1cores -e 10 -t 2 | sed 's/storm_/storm\,/g' > ./_out/m1cores.csv 
	Rscript m1cores.r ./_out/m1cores.csv
