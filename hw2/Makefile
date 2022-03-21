OUTPUT=./
make_all:
	g++ -g -O0 -o ${OUTPUT}/val_test01_solved val_test01_solved.cpp
	g++ -g -O0 -o ${OUTPUT}/val_test02_solved val_test02_solved.cpp
	g++ -fopenmp -O3 -o ${OUTPUT}/MMult1  -march=native MMult1.cpp
	g++ -fopenmp -O3 -o ${OUTPUT}/omp_solved2  omp_solved2.c
	g++ -fopenmp -O3 -o ${OUTPUT}/omp_solved3  omp_solved3.c
	g++ -fopenmp -O3 -o ${OUTPUT}/omp_solved4  omp_solved4.c
	g++ -fopenmp -O3 -o ${OUTPUT}/omp_solved5  omp_solved5.c
	g++ -fopenmp -O3 -o ${OUTPUT}/omp_solved6  omp_solved6.c
	g++ -fopenmp -O3 -o ${OUTPUT}/gs2D-omp  gs2D-omp.cpp
	g++ -fopenmp -O3 -o ${OUTPUT}/jacobi2D-omp  jacobi2D-omp.cpp
	g++ -O3 -o ${OUTPUT}/gs2D-seq  gs2D-omp.cpp
	g++ -O3 -o ${OUTPUT}/jacobi2D-seq jacobi2D-omp.cpp

clean:
	rm -rf ${OUTPUT}/val_test01_solved ${OUTPUT}/val_test02_solved \
	 ${OUTPUT}/MMult1 ${OUTPUT}/omp_solved2 ${OUTPUT}/omp_solved3  \
	 ${OUTPUT}/omp_solved4 ${OUTPUT}/omp_solved5 ${OUTPUT}/omp_solved6 \
	 ${OUTPUT}/gs2D-omp ${OUTPUT}/jacobi2D-omp \
	 ${OUTPUT}/gs2D-seq ${OUTPUT}/jacobi2D-seq
