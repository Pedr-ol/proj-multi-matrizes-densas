CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
OMPFLAGS ?= -fopenmp

BIN  := bin
SRC  := src
SEQ  := $(SRC)/sequencial
OMP  := $(SRC)/openmp
MPI  := $(SRC)/mpi
TEST := $(SRC)/tests

.PHONY: all baseline openmp openmp_blocked mpi testes validar medir graficos limpar clean docker-build docker-shell

all: baseline openmp openmp_blocked mpi testes

$(BIN):
	mkdir -p $(BIN)

baseline: $(BIN)/baseline
openmp: $(BIN)/openmp
openmp_blocked: $(BIN)/openmp_blocked
mpi: $(BIN)/mpi
testes: $(BIN)/test_openmp $(BIN)/test_mpi

$(BIN)/baseline: $(SEQ)/01_matmul.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/openmp: $(OMP)/matmul_openmp.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

$(BIN)/openmp_blocked: $(OMP)/matmul_openmp_blocked.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

$(BIN)/mpi: $(MPI)/matmul_mpi.cpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $< -o $@

$(BIN)/test_openmp: $(TEST)/test_openmp.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

$(BIN)/test_mpi: $(TEST)/test_mpi.cpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $< -o $@

validar: all
	python3 scripts/validar.py

medir: all
	python3 scripts/medir.py

graficos: medir
	python3 scripts/gerar_graficos.py

limpar clean:
	rm -f $(BIN)/baseline $(BIN)/openmp $(BIN)/openmp_blocked $(BIN)/mpi \
	      $(BIN)/test_openmp $(BIN)/test_mpi $(BIN)/*.o

docker-build:
	docker compose build

docker-shell:
	docker compose run --rm ambiente
