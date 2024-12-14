# Variables
CXX = mpicxx                # Compilador
CXXFLAGS = -fopenmp -O2     # Opciones de compilación
EXEC = lab2         # Nombre del ejecutable
SRC = lab2.cpp      # Archivo fuente

# Parámetros de prueba
TEST_POINTS = 100000 500000 1000000000
TEST_THREADS = 2 4 8
TEST_PROCESSES = 2 4

# Objetivo principal: compilar
all: $(EXEC)

# Compilación
$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(SRC)

# Limpieza de archivos generados
clean:
	rm -f $(EXEC) output_*.txt

# Pruebas automatizadas con diferentes configuraciones
test: $(EXEC)
	@echo "Ejecutando pruebas y capturando tiempos..."
	@for points in $(TEST_POINTS); do \
	    for threads in $(TEST_THREADS); do \
	        for procs in $(TEST_PROCESSES); do \
	            export OMP_NUM_THREADS=$$threads; \
	            echo "Prueba: $$points puntos, $$procs procesos MPI, $$threads hilos OpenMP"; \
	            mpirun -np $$procs ./$(EXEC) $$points $$procs $$threads > output_$$points\_$$procs\_$$threads.txt; \
	            echo "Resultados guardados en output_$$points\_$$procs\_$$threads.txt"; \
	        done; \
	    done; \
	done