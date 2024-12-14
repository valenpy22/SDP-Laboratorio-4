#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

/**
 * @brief Inicializa el entorno MPI y obtiene información básica sobre los procesos.
 * 
 * @param rank Referencia a una variable donde se almacenará el identificador del proceso actual.
 * @param size Referencia a una variable donde se almacenará el número total de procesos.
 * 
 * @note Debe llamarse antes de cualquier otra operación de MPI. Es obligatorio finalizar el entorno 
 *       MPI con MPI_Finalize() al terminar el programa.
 */
void initialize_MPI(int &rank, int &size){
    MPI_Init(nullptr, nullptr);

    // Identificador del proceso y número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

/**
 * @brief Genera puntos aleatorios dentro de un cuadrado y cuenta cuántos están dentro del círculo unitario.
 *
 * @param long long points_per_process El número de puntos a generar en este proceso.
 * @param unsigned int seed La semilla para la generación de números aleatorios.
 * @return long long El número de puntos que caen dentro del círculo unitario.
 * 
 * @note La función usa paralelización con OpenMP para optimizar el cálculo en múltiples hilos. 
 *       Cada hilo genera puntos de forma independiente y actualiza el contador global usando 
 *       una sección crítica.
 */
long long calculate_points_in_circle(long long points_per_process, unsigned int seed){
    long long local_count = 0;

    // Paralelización con OpenMP
    #pragma omp parallel
    {
        //Generar una semilla única para cada hilo
        unsigned int thread_seed = seed + omp_get_thread_num();

        // Contador local para cada hilo
        long long thread_count = 0;

        // Generar puntos en paralelo
        #pragma omp for
        for (long long i = 0; i < points_per_process; i++) {

            //Generar coordenadas aleatorias (x, y) en el rango [-1, 1]
            double x = (double)rand_r(&thread_seed) / RAND_MAX * 2.0 - 1.0; // Generar x en [-1, 1]
            double y = (double)rand_r(&thread_seed) / RAND_MAX * 2.0 - 1.0; // Generar y en [-1, 1]

            // Verificar si el punto está dentro del círculo unitario
            if (x * x + y * y <= 1.0) thread_count++;
        }

        // Sección crítica: Actualizar el contador global de puntos dentro del círculo
        #pragma omp critical
        local_count += thread_count;
    }

    // Devolver el conteo total de puntos dentro del círculo para este proceso
    return local_count;
}

/**
 * @brief Combina los resultados de todos los procesos MPI y calcula la estimación de π.
 *
 * @param long long local_count El número de puntos dentro del círculo calculados por este proceso.
 * @param long long total_points El número total de puntos generados en todos los procesos.
 * @param int rank El identificador del proceso actual.
 * @return double La estimación de π calculada como 4 * (puntos_dentro / puntos_totales).
 * 
 * @note Solo el proceso principal (rank 0) realiza el cálculo final y devuelve un valor válido. 
 *       Los demás procesos devuelven 0.0.
 */
double reduce_and_calculate_pi(long long local_count, long long total_points, int rank) {
    long long global_count = 0;

    // Reducir los resultados usando MPI_Reduce
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Calcular pi en el proceso principal
    if (rank == 0) {
        return 4.0 * (double)global_count / (double)total_points;
    }

    // Los demás procesos no necesitan devolver un valor válido
    return 0.0;
}

/**
 * @brief Finaliza el entorno MPI, liberando los recursos utilizados.
 *
 * @note Esta función debe ser llamada al final del programa después de que todas las operaciones
 *       relacionadas con MPI hayan concluido. Es obligatorio para evitar fugas de recursos 
 *       del entorno MPI.
 */
void finalize_MPI() {
    MPI_Finalize();
}

/**
 * @brief Programa principal que estima el valor de π usando MPI y OpenMP con el método de Monte Carlo.
 *
 * @param int argc Número de argumentos proporcionados desde la terminal.
 * @param char* argv[] Lista de argumentos pasados al programa. 
 *                     - argv[1]: Número total de puntos a generar.
 *                     - argv[2]: Número de procesos MPI (solo como referencia, ya que MPI lo determina automáticamente).
 *                     - argv[3]: Número de hilos OpenMP por proceso.
 * 
 * @return int Código de retorno del programa. Devuelve 0 si la ejecución fue exitosa.
 *
 * @note El programa sigue los siguientes pasos:
 *       1. Inicializa MPI y configura los procesos disponibles.
 *       2. Configura el número de hilos OpenMP según el argumento proporcionado.
 *       3. Divide el trabajo entre procesos y calcula localmente cuántos puntos caen dentro del círculo.
 *       4. Combina los resultados usando MPI_Reduce y calcula la estimación final de π.
 *       5. El proceso principal (rank 0) muestra los resultados.
 *       6. Finaliza el entorno MPI.
 *
 * @example
 *       Compilación:
 *       mpicxx -fopenmp -o programa programa.cpp
 *
 *       Ejecución:
 *       mpirun -np 4 ./programa 1000000 4 8
 *       
 *       Salida esperada:
 *       Número total de puntos: 1000000
 *       Número de procesos MPI: 4
 *       Número de hilos OpenMP por proceso: 8
 *       Estimación de π: 3.14159
 */
int main(int argc, char* argv[]) {

    // Identificador del proceso y número total
    int rank, size;

    // Número total de puntos
    long long total_points;

    // Número de hilos OpenMP por proceso
    int num_threads;

    // Verifica que se hayan ingresado los parámetros necesarios por la terminal
    if (argc < 4) {
        if (rank == 0) {
            std::cerr << "Uso: " << argv[0] << " <número_total_puntos> <número_procesos_MPI> <número_hilos_OpenMP>\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Finaliza si faltan parámetros
    } else {
        // Leer parámetros desde los argumentos
        total_points = atoll(argv[1]); // Número total de puntos
        num_threads = atoi(argv[3]);  // Número de hilos OpenMP
    }

    // Inicializar MPI
    initialize_MPI(rank, size);

    // Configurar el número de hilos de OpenMP
    omp_set_num_threads(num_threads);

    // Dividir puntos entre procesos
    long long points_per_process = total_points / size;

    // Generar una semilla única para este proceso basado en su rank
    unsigned int seed = time(NULL) + rank;

    // Medir tiempo de inicio total
    double total_start_time = MPI_Wtime();

    // Medir tiempo para el cálculo
    double computation_start_time = MPI_Wtime();

    // Calcular puntos dentro del círculo para este proceso
    long long local_count = calculate_points_in_circle(points_per_process, seed); 

    // Tiempo de finalización de la sección de cálculo
    double computation_end_time = MPI_Wtime();

    // Reducir y calcular el valor estimado de pi
    double pi_estimate = reduce_and_calculate_pi(local_count, total_points, rank); 

    // Tiempo de finalización total
    double total_end_time = MPI_Wtime();

    // Mostrar resultado final
    if (rank == 0) {
        // Resultados finales
        std::cout << "Número total de puntos: " << total_points << "\n";
        std::cout << "Número de procesos MPI: " << size << "\n";
        std::cout << "Número de hilos OpenMP por proceso: " << num_threads << "\n";
        std::cout << "Estimación de π: " << pi_estimate << "\n";

        // Tiempos de ejecución
        std::cout << "Tiempo total de ejecución: " << (total_end_time - total_start_time) << " segundos\n";
        std::cout << "Tiempo en el cálculo: " << (computation_end_time - computation_start_time) << " segundos\n";
    }

    // Finalizar MPI liberando recursos
    finalize_MPI();

    return 0;
}


