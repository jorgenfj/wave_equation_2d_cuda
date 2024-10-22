#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
#include <cooperative_groups.h>
// END: T1


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b

namespace cg = cooperative_groups;
cudaDeviceProp device_prop;

// Device buffers for three time steps, indexed with 2 ghost points for the boundary
real_t *d_U_prv, *d_U_cur, *d_U_nxt;

#define d_U_prv(i,j) d_U_prev[0][((i)+1)*(N+2)+(j)+1]
#define d_U(i,j)     d_U_cur[1][((i)+1)*(N+2)+(j)+1]
#define d_U_nxt(i,j) d_U_nxt[2][((i)+1)*(N+2)+(j)+1]

// Declare the grid dimensions as constants on the device
__constant__ int_t d_N;
__constant__ int_t d_M;

// Simulation parameters: size, step count, and how often to save the state
int_t
    N = 128,
    M = 128,
    max_iteration = 1000000,
    snapshot_freq = 1000;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

// We only need the current time step on the host side
real_t *h_U_cur;

#define h_U(i,j) h_U_cur[((i)+1)*(N+2)+(j)+1]
// END: T1b

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<M; i++ )
    {
        fwrite ( &U(i,0), sizeof(real_t), N, out );
    }
    fclose ( out );
}


// TASK: T4
// Get rid of all the memory allocations
void domain_finalize ( void )
{
// BEGIN: T4
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition
// BEGIN: T6
void boundary_condition ( void )
{
    for ( int_t i=0; i<M; i++ )
    {
        U(i,-1) = U(i,1);
        U(i,N)  = U(i,N-2);
    }
    for ( int_t j=0; j<N; j++ )
    {
        U(-1,j) = U(1,j);
        U(M,j)  = U(M-2,j);
    }
}
// END: T6


// TASK: T5
// Integration formula
// BEGIN; T5
void time_step ( void )
{
    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(dx*dy) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                     );
        }
    }
}
// END: T5


// TASK: T7
// Main time integration.
void simulate( void )
{
// BEGIN: T7
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
// END: T7
}


// TASK: T8
// GPU occupancy
void occupancy( void )
{
// BEGIN: T8
    ;
// END: T8
}


// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool init_cuda()
{
// BEGIN: T2
     // Check the number of CUDA-capable devices.
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-compatible device found or failed to get device count: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    std::cout << "Number of CUDA-compatible devices: " << device_count << std::endl;

    // Iterate through devices and select a suitable one.
    for (int device = 0; device < device_count; ++device) {
        error = cudaGetDeviceProperties(&device_prop, device);
        if (error != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << device << ": "
                      << cudaGetErrorString(error) << std::endl;
            return false;
        }

        // Print information about the device (similar to Figure 3).
        std::cout << "Device " << device << ": " << device_prop.name << std::endl;
        std::cout << "  Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "  Total global memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << device_prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << device_prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << device_prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: [" << device_prop.maxThreadsDim[0] << ", "
                  << device_prop.maxThreadsDim[1] << ", " << device_prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max grid size: [" << device_prop.maxGridSize[0] << ", "
                  << device_prop.maxGridSize[1] << ", " << device_prop.maxGridSize[2] << "]" << std::endl;

        // Select the device (you can customize which one to select).
        cudaSetDevice(device);
    }
    return true;
// END: T2
}


// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ( void )
{
// BEGIN: T3
    bool locate_cuda = init_cuda();
    if (!locate_cuda)
    {
        exit( EXIT_FAILURE );
    }

    // We only need the current time step on the host
    buffers[1] = (real_t *) malloc ( (M+2)*(N+2)*sizeof(real_t) );

    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Allocate device memory for the three time steps
    cudaErrorCheck(cudaMalloc((void **)&d_U_prv, (M+2)*(N+2)*sizeof(real_t)));
    cudaErrorCheck(cudaMalloc((void **)&d_U_cur, (M+2)*(N+2)*sizeof(real_t)));
    cudaErrorCheck(cudaMalloc((void **)&d_U_nxt, (M+2)*(N+2)*sizeof(real_t)));

    // Copy the initial conditions from host to device for d_U_prv and d_U_cur
    cudaErrorCheck(cudaMemcpy(d_U_prv, h_U_cur, (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_U_cur, h_U_cur, (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc((void **)&d_N, sizeof(int_t)));
    cudaErrorCheck(cudaMalloc((void **)&d_M, sizeof(int_t)));

    // Copy the grid size constants to the device constant memory using cudaMemcpyToSymbol
    cudaErrorCheck(cudaMemcpyToSymbol(d_N, &N, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_M, &M, sizeof(int_t)));

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
// END: T3
}


int main ( void )
{
    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    occupancy();

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
