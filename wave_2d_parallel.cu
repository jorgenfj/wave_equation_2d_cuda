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

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Can't index past the ghost points, because device only allows
// indexing with unsigned ints
#define D_U_PRV(i,j) d_U_prv[((i+1))*(d_N+2)+(j+1)]
#define D_U_CUR(i,j)     d_U_cur[((i+1))*(d_N+2)+(j+1)]
#define D_U_NXT(i,j) d_U_nxt[((i+1))*(d_N+2)+(j+1)]

#define SHARED_U(i,j) shared_U[(i+1)*(BLOCK_SIZE_Y+2)+(j+1)]

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

// Declare the grid dimensions as constants on the device
// Since we don't specify grid dimensions as arguments when running the program,
// we could just set them as constexpr variables on the host side, but I used __constant__ memory instead
// to use more CUDA features.
__constant__ int_t d_N;
__constant__ int_t d_M;

__constant__ real_t d_dt, d_c, d_dx, d_dy;

// dim3 blockDim(16, 16);
// dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

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

void move_buffer_window( void ) {
    real_t* temp = d_U_prv;
    d_U_prv = d_U_cur;
    d_U_cur = d_U_nxt;
    d_U_nxt = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<M; i++ )
    {
        fwrite ( &h_U(i,0), sizeof(real_t), N, out );
    }
    fclose ( out );
}


// TASK: T4
// Get rid of all the memory allocations
void domain_finalize ( void )
{
// BEGIN: T4
    // Free the host memory
    free(h_U_cur);

    // Free the device memory
    cudaFree(d_U_prv);
    cudaFree(d_U_cur);
    cudaFree(d_U_nxt);
// END: T4
}

// TASK: T6
// TASK: T6
__device__ void apply_boundary_conditions_global(real_t *d_U_cur, real_t* shared_U) {

    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int global_j = blockIdx.y * blockDim.y + threadIdx.y;

    int local_i = threadIdx.x;
    int local_j = threadIdx.y;

    // Neumann (reflective) boundary condition
     if (global_i < d_M && global_j < d_N) {
    if (global_j == 0) {
        SHARED_U(local_i, local_j-1) = D_U_CUR(global_i, global_j + 1);
        // D_U_CUR(global_i, global_j -1) = D_U_CUR(global_i, global_j + 1);
    }
    if (global_j == d_N - 1) {
        SHARED_U(local_i, local_j+1) = D_U_CUR(global_i, global_j - 1);
        // D_U_CUR(global_i, d_N) = D_U_CUR(global_i, d_N - 2);
    }
    if (global_i == 0) {
        SHARED_U(local_i-1, local_j) = D_U_CUR(global_i+1, global_j);
        // D_U_CUR(global_i-1, global_j) = D_U_CUR(global_i+1, global_j);
    }
    if (global_i == d_M - 1) {
        SHARED_U(local_i+1, local_j) = D_U_CUR(global_i-1, global_j);
        // D_U_CUR(d_M, global_j) = D_U_CUR(d_M - 2, global_j);
    }
    }
}


// END: T6



// TASK: T5
__global__ void time_step_kernel(real_t* d_U_prv, real_t* d_U_cur, real_t* d_U_nxt) {

    __shared__ real_t shared_U[(BLOCK_SIZE_X+2)*(BLOCK_SIZE_Y+2)];
    
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int global_j = blockIdx.y * blockDim.y + threadIdx.y;

    int local_i = threadIdx.x;
    int local_j = threadIdx.y;

    // Load the current time step into shared memory
    if (global_i < d_M && global_j < d_N) {
        SHARED_U(local_i, local_j) = D_U_CUR(global_i, global_j);
    
        // Load the ghost points for boirder blocks that are not global boundaries
        if (local_i == 0 && global_i > 0) {
            SHARED_U(local_i-1, local_j) = D_U_CUR(global_i-1, global_j);
        }
        if (local_i == blockDim.x - 1 && global_i < d_M - 1) {
            SHARED_U(local_i+1, local_j) = D_U_CUR(global_i+1, global_j);
        }
        if (local_j == 0 && global_j > 0) {
            SHARED_U(local_i, local_j-1) = D_U_CUR(global_i, global_j-1);
        }
        if (local_j == blockDim.y - 1 && global_j < d_N - 1) {
            SHARED_U(local_i, local_j+1) = D_U_CUR(global_i, global_j+1);
        }

        // Apply boundary conditions for the global boundaries
        apply_boundary_conditions_global(d_U_cur, shared_U);
    
    }

    cg::this_thread_block().sync();


    // Perform the time step calculation
    if (global_i < d_M && global_j < d_N) {
        D_U_NXT(global_i, global_j) = -D_U_PRV(global_i, global_j) + 2.0 * SHARED_U(local_i, local_j)
            + (d_dt * d_dt * d_c * d_c) / (d_dx * d_dy) * (
                SHARED_U(local_i-1,local_j) + SHARED_U(local_i+1,local_j)
                + SHARED_U(local_i,local_j-1) + SHARED_U(local_i,local_j+1)
                - 4.0 * SHARED_U(local_i,local_j)

            );
    }
    __syncthreads();
}
// END: T5

// TASK: T7
void simulate(void) {
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++) {
        if ((iteration % snapshot_freq) == 0) {
            cudaMemcpy(h_U_cur, d_U_cur, (M + 2) * (N + 2) * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            domain_save(iteration / snapshot_freq);
        }

        // Launch the kernel with constants as arguments
        void* kernel_args[] = { (void*)&d_U_prv, (void*)&d_U_cur, (void*)&d_U_nxt };
        
       cudaErrorCheck(cudaLaunchCooperativeKernel(
            (void*)time_step_kernel,
            gridDim,
            blockDim,
            kernel_args
        ));

        cudaDeviceSynchronize();
        move_buffer_window();
    }
}
// END: T7



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
    h_U_cur = (real_t *) malloc ( (M+2)*(N+2)*sizeof(real_t) );

    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            h_U(i,j) = exp ( -4.0*delta*delta );
        }
    }
    

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));

    // Allocate device memory for the three time steps
    cudaErrorCheck(cudaMalloc((void **)&d_U_prv, (M+2)*(N+2)*sizeof(real_t)));
    cudaErrorCheck(cudaMalloc((void **)&d_U_cur, (M+2)*(N+2)*sizeof(real_t)));
    cudaErrorCheck(cudaMalloc((void **)&d_U_nxt, (M+2)*(N+2)*sizeof(real_t)));

    // Copy the initial conditions from host to device for d_U_prv and d_U_cur
    cudaErrorCheck(cudaMemcpy(d_U_prv, h_U_cur, (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_U_cur, h_U_cur, (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice));

    // Copy the grid size constants to the device constant memory using cudaMemcpyToSymbol
    cudaErrorCheck(cudaMemcpyToSymbol(d_N, &N, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_M, &M, sizeof(int_t)));

    // Copy the wave equation parameters to the device constant memory using cudaMemcpyToSymbol
    cudaErrorCheck(cudaMemcpyToSymbol(d_dt, &dt, sizeof(real_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_c, &c, sizeof(real_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dx, &dx, sizeof(real_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dy, &dy, sizeof(real_t)));

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
