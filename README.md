# PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time.

# Aim:
To implement matrix multiplication using GPU.

# Procedure:
## Step 1:
Define constants and variables, including matrix sizes and device memory pointers.

## Step 2:
Initialize matrices and allocate GPU memory.

## Step 3:
Copy input matrices from host to device.

## Step 4:
Set grid and block dimensions, launch the kernel function, and copy the result matrix from device to host.

## Step 5:
Measure elapsed time, print the result matrix and elapsed time, and free device memory.

## Step 6:
Terminate the program.

# Program :-

Developed By Gowri M<br>
reg. No.: 212220230019

``` c
#include <stdio.h>
#include <cuda.h>

__global__ void cudaAdd(int* a, int* b, int* c, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] * b[i];
    }
}
int main() {
    srand(time(0));
    int a[100], b[100], c[100];

    for (int i = 0; i < 100; i++) {
        a[i] = 10;
        b[i] = 10;
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, sizeof(int) * 100);
    cudaMalloc(&d_b, sizeof(int) * 100);
    cudaMalloc(&d_c, sizeof(int) * 100);

    cudaMemcpy(d_a, a, sizeof(int) * 100, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * 100, cudaMemcpyHostToDevice);

    cudaMemset(d_c, 0, sizeof(int) * 100);

    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((100 + block.x - 1) / block.x);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    cudaAdd << <grid, block >> > (d_a, d_b, d_c, 100);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    cudaMemcpy(c, d_c, sizeof(int) * 100, cudaMemcpyDeviceToHost);

    printf("The kernel ran for %.2f milliseconds.\n", elapsed);
    for (int i = 0; i < 100; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
```

## Output:
![242962410-1b40b7a7-e980-43f3-8068-ff07cbf951e2](https://github.com/Gowri4622/-PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time./assets/75235455/dc8ffe46-3a45-47a7-bcdb-49be3e6d019d)

## Result:
Thus, the program to implement matrix multiplication using the GPU has been successfully executed.
