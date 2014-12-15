__kernel
void simpleMultiply(__global float* MatC, int w_A, int w_B, __global float* MatA, __global float* MatB) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    float sum = 0.0f;
    for (int i = 0; i < w_A; i++) 
        sum += MatA[row*w_A+i] * MatB[i*w_B+col];
    MatC[row*w_B+col] = sum;
}
