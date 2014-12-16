__kernel 
void __attribute__((vec_type_hint(float4)))
saxpy_naive(__global float *X, 
                 __global float *Y, 
                 __global float *Z, 
                 float alpha, 
                 int dim)
{
    size_t gid = get_global_id(0) * dim;
    size_t upper = gid + dim;
    for (; gid < upper; gid++)
        Z[gid] = X[gid] * alpha + Y[gid]; 
}

__kernel 
void
saxpy_dim4(__global float *X, 
                __global float *Y, 
                __global float *Z, 
                float alpha) 
{
    size_t gid = get_global_id(0) * 4;
    float4 x = (float4)(X[gid], X[gid+1], X[gid+2], X[gid+3]);
    float4 y = (float4)(Y[gid], Y[gid+1], Y[gid+2], Y[gid+3]);
    float4 z = x * alpha + y;
    
    //printf("%.3f %.3f %.3f %.3f\n", x.x, x.y, x.z, x.w);
    //printf("%.3f %.3f %.3f %.3f\n", y.x, y.y, y.z, y.w);
    //printf("%.3f %.3f %.3f %.3f\n", z.x, z.y, z.z, z.w);

    Z[gid] = z.x;
    Z[gid+1] = z.y;
    Z[gid+2] = z.z;
    Z[gid+3] = z.w;
}

__kernel 
void 
saxpy_dim8(__global float *X, 
                __global float *Y, 
                __global float *Z, 
                float alpha) 
{
    size_t gid = get_global_id(0) * 8;
    float8 x = (float8)(X[gid], X[gid+1], X[gid+2], X[gid+3], X[gid+4], X[gid+5], X[gid+6], X[gid+7]);
    float8 y = (float8)(Y[gid], Y[gid+1], Y[gid+2], Y[gid+3], Y[gid+4], Y[gid+5], Y[gid+6], Y[gid+7]);
    float8 z = x * alpha + y;
    
    //printf("%.3f %.3f %.3f %.3f\n", x.x, x.y, x.z, x.w);
    //printf("%.3f %.3f %.3f %.3f\n", y.x, y.y, y.z, y.w);
    //printf("%.3f %.3f %.3f %.3f\n", z.x, z.y, z.z, z.w);

    Z[gid]      = z.s0;
    Z[gid+1]    = z.s1;
    Z[gid+2]    = z.s2;
    Z[gid+3]    = z.s3;
    Z[gid+4]    = z.s4;
    Z[gid+5]    = z.s5;
    Z[gid+6]    = z.s6;
    Z[gid+7]    = z.s7;
}

__kernel 
void 
saxpy_dim16(__global float *X, 
                __global float *Y, 
                __global float *Z, 
                float alpha) 
{
    size_t gid = get_global_id(0) * 16;
    float16 x = (float16)(X[gid], X[gid+1], X[gid+2], X[gid+3], X[gid+4], X[gid+5], X[gid+6], X[gid+7], \
                                X[gid+8], X[gid+9], X[gid+10], X[gid+11], X[gid+12], X[gid+13], X[gid+14], X[gid+15]);
    float16 y = (float16)(Y[gid], Y[gid+1], Y[gid+2], Y[gid+3], Y[gid+4], Y[gid+5], Y[gid+6], Y[gid+7], \
                                Y[gid+8], Y[gid+9], Y[gid+10], Y[gid+11], Y[gid+12], Y[gid+13], Y[gid+14], Y[gid+15]);
    float16 z = x * alpha + y;

    Z[gid]      = z.s0;
    Z[gid+1]    = z.s1;
    Z[gid+2]    = z.s2;
    Z[gid+3]    = z.s3;
    Z[gid+4]    = z.s4;
    Z[gid+5]    = z.s5;
    Z[gid+6]    = z.s6;
    Z[gid+7]    = z.s7;
    Z[gid+8]    = z.s8;
    Z[gid+9]    = z.s9;
    Z[gid+10]   = z.sa;
    Z[gid+11]   = z.sb;
    Z[gid+12]   = z.sc;
    Z[gid+13]   = z.sd;
    Z[gid+14]   = z.se;
    Z[gid+15]   = z.sf;
}
