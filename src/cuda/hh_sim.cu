#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Simple Hodgkin-Huxley CUDA RK4 implementation for small N (2-3 neurons)



__device__ float alpha_m(float V){ return 0.1f*(V+40.0f)/(1.0f - expf(-(V+40.0f)/10.0f)); }
__device__ float beta_m(float V){ return 4.0f * expf(-(V+65.0f)/18.0f); }
__device__ float alpha_h(float V){ return 0.07f * expf(-(V+65.0f)/20.0f); }
__device__ float beta_h(float V){ return 1.0f / (1.0f + expf(-(V+35.0f)/10.0f)); }
__device__ float alpha_n(float V){ return 0.01f*(V+55.0f)/(1.0f - expf(-(V+55.0f)/10.0f)); }
__device__ float beta_n(float V){ return 0.125f * expf(-(V+65.0f)/80.0f); }

// constants
__constant__ float C_m = 1.0f;
__constant__ float g_Na = 120.0f;
__constant__ float g_K = 36.0f;
__constant__ float g_L = 0.3f;
__constant__ float E_Na = 50.0f;
__constant__ float E_K = -77.0f;
__constant__ float E_L = -54.387f;

__device__ void ionic_currents(float V, float m, float h, float n, float &Iion){
    float INa = g_Na * m*m*m * h * (V - E_Na);
    float IK  = g_K  * n*n*n*n * (V - E_K);
    float IL  = g_L  * (V - E_L);
    Iion = INa + IK + IL;
}

// compute derivatives for a single neuron (including coupling)
__device__ void hh_deriv(int idx, int N, const float *V_all, float V, float m, float h, float n, float Iext, float g_gap,
                         float &dV, float &dm, float &dh, float &dn){
    // coupling: all-to-all sum (diffusive)
    float sumV = 0.0f;
    for(int j=0;j<N;++j) sumV += V_all[j];
    float coupling = g_gap * (sumV - V * N);
    float Iion;
    ionic_currents(V,m,h,n,Iion);
    dV = (Iext - Iion + coupling) / C_m;
    dm = alpha_m(V)*(1.0f-m) - beta_m(V)*m;
    dh = alpha_h(V)*(1.0f-h) - beta_h(V)*h;
    dn = alpha_n(V)*(1.0f-n) - beta_n(V)*n;
}

__global__ void hh_rk4_step(const float *V_in, const float *m_in, const float *h_in, const float *n_in,
                           float *V_out, float *m_out, float *h_out, float *n_out,
                           const float *Iext, int N, float dt, float g_gap){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;

    // read current state
    float V = V_in[i];
    float m = m_in[i];
    float h = h_in[i];
    float n = n_in[i];

    // RK4 k1
    float k1V,k1m,k1h,k1n;
    hh_deriv(i,N,V_in,V,m,h,n,Iext[i],g_gap,k1V,k1m,k1h,k1n);

    // construct temporary states using k1
    float V2 = V + 0.5f*dt*k1V;
    float m2 = m + 0.5f*dt*k1m;
    float h2 = h + 0.5f*dt*k1h;
    float n2 = n + 0.5f*dt*k1n;

    // k2: note we still use V_in for coupling (explicit RK4 using current global V)
    float k2V,k2m,k2h,k2n;
    hh_deriv(i,N,V_in,V2,m2,h2,n2,Iext[i],g_gap,k2V,k2m,k2h,k2n);

    float V3 = V + 0.5f*dt*k2V;
    float m3 = m + 0.5f*dt*k2m;
    float h3 = h + 0.5f*dt*k2h;
    float n3 = n + 0.5f*dt*k2n;

    float k3V,k3m,k3h,k3n;
    hh_deriv(i,N,V_in,V3,m3,h3,n3,Iext[i],g_gap,k3V,k3m,k3h,k3n);

    float V4 = V + dt*k3V;
    float m4 = m + dt*k3m;
    float h4 = h + dt*k3h;
    float n4 = n + dt*k3n;

    float k4V,k4m,k4h,k4n;
    hh_deriv(i,N,V_in,V4,m4,h4,n4,Iext[i],g_gap,k4V,k4m,k4h,k4n);

    V_out[i] = V + (dt/6.0f)*(k1V + 2.0f*k2V + 2.0f*k3V + k4V);
    m_out[i] = m + (dt/6.0f)*(k1m + 2.0f*k2m + 2.0f*k3m + k4m);
    h_out[i] = h + (dt/6.0f)*(k1h + 2.0f*k2h + 2.0f*k3h + k4h);
    n_out[i] = n + (dt/6.0f)*(k1n + 2.0f*k2n + 2.0f*k3n + k4n);
}

int main(){
    int N = 3;
    float dt = 0.02f;
    float tmax = 200.0f;
    int steps = (int)ceilf(tmax/dt);
    float g_gap = 0.05f;

    std::vector<float> V_host(N), m_host(N), h_host(N), n_host(N), Iext(N);
    // initial conditions similar to Python
    for(int i=0;i<N;i++){
        V_host[i] = -65.0f;
        // approximate steady-state gating
        float Va = V_host[i];
        float am = 0.1f*(Va+40.0f)/(1.0f - expf(-(Va+40.0f)/10.0f));
        float bm = 4.0f * expf(-(Va+65.0f)/18.0f);
        m_host[i] = am/(am+bm);
        float ah = 0.07f * expf(-(Va+65.0f)/20.0f);
        float bh = 1.0f / (1.0f + expf(-(Va+35.0f)/10.0f));
        h_host[i] = ah/(ah+bh);
        float an = 0.01f*(Va+55.0f)/(1.0f - expf(-(Va+55.0f)/10.0f));
        float bn = 0.125f * expf(-(Va+65.0f)/80.0f);
        n_host[i] = an/(an+bn);
        Iext[i] = 0.0f;
    }
    // stimulate neuron 0 briefly
    Iext[0] = 10.0f;

    float *d_V[2], *d_m[2], *d_h[2], *d_n[2], *d_Iext;
    size_t sz = N * sizeof(float);
    cudaMalloc(&d_V[0], sz); cudaMalloc(&d_V[1], sz);
    cudaMalloc(&d_m[0], sz); cudaMalloc(&d_m[1], sz);
    cudaMalloc(&d_h[0], sz); cudaMalloc(&d_h[1], sz);
    cudaMalloc(&d_n[0], sz); cudaMalloc(&d_n[1], sz);
    cudaMalloc(&d_Iext, sz);

    cudaMemcpy(d_V[0], V_host.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m[0], m_host.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h[0], h_host.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n[0], n_host.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Iext, Iext.data(), sz, cudaMemcpyHostToDevice);

    int cur = 0, next = 1;
    int threads = 128;
    int blocks = (N + threads - 1)/threads;

    // store a few samples to host for quick inspection
    std::vector<float> record(steps*N);

    for(int s=0;s<steps;++s){
        float t = s*dt;
        if(t>5.0f){
            // zero the stimulus after 5 ms
            std::vector<float> zeroI(N,0.0f);
            cudaMemcpy(d_Iext, zeroI.data(), sz, cudaMemcpyHostToDevice);
        }
        hh_rk4_step<<<blocks,threads>>>(d_V[cur], d_m[cur], d_h[cur], d_n[cur], d_V[next], d_m[next], d_h[next], d_n[next], d_Iext, N, dt, g_gap);
        cudaDeviceSynchronize();
        // copy back voltages to host for recording (small N)
        cudaMemcpy(V_host.data(), d_V[next], sz, cudaMemcpyDeviceToHost);
        for(int i=0;i<N;i++) record[s*N + i] = V_host[i];
        // swap buffers
        cur = 1 - cur; next = 1 - next;
    }

    // print final voltages
    printf("Final voltages:\n");
    for(int i=0;i<N;i++) printf("neuron %d: %f\n", i, record[(steps-1)*N + i]);

    // cleanup
    cudaFree(d_V[0]); cudaFree(d_V[1]);
    cudaFree(d_m[0]); cudaFree(d_m[1]);
    cudaFree(d_h[0]); cudaFree(d_h[1]);
    cudaFree(d_n[0]); cudaFree(d_n[1]);
    cudaFree(d_Iext);

    return 0;
}
