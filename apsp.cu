#include <cstdio>
#include <cassert>

#include "apsp.h"

#define max_dist 10001
#define batch_size 9
#define alt_batch_size 9
#define type short

__device__ int slice_graph(int n, int *g, int i0, int j0) {
	int i = i0 + threadIdx.y, j = j0 + threadIdx.x;
	return (i < n && j < n) ? g[i * n + j] : max_dist;
}

template <typename T>
__device__ void fill_graph(int n, int *g, int i0, int j0, T ele) {
	int i = i0 + threadIdx.y, j = j0 + threadIdx.x;
	if (i < n && j < n)
		g[i * n + j] = ele;
}

__global__ void first_step(int n, int *g, int r_index) {
	__shared__ int c_sub_graph[32][32];
	int c_i = r_index * 32, c_j = r_index * 32, cur = max_dist;
	c_sub_graph[threadIdx.y][threadIdx.x] = slice_graph(n, g, c_i, c_j);
	__syncthreads();
	for (int k = 0; k < 32; k++)
		cur = min(cur, c_sub_graph[threadIdx.y][k] + c_sub_graph[k][threadIdx.x]);
	fill_graph<int>(n, g, c_i, c_j, cur);
}

__global__ void second_step(int n, int *g, int r_index) {
	__shared__ type c_sub_graph[32][32], v_sub_graph[batch_size][32][32], h_sub_graph[batch_size][32][32];
	int c_i = r_index * 32, c_j = r_index * 32, o = blockIdx.x * batch_size * 32, cur = max_dist;
	c_sub_graph[threadIdx.y][threadIdx.x] = slice_graph(n, g, c_i, c_j);
	for (int p = 0, x = o; p < batch_size; p++, x += 32)
		v_sub_graph[p][threadIdx.y][threadIdx.x] = slice_graph(n, g, x, c_j);
	for (int p = 0, y = o; p < batch_size; p++, y += 32)
		h_sub_graph[p][threadIdx.y][threadIdx.x] = slice_graph(n, g, c_i, y);
	__syncthreads();
	for (int p = 0, x = o; p < batch_size; p++, x += 32) {
		cur = max_dist;
		for (int k = 0; k < 32; k++)
			cur = min(cur, v_sub_graph[p][threadIdx.y][k] + c_sub_graph[k][threadIdx.x]);
		fill_graph<type>(n, g, x, c_j, cur);
	}
	for (int p = 0, y = o; p < batch_size; p++, y += 32) {
		cur = max_dist;
		for (int k = 0; k < 32; k++)
			cur = min(cur, c_sub_graph[threadIdx.y][k] + h_sub_graph[p][k][threadIdx.x]);
		fill_graph<type>(n, g, c_i, y, cur);
	}
}

__global__ void third_step(int n, int *g, int r_index) {
	__shared__ type v_sub_graph[alt_batch_size][32][32], h_sub_graph[alt_batch_size][32][32];
	int cur;
	int c_i = r_index * 32, c_j = r_index * 32;
	int x = blockIdx.y * alt_batch_size * 32, y = blockIdx.x * alt_batch_size * 32;
	for (int k = 0, i = x; k < alt_batch_size; k++, i += 32)
		v_sub_graph[k][threadIdx.y][threadIdx.x] = slice_graph(n, g, i, c_j);
	for (int k = 0, j = y; k < alt_batch_size; k++, j += 32)
		h_sub_graph[k][threadIdx.y][threadIdx.x] = slice_graph(n, g, c_i, j);
	__syncthreads();
	for (int p = 0, i = x; p < alt_batch_size; p++, i += 32) {
		for (int q = 0, j = y; q < alt_batch_size; q++, j += 32) {
			cur = slice_graph(n, g, i, j);
			for (int k = 0; k < 32; k++)
				cur = min(cur, v_sub_graph[p][threadIdx.y][k] + h_sub_graph[q][k][threadIdx.x]);
			fill_graph(n, g, i, j, cur);
		}
	}
}

void apsp(int n, /* device */ int *g) {
	dim3 thr(32, 32);
	const int r_cnt = (n + 32 - 1) / 32;
	const int b_cnt = (n - 1) / (batch_size * 32) + 1;
	const int a_cnt = (n - 1) / (alt_batch_size * 32) + 1;
	for (int r_index = 0; r_index < r_cnt; r_index++) {
		first_step<<<1, thr>>>(n, g, r_index);
		second_step<<<b_cnt, thr>>>(n, g, r_index);
		third_step<<<dim3(a_cnt, a_cnt), thr>>>(n, g, r_index);
	}
}
