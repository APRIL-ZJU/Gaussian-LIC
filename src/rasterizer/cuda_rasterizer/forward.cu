/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "forward.h"
#include "auxiliary.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, bool* clamped)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * direct_color[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[3] +
				SH_C2[1] * yz * sh[4] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
				SH_C2[3] * xz * sh[6] +
				SH_C2[4] * (xx - yy) * sh[7];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
					SH_C3[1] * xy * z * sh[9] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
					SH_C3[5] * z * (xx - yy) * sh[13] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
			}
		}
	}

	result += 0.5f;

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, 
							   float limx_neg, float limx_pos, float limy_neg, float limy_pos, const float* cov3D, const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);

	// const float limx = 1.3f * tan_fovx;
	// const float limy = 1.3f * tan_fovy;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;

	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx_pos, max(limx_neg, txtz)) * t.z;
	t.y = min(limy_pos, max(limy_neg, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * Vrk * T;

	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ inline int computeTilebasedCullingTileCount(const bool active, const float4 co_init, const float2 xy_init,  
													   const float opacity_power_threshold_init,
													   const uint2 rect_min_init,  const uint2 rect_max_init) 
{
	const int32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * (rect_max_init.x - rect_min_init.x);
	int tile_count = 0;
	if (active) 
	{
		const uint32_t rect_width = (rect_max_init.x - rect_min_init.x);
		for (int tile_idx = 0; tile_idx < tile_count_init && tile_idx < SEQUENTIAL_TILE_THRESH; tile_idx++) 
		{
			const int y = (tile_idx / rect_width) + rect_min_init.y;
			const int x = (tile_idx % rect_width) + rect_min_init.x;

			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};

			glm::vec2 max_pos;
			float max_opac_factor = max_contrib_power_rect_gaussian_float(co_init, xy_init, tile_min, tile_max, max_pos);
			tile_count += (max_opac_factor <= opacity_power_threshold_init);
		}
	}

	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	const uint32_t warp_idx = cg::this_thread_block().thread_rank() / WARP_SIZE;

	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);
	if (remaining_threads == 0)
		return tile_count;

	const uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++) 
	{
		const uint32_t i = __fns(remaining_threads, 0, n+1); // find lane index of next remaining thread

		const uint2 rect_min = make_uint2(__shfl_sync(WARP_MASK, rect_min_init.x, i), __shfl_sync(WARP_MASK, rect_min_init.y, i));
		const uint2 rect_max = make_uint2(__shfl_sync(WARP_MASK, rect_max_init.x, i), __shfl_sync(WARP_MASK, rect_max_init.y, i));
		const float2 xy = { __shfl_sync(WARP_MASK, xy_init.x, i), __shfl_sync(WARP_MASK, xy_init.y, i) };

		const float4 co = 
		{
			__shfl_sync(WARP_MASK, co_init.x, i),
			__shfl_sync(WARP_MASK, co_init.y, i),
			__shfl_sync(WARP_MASK, co_init.z, i),
			__shfl_sync(WARP_MASK, co_init.w, i),
		};
		const float opacity_power_threshold = __shfl_sync(WARP_MASK, opacity_power_threshold_init, i);


		const uint32_t rect_width = (rect_max.x - rect_min.x);
		const uint32_t rect_tile_count = (rect_max.y - rect_min.y) * rect_width;
		const uint32_t remaining_rect_tile_count = rect_tile_count - SEQUENTIAL_TILE_THRESH;

		const int32_t n_iterations = (remaining_rect_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		for (int it = 0; it < n_iterations; it++) 
		{
			const int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			const int active_curr_it = tile_idx < rect_tile_count;

			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;

			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};

			glm::vec2 max_pos;
			const float max_opac_factor = max_contrib_power_rect_gaussian_float(co, xy, tile_min, tile_max, max_pos);

			const uint32_t tile_contributes = active_curr_it && max_opac_factor <= opacity_power_threshold;

			const uint32_t contributes_ballot = __ballot_sync(WARP_MASK, tile_contributes);
			const uint32_t n_contribute = __popc(contributes_ballot);

			tile_count += (i == lane_idx) * n_contribute;
		}
	}

	return tile_count;
}

template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* dc,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered, bool no_color) 
{
	auto idx = cg::this_grid().thread_rank();
	bool active = true;
	if (idx >= P) 
	{
		active = false;
		idx = P - 1;
	}
	
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) { active = false; }

	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	const float* cov3D = cov3Ds + idx * 6;
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, limx_neg, limx_pos, limy_neg, limy_pos, cov3D, viewmatrix);

	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) { active = false; }
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	const float4 co = { conic.x, conic.y, conic.z, opacities[idx] };
	if (co.w < OPACITY_THRESHOLD) { active = false; }
	if (__ballot_sync(WARP_MASK, active) == 0) return;  // early stop if whole warp culled
	
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(lambda1));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	const float opacity_factor_threshold = logf(co.w / OPACITY_THRESHOLD);
	const int tile_count = computeTilebasedCullingTileCount(active, co, point_image,  opacity_factor_threshold, rect_min, rect_max);
	if (tile_count == 0 || !active) return;  // Cooperative threads no longer needed (after load balancing)

	if (!no_color) 
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, dc, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic_opacity[idx] = co;
	tiles_touched[idx] = tile_count;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset, uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T, float* __restrict__ sampled_ar,
	int W, int H,
	const float2* __restrict__ points_xy_image, 
	const float* __restrict__ depths,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_final_T,
	bool no_color) 
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// what is the number of buckets before me? what is my offset?
	uint32_t bbm = 0;
	if (!no_color) 
	{
		bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
		// let's first quickly also write the bucket-to-tile mapping
		int num_buckets = (toDo + 31) / 32;
		for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) 
		{
			int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
			if (bucket_idx < num_buckets) 
			{
				bucket_to_tile[bbm + bucket_idx] = tile_id;
			}
		}
	}

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t contributor_real = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) 
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) 
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) 
		{
			// add incoming T value for every 32nd gaussian
			if (j % 32 == 0 && !no_color) 
			{
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;  //
				for (int ch = 0; ch < CHANNELS; ++ch) 
				{
					sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];  //
				}
				++bbm;
			}

			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) { continue; }
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) { continue; }
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) 
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			if (!no_color) 
			{
				for (int ch = 0; ch < CHANNELS; ch++) 
					C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				contributor_real++;
			}

			T = test_T;
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside) 
	{
		out_final_T[pix_id] = T;
		if (!no_color) 
		{
			n_contrib[pix_id] = last_contributor;
			for (int ch = 0; ch < CHANNELS; ch++) 
				out_color[ch * H * W + pix_id] = C[ch];
		}
	}
	if (no_color) { return; }

	// max reduce the last contributor
    // typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
	typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduce;  //
    __shared__ typename BlockReduce::TempStorage temp_storage;
    last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0) 
	{
		max_contrib[tile_id] = last_contributor;
	}
}


void FORWARD::render( const dim3 grid, dim3 block, const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
	float* sampled_T, float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* depths,
	const float* colors,
	const float4* conic_opacity,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	const float* bg_color,
	float* out_color, 
	float* out_final_T,
	bool no_color) 
{
	renderCUDA<NUM_CHAFFELS> <<<grid, block>>> (
		ranges,
		point_list,
		per_tile_bucket_offset, bucket_to_tile,
		sampled_T, sampled_ar,
		W, H,
		means2D,
		depths,
		colors,
		conic_opacity,
		n_contrib,
		max_contrib,
		bg_color,
		out_color,
		out_final_T,
		no_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* dc,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered, bool no_color)
{
	preprocessCUDA<NUM_CHAFFELS> <<<(P + 255) / 256, 256>>> (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		dc,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		limx_neg,
		limx_pos,
		limy_neg,
		limy_pos,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered, no_color);
}