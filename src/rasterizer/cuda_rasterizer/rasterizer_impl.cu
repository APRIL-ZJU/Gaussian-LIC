/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__global__ void duplicateWithKeys( 
	int P, 
	const float2* points_xy, 
	const float4* __restrict__ conic_opacity, 
	const float* depths, 
	const uint32_t* offsets, 
	uint64_t* gaussian_keys_unsorted, 
	uint32_t* gaussian_values_unsorted, 
	int* radii, 
	dim3 grid, 
	int2* rects) 
{
	auto idx = cg::this_grid().thread_rank();
	bool active =  true;
	if (idx >= P) 
	{
		active = false;
		idx = P - 1;
	}
	if (radii[idx] <= 0) { active = false; }
	if (__ballot_sync(WARP_MASK, active) == 0) { return; }

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	const uint32_t offset_to = offsets[idx];

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	const float2 xy = points_xy[idx];
	const float4 co = conic_opacity[idx];
	const float opacity_factor_threshold = logf(co.w / OPACITY_THRESHOLD);
	const uint32_t rect_width = (rect_max.x - rect_min.x);
	const int32_t tile_count_init = (rect_max.y - rect_min.y) * rect_width;
	if (active) 
	{
		for (int tile_idx = 0; tile_idx < tile_count_init && tile_idx < SEQUENTIAL_TILE_THRESH && off < offset_to; tile_idx++) 
		{
			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;
			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};
			glm::vec2 max_pos;
			float max_opac_factor = max_contrib_power_rect_gaussian_float(co, xy, tile_min, tile_max, max_pos);
			if (max_opac_factor <= opacity_factor_threshold) 
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}

	// load balance
	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	unsigned int lane_mask_allprev_excl = 0xFFFFFFFFU >> (WARP_SIZE - lane_idx);
	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);
	if (remaining_threads == 0) 
	{ 
		while (off < offset_to) 
		{
			uint64_t key = (uint32_t) -1;
			key <<= 32;
			const float depth = FLT_MAX;
			key |= *((uint32_t*)&depth);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
			off++;
		}
		return; 
	}

	uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++) 
	{
		int i = __fns(remaining_threads, 0, n + 1); 
		uint32_t idx_i = __shfl_sync(WARP_MASK, idx, i);
		uint32_t off_i = __shfl_sync(WARP_MASK, off, i);
		const uint32_t offset_to_i = __shfl_sync(WARP_MASK, offset_to, i);

		const uint2 rect_min_i = make_uint2(__shfl_sync(WARP_MASK, rect_min.x, i), __shfl_sync(WARP_MASK, rect_min.y, i));
		const uint2 rect_max_i = make_uint2(__shfl_sync(WARP_MASK, rect_max.x, i), __shfl_sync(WARP_MASK, rect_max.y, i));
		const float2 xy_i = { __shfl_sync(WARP_MASK, xy.x, i), __shfl_sync(WARP_MASK, xy.y, i) };
		const float4 co_i = 
		{
			__shfl_sync(WARP_MASK, co.x, i),
			__shfl_sync(WARP_MASK, co.y, i),
			__shfl_sync(WARP_MASK, co.z, i),
			__shfl_sync(WARP_MASK, co.w, i),
		};
		const float opacity_factor_threshold_i = __shfl_sync(WARP_MASK, opacity_factor_threshold, i);
		const uint32_t rect_width_i = (rect_max_i.x - rect_min_i.x);
		const uint32_t tile_count_i = (rect_max_i.y - rect_min_i.y) * rect_width_i;

		const uint32_t remaining_tile_count = tile_count_i - SEQUENTIAL_TILE_THRESH;
		const int32_t n_iterations = (remaining_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		for (int it = 0; it < n_iterations; it++) 
		{
			int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			int y = (tile_idx / rect_width_i) + rect_min_i.y;
			int x = (tile_idx % rect_width_i) + rect_min_i.x;
			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};
			glm::vec2 max_pos;
			bool write = (tile_idx < tile_count_i) && (max_contrib_power_rect_gaussian_float(co_i, xy_i, tile_min, tile_max, max_pos) <= opacity_factor_threshold_i);
			const uint32_t write_ballot = __ballot_sync(WARP_MASK, write);
			const uint32_t n_writes = __popc(write_ballot);
			const uint32_t write_offset = off_i + __popc(write_ballot & lane_mask_allprev_excl);
			if (write && write_offset < offset_to_i) 
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx_i]);
				gaussian_keys_unsorted[write_offset] = key;
				gaussian_values_unsorted[write_offset] = idx_i;
			}
			off_i += n_writes;
			off += (i == lane_idx) * n_writes;
		}
	}

	while (off < offset_to) 
	{
		uint64_t key = (uint32_t) -1;
		key <<= 32;
		const float depth = FLT_MAX;
		key |= *((uint32_t*)&depth);
		gaussian_keys_unsorted[off] = key;
		gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
		off++;
	}
}

__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L) return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile != (uint32_t) -1;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			if (valid_tile) 
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
}

// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) 
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;
	
	uint2 range = ranges[idx];
	int num_splats = range.y - range.x;
	int num_buckets = (num_splats + 31) / 32;
	bucketCount[idx] = (uint32_t) num_buckets;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N, size_t M)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, M, 128);
	int* dummy;
	int* wummy;
	cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N);
	obtain(chunk, img.contrib_scan, img.scan_size, 128);

	obtain(chunk, img.max_contrib, N, 128);
	obtain(chunk, img.pixel_colors, N * NUM_CHAFFELS, 128);
	obtain(chunk, img.bucket_count, N, 128);
	obtain(chunk, img.bucket_offsets, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

	return img;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char *& chunk, size_t C) {
	SampleState sample;
	obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHAFFELS * C * BLOCK_SIZE, 128);
	return sample;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__global__ void zero(int N, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;
	space[idx] = 0;
}

__global__ void set(int N, uint32_t* where, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;

	int off = (idx == 0) ? 0 : where[idx-1];

	space[off] = 1;
}

std::tuple<int,int> CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> sampleBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	const bool prefiltered,
	float* out_color,
	float* out_final_T,
	int* radii,
	bool debug, bool no_color) 
{
	if (NUM_CHAFFELS != 3 && colors_precomp == nullptr) 
	{ 
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!"); 
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 block(BLOCK_X, BLOCK_Y, 1);
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	size_t geom_chunk_size = required<GeometryState>(P);
	char* geom_chunkptr = geometryBuffer(geom_chunk_size);
	GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

	size_t img_chunk_size = required<ImageState>(width * height, tile_grid.x * tile_grid.y);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height, tile_grid.x * tile_grid.y);

	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		limx_neg,
		limx_pos,
		limy_neg,
		limy_pos,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered, 
		no_color
	), debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	duplicateWithKeys <<<(P + 255) / 256, 256>>> (
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		nullptr)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);  // TODO

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	if (num_rendered > 0)
		identifyTileRanges <<<(num_rendered + 255) / 256, 256>>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	SampleState sampleState;
	unsigned int bucket_sum = 0;
	if (!no_color) 
	{
		int num_tiles = tile_grid.x * tile_grid.y;
		perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug)
		CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
		// create a state to store. size is number is the total number of buckets * block_size
		size_t sample_chunk_size = required<SampleState>(bucket_sum);
		char* sample_chunkptr = sampleBuffer(sample_chunk_size);
		sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);
	}

	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets, 
		sampleState.bucket_to_tile,
		sampleState.T, 
		sampleState.ar,
		width, height,
		geomState.means2D,
		geomState.depths,
		feature_ptr,
		geomState.conic_opacity,
		imgState.n_contrib,
		imgState.max_contrib,
		background,
		out_color, out_final_T, no_color), debug)

	if (!no_color) 
	{
		// out_color -> imgState.pixel_colors
		CHECK_CUDA(cudaMemcpy(imgState.pixel_colors, out_color, sizeof(float) * width * height * NUM_CHAFFELS, cudaMemcpyDeviceToDevice), debug);
	}
	return std::make_tuple(num_rendered, bucket_sum);
}

void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	char* sample_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	const float lambda_erank,
	bool debug) 
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height, tile_grid.x * tile_grid.y);
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B);

	const float* feature_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, R, B,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		background,
		geomState.means2D,
		geomState.depths,
		geomState.conic_opacity,
		feature_ptr,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		limx_neg,
		limx_pos,
		limy_neg,
		limy_pos,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		lambda_erank), debug)
}