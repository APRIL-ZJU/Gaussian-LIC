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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// #define DEPTH_GRAD

__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_ddc, glm::vec3* dL_dshs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	// glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	glm::vec3* dL_ddirect_color = dL_ddc + idx;
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	float dRGBdsh0 = SH_C0;
	dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[0] = dRGBdsh1 * dL_dRGB;
		dL_dsh[1] = dRGBdsh2 * dL_dRGB;
		dL_dsh[2] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[2];
		dRGBdy = -SH_C1 * sh[0];
		dRGBdz = SH_C1 * sh[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[3] = dRGBdsh4 * dL_dRGB;
			dL_dsh[4] = dRGBdsh5 * dL_dRGB;
			dL_dsh[5] = dRGBdsh6 * dL_dRGB;
			dL_dsh[6] = dRGBdsh7 * dL_dRGB;
			dL_dsh[7] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] + SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
			dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] + SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
			dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] + SH_C2[3] * x * sh[6];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[8] = dRGBdsh9 * dL_dRGB;
				dL_dsh[9] = dRGBdsh10 * dL_dRGB;
				dL_dsh[10] = dRGBdsh11 * dL_dRGB;
				dL_dsh[11] = dRGBdsh12 * dL_dRGB;
				dL_dsh[12] = dRGBdsh13 * dL_dRGB;
				dL_dsh[13] = dRGBdsh14 * dL_dRGB;
				dL_dsh[14] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[8] * 3.f * 2.f * xy +
					SH_C3[1] * sh[9] * yz +
					SH_C3[2] * sh[10] * -2.f * xy +
					SH_C3[3] * sh[11] * -3.f * 2.f * xz +
					SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[13] * 2.f * xz +
					SH_C3[6] * sh[14] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[8] * 3.f * (xx - yy) +
					SH_C3[1] * sh[9] * xz +
					SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[11] * -3.f * 2.f * yz +
					SH_C3[4] * sh[12] * -2.f * xy +
					SH_C3[5] * sh[13] * -2.f * yz +
					SH_C3[6] * sh[14] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[9] * xy +
					SH_C3[2] * sh[10] * 4.f * 2.f * yz +
					SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[12] * 4.f * 2.f * xz +
					SH_C3[5] * sh[13] * (xx - yy));
			}
		}
	}

	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0)) return;

	const float* cov3D = cov3Ds + 6 * idx;

	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	// const float limx = 1.3f * tan_fovx;
	// const float limy = 1.3f * tan_fovy;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;
	
	// const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	// const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx_pos, max(limx_neg, txtz)) * t.z;
	t.y = min(limy_pos, max(limy_neg, tytz)) * t.z;
	
	const float x_grad_mul = txtz < limx_neg || txtz > limx_pos ? 0 : 1;
	const float y_grad_mul = tytz < limy_neg || tytz > limy_pos ? 0 : 1;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov2D = glm::transpose(T) * Vrk * T;

	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	float dL_dtx = x_grad_mul * -focal_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -focal_y * tz2 * dL_dJ12;
	float dL_dtz = -focal_x * tz2 * dL_dJ00 - focal_y * tz2 * dL_dJ11 + (2 * focal_x * t.x) * tz3 * dL_dJ02 + (2 * focal_y * t.y) * tz3 * dL_dJ12;

	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	dL_dmeans[idx] = dL_dmean;
}

__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	glm::mat3 S = glm::mat3(1.0f);
	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

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

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	// glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	// glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[0][1] + dL_dMt[1][0]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[0][1] + dL_dMt[1][0]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view_matrix,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const float lambda_erank) 
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0)) return;

	float3 p_orig = means[idx];
	float4 p_hom = transformPoint4x4(p_orig, proj);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);

	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * p_orig.x + proj[4] * p_orig.y + proj[8] * p_orig.z + proj[12]) * p_w * p_w;
	float mul2 = (proj[1] * p_orig.x + proj[5] * p_orig.y + proj[9] * p_orig.z + proj[13]) * p_w * p_w;
	dL_dmean.x = (proj[0] * p_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * p_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * p_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * p_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * p_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * p_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	dL_dmeans[idx] += dL_dmean;

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, dc, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_ddc, (glm::vec3*)dL_dsh);

	if (scales)
	{
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
		if (lambda_erank > 0) 
		{
			glm::vec3* dL_dscale_idx = dL_dscale + idx;
			const glm::vec3 scale = scales[idx];
			const float s1s1 = scale.x * scale.x, s2s2 = scale.y * scale.y, s3s3 = scale.z * scale.z;
			const float sum = s1s1 + s2s2 + s3s3;
			const float q1 = scale.x / sum, q2 = scale.y / sum, q3 = scale.z / sum;
			const float erank = exp(-q1 * log(q1) - q2 * log(q2) - q3 * log(q3));
			if (-log(erank - 1 + 1e-5) > 0) 
			{
				glm::vec3 derank_dq = static_cast<float>(erank / (erank - 1 + 1e-5)) * glm::vec3(-log(q1) - 1, -log(q2) - 1, -log(q3) - 1);
				const float lambda_erank_ = lambda_erank * 2.f / (sum * sum);
				dL_dscale_idx->x += lambda_erank_ * scale.x * (derank_dq.x * (s2s2 + s3s3) - derank_dq.y * s2s2 - derank_dq.z * s3s3);
				dL_dscale_idx->y += lambda_erank_ * scale.y * (-derank_dq.x * s1s1 + derank_dq.y * (s1s1 + s3s3) - derank_dq.z * s3s3);
				dL_dscale_idx->z += lambda_erank_ * scale.z * (-derank_dq.x * s1s1 - derank_dq.y * s2s2 + derank_dq.z * (s1s1 + s2s2));
			}
			dL_dscale_idx->z += 1;
		}
	}
}

template<uint32_t C>
__global__ void
PerGaussianRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int B,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	const uint32_t* __restrict__ bucket_to_tile,
	const float* __restrict__ sampled_T, const float* __restrict__ sampled_ar,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ points_depths,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ max_contrib,
	const float* __restrict__ pixel_colors,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors) 
{
	// global_bucket_idx = warp_idx
	auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	// uint32_t global_bucket_idx = block.group_index().x;  //
	bool valid_bucket = global_bucket_idx < (uint32_t) B;
	if (!valid_bucket) return;

	bool valid_splat = false;

	uint32_t tile_id, bbm;
	uint2 range;
	int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile, splat_idx_global;

	tile_id = bucket_to_tile[global_bucket_idx];
	range = ranges[tile_id];
	num_splats_in_tile = range.y - range.x;
	// What is the number of buckets before me? what is my offset?
	bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global = range.x + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) { return; }

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	float2 xy = {0.0f, 0.0f};
	float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
	float c[C] = {0.0f};
	if (valid_splat) 
	{
		gaussian_idx = point_list[splat_idx_global];
		xy = points_xy_image[gaussian_idx];
		con_o = conic_opacity[gaussian_idx];
		for (int ch = 0; ch < C; ++ch)
			c[ch] = colors[gaussian_idx * C + ch];
	}

	// Gradient accumulation variables
	float Register_dL_dmean2D_x = 0.0f;
	float Register_dL_dmean2D_y = 0.0f;
	float Register_dL_dconic2D_x = 0.0f;
	float Register_dL_dconic2D_y = 0.0f;
	float Register_dL_dconic2D_w = 0.0f;
	float Register_dL_dopacity = 0.0f;
	float Register_dL_dcolors[C] = {0.0f};

	// tile metadata
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

	// values useful for gradient calculation
	float T;
	float last_contributor;
	float ar[C];
	float dL_dpixel[C];
	float ad;
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// shared memory
	__shared__ float Shared_sampled_ar[32 * C + 1];  // color
	sampled_ar += global_bucket_idx * BLOCK_SIZE * C;
	__shared__ float Shared_pixels[32 * C];

	// iterate over all pixels in the tile
	// int start = my_warp.meta_group_rank() * (BLOCK_SIZE / my_warp.meta_group_size());
	// int end = start + (BLOCK_SIZE / my_warp.meta_group_size());
	// for (int i = start; i < end + 31; ++i)
	#pragma unroll
	for (int i = 0; i < BLOCK_SIZE + 31; ++i) 
	{
		if (i % 32 == 0)
		{
			for (int ch = 0; ch < C; ++ch) 
			{
				int shift = BLOCK_SIZE * ch + i + block.thread_rank();
				Shared_sampled_ar[ch * 32 + block.thread_rank()] = sampled_ar[shift];
			}
			const uint32_t local_id = i + block.thread_rank();
			const uint2 pix = {pix_min.x + local_id % BLOCK_X, pix_min.y + local_id / BLOCK_X};
			const uint32_t id = W * pix.y + pix.x;
			for (int ch = 0; ch < C; ++ch) 
			{
				Shared_pixels[ch * 32 + block.thread_rank()] = pixel_colors[ch * H * W + id];
			}

			block.sync();
		}

		// SHUFFLING

		// At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
		T = my_warp.shfl_up(T, 1);
		last_contributor = my_warp.shfl_up(last_contributor, 1);
		for (int ch = 0; ch < C; ++ch) 
		{
			ar[ch] = my_warp.shfl_up(ar[ch], 1);
			dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
		}

		// which pixel index should this thread deal with?
		int idx = i - my_warp.thread_rank();
		const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
		const uint32_t pix_id = W * pix.y + pix.x;
		const float2 pixf = {(float) pix.x, (float) pix.y};
		bool valid_pixel = pix.x < W && pix.y < H;
		
		// every 32nd thread should read the stored state from memory
		// TODO: perhaps store these things in shared memory?
		// if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < end)  //
		if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) 
		{
			T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
			int ii = i % 32;
			for (int ch = 0; ch < C; ++ch)
				ar[ch] = -Shared_pixels[ch * 32 + ii] + Shared_sampled_ar[ch * 32 + ii];
			last_contributor = n_contrib[pix_id];
			for (int ch = 0; ch < C; ++ch) 
			{ 
				dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id]; 
			}
		}

		// do work
		// if (valid_splat && valid_pixel && start <= idx && idx < end)  //
		if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) 
		{
			// if (W <= pix.x || H <= pix.y) continue;

			if (splat_idx_in_tile >= last_contributor) continue;

			// compute blending values
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) continue;

			const float dchannel_dcolor = alpha * T;

			// add the gradient contribution of this pixel to the gaussian
			float dL_dalpha = 0.0f;
			float alpha_inverse = 1.0f / (1.0f - alpha);
			for (int ch = 0; ch < C; ++ch) 
			{
				ar[ch] += T * alpha * c[ch];
				const float &dL_dchannel = dL_dpixel[ch];
				Register_dL_dcolors[ch] += dchannel_dcolor * dL_dchannel;
				dL_dalpha += ((c[ch] * T) - alpha_inverse * (-ar[ch])) * dL_dchannel;
			}
			T *= (1.0f - alpha);



			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// accumulate the gradients
			const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
			Register_dL_dmean2D_x += tmp_x;
			const float tmp_y = dL_dG * dG_ddely * ddely_dy;
			Register_dL_dmean2D_y += tmp_y;

			Register_dL_dconic2D_x += -0.5f * gdx * d.x * dL_dG;
			Register_dL_dconic2D_y += -0.5f * gdx * d.y * dL_dG;
			Register_dL_dconic2D_w += -0.5f * gdy * d.y * dL_dG;
			Register_dL_dopacity += G * dL_dalpha;
		}
	}

	// finally add the gradients using atomics
	if (valid_splat) {
		atomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);
		atomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].x, Register_dL_dconic2D_x);
		atomicAdd(&dL_dconic2D[gaussian_idx].y, Register_dL_dconic2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].w, Register_dL_dconic2D_w);
		atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);
		for (int ch = 0; ch < C; ++ch) 
		{
			atomicAdd(&dL_dcolors[gaussian_idx * C + ch], Register_dL_dcolors[ch]);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float limx_neg,
	const float limx_pos,
	const float limy_neg,
	const float limy_pos,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const float lambda_erank) 
{
	computeCov2DCUDA <<<(P + 255) / 256, 256>>> (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		limx_neg,
	    limx_pos,
	    limy_neg,
	    limy_pos,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	preprocessCUDA<NUM_CHAFFELS> <<<(P + 255) / 256, 256>>> (
		P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		dL_dscale,
		dL_drot,
		lambda_erank);
}

void BACKWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int R, int B,
	const uint32_t* per_bucket_tile_offset,
	const uint32_t* bucket_to_tile,
	const float* sampled_T, const float* sampled_ar,
	const float* bg_color,
	const float2* means2D,
	const float* depths,
	const float4* conic_opacity,
	const float* colors,
	const uint32_t* n_contrib,
	const uint32_t* max_contrib,
	const float* pixel_colors,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors) 
{
	const int THREADS = 32;
	PerGaussianRenderCUDA<NUM_CHAFFELS> <<<((B*32) + THREADS - 1), THREADS>>>(
	// const int THREADS = 64;
	// PerGaussianRenderCUDA<NUM_CHAFFELS> <<<B, THREADS>>>(
		ranges,
		point_list,
		W, H, B,
		per_bucket_tile_offset,
		bucket_to_tile,
		sampled_T, sampled_ar,
		bg_color,
		means2D,
		depths,
		conic_opacity,
		colors,
		n_contrib,
		max_contrib,
		pixel_colors,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}