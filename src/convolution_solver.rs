//
// Generate a convoluted image from a scan
//
// This is not a full reconstruction algorithm - it generates a
// blurred image from a scan, and works on the observation that if you
// integrate over all the rays passing through a point, you get the value
// of the image at that point filtered through a 1/r-weighted filter.
//
// A full reconstruction algorithm can then be made by deconvolving
// this image.
//

use crate::tomo_image::Image;
use crate::tomo_scan::{Scan, scan};

// Integrate across all the rays that pass through a given point.
// Given point is in (-1..1, -1..1) coordinates.
fn circular_integral(scan: &Scan, x: f64, y: f64) -> f64 {
    let mut total = 0.0;

    // Consistent with generate_forwards_matrix, from which much of
    // the code was copied (*sigh* - so much for DRY).
    let angle_step = std::f64::consts::PI / scan.angles as f64;
    // Image is inside an axis-aligned square -1..1, so max radius is sqrt(2).
    let ray_offset = 2_f64.sqrt();
    let ray_step_inv = (scan.rays - 1) as f64 / (2.0 * ray_offset);
    for angle_num in 0..scan.angles {
        let angle = angle_num as f64 * angle_step;

        // 1. For each angle, find the ray numbers closest to our point:

        // a) Generate a unit vector normal to the direction of the rays.
        let normal = (-angle.sin(), angle.cos());

        // b) Dot product it with the vector to the point to get the
        // (signed) size of the component normal to the ray direction,
        // which represents the distace of closest approach.
        let dist = normal.0 * x + normal.1 * y;

        // c) Convert this distance to a ray number - a floating point
        // verison of the index into the range of rays.
        let ray_num = (dist + ray_offset) * ray_step_inv;

        // 2. Then we interpolate between those rays to approximate
        // the contribution of the ray at that angle passing through
        // the point.
        let r0 = ray_num.floor() as isize;
        let r1 = r0 + 1;
        let fract = ray_num - (r0 as f64);
        // fract is in [0.0, 1.0).
        // We linearly interpolate by adding (1.0 - fract) * w(r0) + fract * w(r1).
        for (wt, ray_num) in [((1.0 - fract), r0), (fract, r1)].iter() {
            if 0 <= *ray_num && (*ray_num as usize) < scan.rays {
                let idx = angle_num * scan.rays + *ray_num as usize;
                total += wt * scan.data[idx];
            }
        }
    }

    total / scan.angles as f64
}

// Generate the circular integrals of all points in a grid from a
// Scan. The result is a convolved image.
//
// Not a real attempt at reconstruction, we just do the
// convolution-generation step and return it without attempting to
// deconvolve. It does give a nice blurry version of the original!
pub fn reconstruct(scan: &Scan, width: usize, height: usize) -> Image {
    let mut data = Vec::new();

    let y_scale = 2.0 / height as f64;
    let y_offset = -1.0;
    // Sample at the centre of the pixels - add 0.5
    for y in (0..height).map(|idx| (idx as f64 + 0.5) * y_scale + y_offset) {
        let x_scale = 2.0 / width as f64;
        let x_offset = -1.0;
        for x in (0..width).map(|idx| (idx as f64 + 0.5) * x_scale + x_offset) {
            data.push(circular_integral(scan, x, y));
        }
    }

    Image {
        width,
        height,
        data,
    }
}

// Build an image that represents the convolution filter applied by
// converting an image and then running generate_convolved_tomo on it.
//
// Not an actual Image as we want to stay in the f64 domain.
//
// Width and height are the width and height of the area representing
// (-1..1, -1..1). "Overscan" is the fraction extra we slap on each
// edge because the filter doesn't ever fully drop off to zero.
//
// Weights are set such that integrating over a unit disc should
// produce 1.0.
pub fn build_convolution_filter(
    width: usize,
    height: usize,
    overscan: f64,
) -> Image {
    let w_overscan = (width as f64 * overscan).ceil() as usize;
    let h_overscan = (height as f64 * overscan).ceil() as usize;

    let w_full = width + 2 * w_overscan;
    let h_full = height + 2 * h_overscan;

    let x_step = 2.0 / width as f64;
    let y_step = 2.0 / height as f64;

    // Sample the centre of pixels - offset by 0.5.
    let x_offset = (w_full as f64 - 1.0) / 2.0;
    let y_offset = (h_full as f64 - 1.0) / 2.0;

    let mut res = Vec::new();

    // "Fudge" boosts the radius by a tiny amount to avoid a
    // singularity at the origin, since we're really trying to
    // sample areas rather than points.
    //
    // We make the radius bump half the "radius" of a pixel.
    let fudge = (x_step * x_step + y_step * y_step).sqrt() / 4.0;

    for y in (0..h_full).map(|y| (y as f64 - y_offset) * y_step) {
        for x in (0..w_full).map(|x| (x as f64 - x_offset) * x_step) {
            let r = (x * x + y * y).sqrt().max(fudge);
            let r_frac = r * 2.0 * std::f64::consts::PI;
            let weight = (x_step * y_step) / r_frac;
            res.push(weight);
        }
    }

    // We could normalise, so that the integral inside the unit disc
    // is precisely 1.0, but we're within 0.5% with a decent-sized grid,
    // which is good enough for what we're doing here.

    Image {
        width: w_full,
        height: h_full,
        data: res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Check that the integral over the convolution filter is what we expect
    // (integral = r).
    #[test]
    fn test_convolution_integral_even() {
        let (base_width, base_height) = (128, 128);
        let overscan = 0.3;
        let filter = build_convolution_filter(base_width, base_height, overscan);
        assert!(filter.width % 2 == 0);
        assert!(filter.height % 2 == 0);

        // Yet more C&P of the pixel iterator code...
        let x_step = 2.0 / base_width as f64;
        let y_step = 2.0 / base_height as f64;

        // Sample the centre of pixels - offset by 0.5.
        let x_offset = (filter.width as f64 - 1.0) / 2.0;
        let y_offset = (filter.height as f64 - 1.0) / 2.0;

        let mut integral_120 = 0.0;
        let mut integral_100 = 0.0;
        let mut integral_070 = 0.0;
        let mut integral_050 = 0.0;

        for y_idx in 0..filter.height {
            let y = (y_idx as f64 - y_offset) * y_step;
            for x_idx in 0..filter.width {
                let x = (x_idx as f64 - x_offset) * x_step;

                let r = (x * x + y * y).sqrt();
                let weight = filter.data[y_idx * filter.width + x_idx];
                if r <= 1.2 {
                    integral_120 += weight;
                }
                if r <= 1.0 {
                    integral_100 += weight;
                }
                if r <= 0.7 {
                    integral_070 += weight;
                }
                if r <= 0.5 {
                    integral_050 += weight;
                }
            }
        }

        assert!((integral_120 - 1.2).abs() < 0.005);
        assert!((integral_100 - 1.0).abs() < 0.005);
        assert!((integral_070 - 0.7).abs() < 0.005);
        assert!((integral_050 - 0.5).abs() < 0.005);
    }

    // Like test_convolution_integral_even, but for an odd grid size,
    // which means we sample on the singularity at r = 0.
    #[test]
    fn test_convolution_integral_odd() {
        let (base_width, base_height) = (129, 129);
        let overscan = 0.3;
        let filter = build_convolution_filter(base_width, base_height, overscan);
        assert!(filter.width % 2 == 1);
        assert!(filter.height % 2 == 1);

        // Yet more C&P of the pixel iterator code...
        let x_step = 2.0 / base_width as f64;
        let y_step = 2.0 / base_height as f64;

        // Sample the centre of pixels - offset by 0.5.
        let x_offset = (filter.width as f64 - 1.0) / 2.0;
        let y_offset = (filter.height as f64 - 1.0) / 2.0;

        let mut integral_120 = 0.0;
        let mut integral_100 = 0.0;
        let mut integral_070 = 0.0;
        let mut integral_050 = 0.0;

        for y_idx in 0..filter.height {
            let y = (y_idx as f64 - y_offset) * y_step;
            for x_idx in 0..filter.width {
                let x = (x_idx as f64 - x_offset) * x_step;

                let r = (x * x + y * y).sqrt();
                let weight = filter.data[y_idx * filter.width + x_idx];
                if r <= 1.2 {
                    integral_120 += weight;
                }
                if r <= 1.0 {
                    integral_100 += weight;
                }
                if r <= 0.7 {
                    integral_070 += weight;
                }
                if r <= 0.5 {
                    integral_050 += weight;
                }
            }
        }

        assert!((integral_120 - 1.2).abs() < 0.005);
        assert!((integral_100 - 1.0).abs() < 0.005);
        assert!((integral_070 - 0.7).abs() < 0.005);
        assert!((integral_050 - 0.5).abs() < 0.005);
    }

    // Check that the convolution filter is symmetric.
    #[test]
    fn test_convolution_symmetric() {
        let (base_width, base_height) = (128, 129);
        let overscan = 0.3;
        let filter = build_convolution_filter(base_width, base_height, overscan);
        assert!(filter.width % 2 == 0);
        assert!(filter.height % 2 == 1);

        for y1 in 0..filter.height {
            let y2 = filter.height - 1 - y1;

            for x1 in 0..filter.width {
                let x2 = filter.width - 1 - x1;

                let p11 = filter.data[y1 * filter.width + x1];
                let p12 = filter.data[y1 * filter.width + x2];
                let p21 = filter.data[y2 * filter.width + x1];
                let p22 = filter.data[y2 * filter.width + x2];

                assert_eq!(p11, p12);
                assert_eq!(p11, p21);
                assert_eq!(p11, p22);
            }
        }
    }

    // Check that the core region of the overscan is the same as the
    // non-overscanned version.
    #[test]
    fn test_convolution_overscan() {
        let (base_width, base_height) = (128, 129);
        let overscan = 0.3;
        let filter = build_convolution_filter(base_width, base_height, overscan);
        let filter_no_over = build_convolution_filter(base_width, base_height, 0.0);
        assert_eq!(filter_no_over.width, base_width);
        assert_eq!(filter_no_over.height, base_height);

        let w_adj = (filter.width - base_width) / 2;
        let h_adj = (filter.height - base_height) / 2;

        for y in 0..filter_no_over.height {
            let y2 = y + h_adj;
            for x in 0..filter_no_over.width {
                let x2 = x + w_adj;

                let p1 = filter_no_over.data[y * filter_no_over.width + x];
                let p2 = filter.data[y2 * filter.width + x2];

                assert_eq!(p1, p2);
            }
        }
    }

    // Scales down the image by the given factor, which must divide
    // width and height.
    //
    // TODO: Currently only used for testing, but may come in handy...
    // maybe move to Image?
    fn downscale(image: &Image, factor: usize) -> Vec<f64> {
        assert!(image.width % factor == 0);
        assert!(image.height % factor == 0);

        let new_w = image.width / factor;
        let new_h = image.height / factor;

        let mut res = Vec::new();
        for y in (0..new_h).map(|y| y * factor) {
            for x in (0..new_w).map(|x| x * factor) {
                let mut pixel: f64 = 0.0;
                for sub_y in 0..factor {
                    for sub_x in 0..factor {
                        pixel += image.data[(y + sub_y) * image.width + (x + sub_x)];
                    }
                }
                res.push(pixel / (factor as f64 * factor as f64));
            }
        }
        res
    }

    // Test that the convolution filter we generate is more-or-less
    // the same as what we get by scanning a point and convoluting it.
    #[test]
    fn test_convolution_comparison() {
        let (width, height) = (65, 65);
        let oversample_factor = 5;

        // Generate a convolution filter, no overscan. Integral up to
        // radius 1.0 is 1.0
        let generated = build_convolution_filter(width, height, 0.0).data;

        // Generate an image with a circle at the centre. We will later
        // scale it down to a circle one pixel in diameter, to be close
        // to what the convolution filter is generating.
        let (o_width, o_height) = (width * oversample_factor, height * oversample_factor);

        let mut circle_area = 0.0;

        let image = (0..o_width * o_height).map(|idx| {
            let (x_idx, y_idx) = (idx % o_width, idx / o_width);
            // Adjust to centre of pixels.
            let x = x_idx as f64 + 0.5 - o_width as f64 / 2.0;
            let y = y_idx as f64 + 0.5 - o_height as f64 / 2.0;

            // Set radius = oversample_factor / 2 pixels,
            // diameter = oversample_factor.
            let r2 = x * x + y * y;
            if r2 <= oversample_factor as f64 * oversample_factor as f64 / 4.0 {
                circle_area += 1.0;
                1.0
            } else {
                0.0
            }
        }).collect::<Vec<_>>();

        let src_img = Image {
            width: o_width,
            height: o_height,
            data: image,
        };

        // Make sure we don't lose precision in the scan.
        let rays = o_width * 2;
        let angles = o_height * 2;
        let scan = scan(&src_img, angles, rays);
        let reconstructed = reconstruct(&scan, o_width, o_height);

        // The resulting image's integral up to radius 1.0 is circle_area.

        // Downscaling the image reduces the integral by oversample_factor^2.
        let downscaled = downscale(&reconstructed, oversample_factor);

        // Then normalise to the same scale as the generated filter by
        // dividing through by circle_area / oversample_factor^2
        let scale_factor = (oversample_factor as f64 * oversample_factor as f64) / circle_area;
        let normalised = downscaled
            .iter()
            .map(|p| p * scale_factor)
            .collect::<Vec<_>>();

        // Calculate the total error, integrated over the full image.
        assert_eq!(generated.len(), normalised.len());
        let error: f64 = generated
            .iter()
            .zip(normalised.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // Given the integral within unit radius is 1, and the
        // integral over the whole square is a bit more, this is not
        // far off a 1% error. If oversample_factor is 9, the error
        // reduces to around 0.01. Given the amount of half-assed
        // numerical methods that I'm doing here, having an error
        // around 1% seems good enough to me! I think the algorithms
        // do pretty much match.
        assert!(error < 0.016);
    }
}
