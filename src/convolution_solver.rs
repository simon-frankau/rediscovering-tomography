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

use std::path::Path;

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

// Helper function that, given a size and amount of overscan, and a
// function that goes from -1..1 coordinates to value, will generate an
// image.
fn build_image<F: Fn(f64, f64) -> f64>(
   width: usize,
   height: usize,
   width_overscan: usize,
   height_overscan: usize,
   value_fn: F,
) -> Image {
    // Overscan is applied on both edges
    let w_full = width + 2 * width_overscan;
    let h_full = height + 2 * height_overscan;

    let y_step = 2.0 / height as f64;
    let x_step = 2.0 / width as f64;

    // Sample at the centre of the pixels - add 0.5.
    let y_offset = (h_full as f64 - 1.0)/ 2.0;
    let x_offset = (w_full as f64 - 1.0)/ 2.0;

    let mut data = Vec::new();

    for y in (0..h_full).map(|idx| (idx as f64 - y_offset) * y_step) {
        for x in (0..w_full).map(|idx| (idx as f64 - x_offset) * x_step) {
            data.push(value_fn(x, y));
        }
    }

    Image {
        width: w_full,
        height: h_full,
        data,
    }
}

// Generate the circular integrals of all points in a grid from a
// Scan. The result is a convolved image. The x_overscan and y_overscan
// are the number of pixels added onto the left and right, and top and
// bottom margins respectively, since the convoluted image is still
// non-zero outside the original image, and this can be used in
// reconstruction. Image::trim can be used to trim it back to a
// width x height image at the right scale.
pub fn reconstruct_overscan(
    scan: &Scan,
    width: usize,
    height: usize,
    x_overscan: usize,
    y_overscan: usize
) -> Image {
    build_image(
        width,
        height,
        x_overscan,
        y_overscan,
        |x, y| circular_integral(scan, x, y))
}

// Generate the circular integrals of all points in a grid from a
// Scan. The result is a convolved image. No overscan.
//
// Not a real attempt at reconstruction, we just do the
// convolution-generation step and return it without attempting to
// deconvolve. It does give a nice blurry version of the original!
pub fn reconstruct(scan: &Scan, width: usize, height: usize) -> Image {
    reconstruct_overscan(scan, width, height, 0, 0)
}

// Build an image that represents the convolution filter applied by
// converting an image and then running generate_convolved_tomo on it.
//
// Not an actual Image as we want to stay in the f64 domain.
//
// Width and height are the width and height of the area representing
// (-1..1, -1..1). Overscan is in pixels-per-side, like for
// reconstruct_overscan. It's useful because the filter doesn't ever
// drop off to zero, so our calculations may extend beyond the core
// image.
//
// Weights are set such that integrating over a unit disc should
// produce 1.0.
pub fn build_convolution_filter(
    width: usize,
    height: usize,
    x_overscan: usize,
    y_overscan: usize
) -> Image {
    // "Fudge" boosts the radius by a tiny amount to avoid a
    // singularity at the origin, since we're really trying to
    // sample areas rather than points.
    //
    // We make the radius bump half the "radius" of a pixel.
    let y_step = 2.0 / height as f64;
    let x_step = 2.0 / width as f64;
    let fudge = (x_step * x_step + y_step * y_step).sqrt() / 4.0;

    build_image(
        width,
        height,
        x_overscan,
        y_overscan,
        | x, y | {
            let r = (x * x + y * y).sqrt().max(fudge);
            let r_frac = r * 2.0 * std::f64::consts::PI;
            (x_step * y_step) / r_frac
        })

    // We could normalise the image, so that the integral inside the
    // unit disc is precisely 1.0, but we're within 0.5% with a
    // decent-sized grid, which is good enough for what we're doing
    // here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruct() {
        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let (rays, angles) = (35, 40);

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let (width, height) = (src_img.width, src_img.height);

        let dst_img = reconstruct(&scan, width, height);

        // Fairly chunky error, but this is to be expected when we
        // don't have a real reconstruction, but a blurry-filtered
        // version of the original.
        let average_error = src_img.average_diff(&dst_img);
        assert!(30.0 < average_error && average_error < 40.0);
    }

    #[test]
    fn test_reconstruct_overscan() {
        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let (rays, angles) = (35, 40);

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let (width, height) = (src_img.width, src_img.height);

        let dst_img = reconstruct(&scan, width, height);

        let (x_over, y_over) = (5, 7);
        let dst_img_overscan =
            reconstruct_overscan(&scan, width, height, x_over, y_over);

        let dst_img_trimmed =
            dst_img_overscan.trim(x_over, y_over, width, height);

        for (p1, p2) in dst_img.data.iter().zip(dst_img_trimmed.data.iter()) {
            assert_eq!(p1, p2);
        }
    }

    // Most of the tests exercise overscan, since we want to make sure
    // the full functionality works. This helper adds 30% extra pixels
    // all around the border.
    fn build_convolution_filter_overscan(w: usize, h: usize) -> Image {
        let overscan = 0.3;
        let w_over = (w as f64 * overscan).ceil() as usize;
        let h_over = (h as f64 * overscan).ceil() as usize;
        let filter = build_convolution_filter(w, h, w_over, h_over);

        // Since our tests want to exercise the centre being pixel-aligned
        // or not, check the parity is preserved when adding a border.
        assert!(filter.width % 2 == w % 2);
        assert!(filter.height % 2 == h % 2);

        filter
    }


    // Helper function, like build_image, except it passes over
    // an existing image, calling a closure with coordinates and
    // value at a point. Pass in the non-overscanned width and height.
    fn scan_image<F: FnMut(f64, f64, f64)> (
       image: &Image,
       base_width: usize,
       base_height: usize,
       mut value_fn: F,
    ) {
        let y_step = 2.0 / base_width as f64;
        let x_step = 2.0 / base_height as f64;

        // Sample at the centre of the pixels - add 0.5.
        let y_offset = (image.height as f64 - 1.0)/ 2.0;
        let x_offset = (image.width as f64 - 1.0)/ 2.0;

        for y_idx in 0..image.height {
            let y = (y_idx as f64 - y_offset) * y_step;
            for x_idx in 0..image.width {
            let x = (x_idx as f64 - x_offset) * x_step;
                value_fn(x, y, image[(x_idx, y_idx)]);
            }
        }
    }

    // Check that the integral over the convolution filter is what we expect
    // (integral = r).
    #[test]
    fn test_convolution_integral_even() {
        let (base_w, base_h) = (128, 128);
        let filter = build_convolution_filter_overscan(base_w, base_h);

        let mut integral_120 = 0.0;
        let mut integral_100 = 0.0;
        let mut integral_070 = 0.0;
        let mut integral_050 = 0.0;

        scan_image(&filter, base_w, base_h, |x, y, p| {
            let r = (x * x + y * y).sqrt();
            if r <= 1.2 {
                integral_120 += p;
            }
            if r <= 1.0 {
                integral_100 += p;
            }
            if r <= 0.7 {
                integral_070 += p;
            }
            if r <= 0.5 {
                integral_050 += p;
            }
        });

        assert!((integral_120 - 1.2).abs() < 0.005);
        assert!((integral_100 - 1.0).abs() < 0.005);
        assert!((integral_070 - 0.7).abs() < 0.005);
        assert!((integral_050 - 0.5).abs() < 0.005);
    }

    // Like test_convolution_integral_even, but for an odd grid size,
    // which means we sample on the singularity at r = 0.
    #[test]
    fn test_convolution_integral_odd() {
        let (base_w, base_h) = (129, 129);
        let filter = build_convolution_filter_overscan(base_w, base_h);

        let mut integral_120 = 0.0;
        let mut integral_100 = 0.0;
        let mut integral_070 = 0.0;
        let mut integral_050 = 0.0;

        scan_image(&filter, base_w, base_h, |x, y, p| {
            let r = (x * x + y * y).sqrt();
            if r <= 1.2 {
                integral_120 += p;
            }
            if r <= 1.0 {
                integral_100 += p;
            }
            if r <= 0.7 {
                integral_070 += p;
            }
            if r <= 0.5 {
                integral_050 += p;
            }
        });

        assert!((integral_120 - 1.2).abs() < 0.005);
        assert!((integral_100 - 1.0).abs() < 0.005);
        assert!((integral_070 - 0.7).abs() < 0.005);
        assert!((integral_050 - 0.5).abs() < 0.005);
    }

    // Check that the convolution filter is symmetric.
    #[test]
    fn test_convolution_symmetric() {
        let (base_w, base_h) = (128, 129);
        let filter = build_convolution_filter_overscan(base_w, base_h);

        for y1 in 0..filter.height {
            let y2 = filter.height - 1 - y1;

            for x1 in 0..filter.width {
                let x2 = filter.width - 1 - x1;

                let p11 = filter[(x1, y1)];
                let p12 = filter[(x2, y1)];
                let p21 = filter[(x1, y2)];
                let p22 = filter[(x2, y2)];

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
        let (base_w, base_h) = (128, 129);
        let filter = build_convolution_filter_overscan(base_w, base_h);
        let filter_no_over = build_convolution_filter(base_w, base_h, 0, 0);
        assert_eq!(filter_no_over.width, base_w);
        assert_eq!(filter_no_over.height, base_h);

        let w_adj = (filter.width - base_w) / 2;
        let h_adj = (filter.height - base_h) / 2;

        for y in 0..filter_no_over.height {
            let y2 = y + h_adj;
            for x in 0..filter_no_over.width {
                let x2 = x + w_adj;

                let p1 = filter_no_over[(x, y)];
                let p2 = filter[(x2, y2)];

                assert_eq!(p1, p2);
            }
        }
    }

    // Test that the convolution filter we generate is more-or-less
    // the same as what we get by scanning a point and convoluting it.
    #[test]
    fn test_convolution_comparison() {
        let (width, height) = (65, 65);
        let oversample_factor = 5;

        // Generate a convolution filter, no overscan. Integral up to
        // radius 1.0 is 1.0
        let generated = build_convolution_filter(width, height, 0, 0);

        // Generate an image with a circle at the centre. We will later
        // scale it down to a circle one pixel in diameter, to be close
        // to what the convolution filter is generating.
        let o_width = width * oversample_factor;
        let o_height =  height * oversample_factor;
        let radius = oversample_factor as f64 / 2.0;
        let radius2 = radius * radius;

        let src_img = build_image(o_width, o_height, 0, 0, | x, y | {
            // Convert -1..1 coordinates to pixel coordinates.
            let px = x * o_width as f64 / 2.0;
            let py = y * o_height as f64 / 2.0;

            // Calculate the square of distance from the centre in pixel units.
            let r2 = px * px + py * py;
            // Fill in if within the central circle.
            if r2 <= radius2 { 1.0 } else { 0.0 }
        });

        let circle_area = src_img.data.iter().sum::<f64>();

        // Make sure we don't lose precision in the scan.
        let rays = o_width * 2;
        let angles = o_height * 2;
        let scan = scan(&src_img, angles, rays);
        let reconstructed = reconstruct(&scan, o_width, o_height);

        // The resulting image's integral up to radius 1.0 is circle_area.

        // Downscaling the image reduces the integral by oversample_factor^2.
        let downscaled = reconstructed.downscale(oversample_factor);

        // Then normalise to the same scale as the generated filter by
        // dividing through by circle_area / oversample_factor^2
        let scale_factor = (4.0 * radius2) / circle_area;
        let normalised = downscaled.scale_values(scale_factor);

        // Calculate the total error, integrated over the full image.
        let average_error = generated.average_diff(&normalised);
        let total_error = average_error * (generated.width * generated.height) as f64;

        // Given the integral within unit radius is 1, and the
        // integral over the whole square is a bit more, this is not
        // far off a 1% error. If oversample_factor is 9, the error
        // reduces to around 0.01. Given the amount of half-assed
        // numerical methods that I'm doing here, having an error
        // around 1% seems good enough to me! I think the algorithms
        // do pretty much match.
        assert!(0.012 < total_error && total_error < 0.016);
    }
}
