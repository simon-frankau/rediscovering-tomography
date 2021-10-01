//
// Image reconstruction via matrix inversion
//
// The idea here is that scanning an object is really just a linear
// transform, so if we take the inverse of that, we will be able to
// reconstruct the image from the scan. This is a very computationally
// expensive approach - O(n^3) with the basic algorithm, where n is the
// number of pixels - so roughly O(n^6) in image edge length!
//

use nalgebra::base::{DMatrix, DVector};
use std::path::Path;

use crate::tomo_image::Image;
use crate::tomo_scan::{Scan, calculate_scan_weights, scan};


// Generate the matrix that goes from the image to the scan of the
// image. This matrix is what we'll invert.
fn generate_forwards_matrix(
    width: usize,
    height: usize,
    angles: usize,
    rays: usize,
) -> DMatrix<f64> {
    let src_dim = width * height;
    let dst_dim = angles * rays;

    let mut res = DMatrix::from_element(dst_dim, src_dim, 0.0f64);

    // Construction of rays is copy-and-pasted from "scan"...
    assert!(angles > 0);
    assert!(rays > 1);
    // Avoid 180 degree case, as it's same as 0.
    let angle_step = std::f64::consts::PI / angles as f64;
    // Image is inside an axis-aligned square -1..1, so max radius is sqrt(2).
    let ray_offset = 2_f64.sqrt();
    let ray_step = 2.0 * ray_offset / (rays - 1) as f64;
    for angle_num in 0..angles {
        let angle = angle_num as f64 * angle_step;

        let axis_start = (angle.cos(), angle.sin());
        let axis_end = (-axis_start.0, -axis_start.1);
        for ray_num in 0..rays {
            let ray = ray_num as f64 * ray_step - ray_offset;

            // Rays are parallel, so move end points at a normal.
            let parallel_offset = (ray * -axis_start.1, ray * axis_start.0);
            let ray_start = (
                axis_start.0 + parallel_offset.0,
                axis_start.1 + parallel_offset.1,
            );
            let ray_end = (
                axis_end.0 + parallel_offset.0,
                axis_end.1 + parallel_offset.1,
            );

            let dst_row = angle_num * rays + ray_num;
            for (x, y, wt) in calculate_scan_weights(ray_start, ray_end, width, height) {
                let src_col = y * width + x;
                res[(dst_row, src_col)] += wt;
            }
        }
    }

    res
}

// We actually generate the pseudo-inverse, so that the matrix doesn't
// need to be precisely square - we don't require that width * height =
// rays * angles. nalgebra does all the heavy lifting.
fn generate_inverse_matrix(
    width: usize,
    height: usize,
    angles: usize,
    rays: usize,
) -> DMatrix<f64> {
    let forwards = generate_forwards_matrix(width, height, angles, rays);
    forwards.pseudo_inverse(1e-6).unwrap()
}

// And then wrap it up with an easy-to-use function!
pub fn reconstruct(scan: &Scan, width: usize, height: usize) -> Image {
    let matrix = generate_inverse_matrix(width, height, scan.angles, scan.rays);
    let input: DVector<f64> =
        DVector::from_iterator(scan.angles * scan.rays, scan.data.iter().copied());
    let reconstruction = matrix * input;
    Image {
        width,
        height,
        data: reconstruction.iter().copied().collect::<Vec<_>>(),
    }
}

// TODO: See what effect adding noise and changing rays/angles has on
// accuracy.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forwards_matrix_construction() {
        let rays = 75;
        let angles = 50;

        let src_img = Image::load(Path::new("images/test.png"));
        let src_data = DVector::from_iterator(src_img.width * src_img.height, src_img.data.iter().copied());

        let matrix = generate_forwards_matrix(src_img.width, src_img.height, angles, rays);

        let matrixed = (matrix * src_data).iter().copied().collect::<Vec<_>>();
        let scanned = scan(&src_img, angles, rays).data;

        assert_eq!(scanned.len(), matrixed.len());
        for (s1, s2) in scanned.iter().zip(matrixed.iter()) {
            // Slightly larger epsilon for this one, since more operations are applied.
            assert!((s1 - s2).abs() < 1e-10);
        }
    }

    // This test takes 20s on my old Macbook, so let's not run by default.
    #[ignore]
    #[test]
    fn test_end_to_end_expensive() {
        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let rays = 35;
        let angles = 40;

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);
        let dst_img = reconstruct(&scan, src_img.width, src_img.height);

        let total_error: f64 = src_img
            .data
            .iter()
            .zip(dst_img.data.iter())
            .map(|(&p1, &p2)| (p1 as f64 - p2 as f64).abs())
            .sum();

        let average_error = total_error / (dst_img.width * dst_img.height) as f64;

        // Less than 1/2560 average error!
        assert!(average_error < 0.1);
    }

    #[test]
    fn test_end_to_end_cheap() {
        let rays = 10;
        let angles = 10;

        let width = 8;
        let height = 8;
        let data = (0..8 * 8).map(|p| {
            let (x, y) = (p % 8, p / 8);
            x as f64 * 8.0 + if y >= 4 { 64.0 } else { 0.0 }
        }).collect::<Vec<_>>();

        let src_img = Image {
            width,
            height,
            data,
        };
        let scan = scan(&src_img, angles, rays);
        let dst_img = reconstruct(&scan, src_img.width, src_img.height);

        let total_error: f64 = src_img
            .data
            .iter()
            .zip(dst_img.data.iter())
            .map(|(&p1, &p2)| (p1 as f64 - p2 as f64).abs())
            .sum();

        let average_error = total_error / (dst_img.width * dst_img.height) as f64;

        assert!(average_error < 0.5);
    }
}