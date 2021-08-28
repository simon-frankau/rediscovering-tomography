extern crate image;
extern crate nalgebra;

use image::{GrayImage, Pixel};
use nalgebra::base::DVector;
use std::path::Path;

////////////////////////////////////////////////////////////////////////
// Image load/save
//

struct TomoImage {
    width: usize,
    height: usize,
    data: DVector<u8>,
}

fn image_load(path: &Path) -> TomoImage {
    let orig_img = image::open(path).unwrap();
    let grey_img = orig_img.into_luma8();

    let width = grey_img.width() as usize;
    let height = grey_img.height() as usize;

    // Do anything that returns a vector.
    TomoImage {
        width,
        height,
        data: DVector::from_iterator(width * height, grey_img.pixels().map(|p| p.channels()[0])),
    }
}

fn image_save(path: &Path, image: TomoImage) {
    let v = image.data.iter().copied().collect::<Vec<u8>>();
    let img = GrayImage::from_vec(image.width as u32, image.height as u32, v).unwrap();
    img.save(path).unwrap();
}

///////////////////////////////////////////////////////////////////////
// Weight calculation
//

// Calculate the weights along a line with |dy/dx| <= 1, using
// something very much like Bresenham's algorithm. We pass in the
// y-value of the x-intercept, and calculate all the weights for
// unit pixels in (0..w, 0..h), as a vector of (x, y, weight).
fn calculate_weights_inner(y_0: f64, dy_dx: f64, w: usize, h: usize) -> Vec<(usize, usize, f64)> {
    assert!(dy_dx.abs() <= 1.0);

    // Crossing a single pixel, left-to-right, without crossing a vertical
    // pixel boundary, has a basic distance. Boundary-crossing path lengths
    // can be derived from this.
    //
    // This is chosen so that crossing the whole pixel area, either
    // vertically or horizontally, is 1 unit.
    let (wf64, hf64) = (w as f64, h as f64);
    let (y_scale, x_scale) = ((1.0 / (hf64 * hf64)), (1.0 / (wf64 * wf64)));
    let base_weight = (x_scale + dy_dx * dy_dx * y_scale).sqrt();

    let mut weights: Vec<(usize, usize, f64)> = Vec::new();

    // Y value is y_int + y_frac, 0 <= y_frac < 1.
    let y_round = y_0.floor();
    let mut y_int: isize = y_round as isize;
    let mut y_fract = y_0 - y_round;

    for x in 0..w {
        let mut add_pixel = |x: usize, y: isize, w: f64| {
            if 0 <= y && y < h as isize {
                weights.push((x, y as usize, w));
            }
        };

        y_fract += dy_dx;
        if y_fract >= 1.0 {
            y_fract -= 1.0;
            let rhs = y_fract / dy_dx;
            let lhs = 1.0 - rhs;
            add_pixel(x, y_int, lhs * base_weight);
            y_int += 1;
            add_pixel(x, y_int, rhs * base_weight);
        } else if y_fract < 0.0 {
            let rhs = y_fract / dy_dx;
            let lhs = 1.0 - rhs;
            y_fract += 1.0;
            add_pixel(x, y_int, lhs * base_weight);
            y_int -= 1;
            add_pixel(x, y_int, rhs * base_weight);
        } else {
            add_pixel(x, y_int, base_weight);
        }
    }

    weights
}

// Calculate the weights along the infinite line passing through (x1,
// y1) and (x2, y2), with |dy/dx| <= 1. The pixel grid is at (0..w,
// 0..h) ("pixel space").
fn calculate_weights_horiz(
    (x1, y1): (f64, f64),
    (x2, y2): (f64, f64),
    w: usize,
    h: usize,
) -> Vec<(usize, usize, f64)> {
    // Safe as | dy/dx | <= 1.0.
    let dy_dx = (y2 - y1) / (x2 - x1);

    let y_0 = y1 - x1 * dy_dx;

    calculate_weights_inner(y_0, dy_dx, w, h)
}

// Calculate the weights along the infinite line passing through (x1,
// y1) and (x2, y2). The pixel grid is at (0..w, 0..h) ("pixel
// space").
fn calculate_pixel_weights(
    (x1, y1): (f64, f64),
    (x2, y2): (f64, f64),
    w: usize,
    h: usize,
) -> Vec<(usize, usize, f64)> {
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();

    if dy > dx {
        // Would result in | dy/dx | > 1, so flip axes.
        let flipped_results = calculate_weights_horiz((y1, x1), (y2, x2), h, w);
        flipped_results
            .iter()
            .map(|(y, x, w)| (*x, *y, *w))
            .collect()
    } else {
        calculate_weights_horiz((x1, y1), (x2, y2), w, h)
    }
}

// Calculate the weights along the infinite line passing through (x1,
// y1) and (x2, y2). The pixel grid is at (-1.0..1.0, -1.0..1.0)
// ("scan space").
fn calculate_scan_weights(
    (x1, y1): (f64, f64),
    (x2, y2): (f64, f64),
    w: usize,
    h: usize,
) -> Vec<(usize, usize, f64)> {
    // This is a simple matter of coordinate transforms. We move (-1, -1)
    // to the origin, and then scale (0.0..2.0, 0.0..2.0) to (0.0..w,
    // 0.0..h).
    let x1p = (x1 + 1.0) * (w as f64 / 2.0);
    let y1p = (y1 + 1.0) * (h as f64 / 2.0);
    let x2p = (x2 + 1.0) * (w as f64 / 2.0);
    let y2p = (y2 + 1.0) * (h as f64 / 2.0);

    // The resulting weights, based on pixel coordinates, don't need
    // to be transformed back, as that's the format we want.
    calculate_pixel_weights((x1p, y1p), (x2p, y2p), w, h)
}

////////////////////////////////////////////////////////////////////////
// Image transformation
//

// Convert the image into a transformed version, where the returned vector
// is a flattened array of (outer layer) different angles with (inner layer)
// parallel rays.
//
// 'angles' returns how many angles are scanned, over a 180 degree range
// (only 180 degrees is needed due to symmetry)
// 'rays' counts how many parallel rays will be spread over the object.
fn scan(image: &TomoImage, angles: usize, rays: usize) -> Vec<f64> {
    // We could just construct a great big matrix and apply that to
    // image.data, but given the inverse transformation is going to be
    // an inverse of that matrix, that feels a bit cheaty. So we'll
    // do the forwards transformation by hand.
    assert!(angles > 1);
    assert!(rays > 1);
    let mut res = Vec::new();
    let angle_step = std::f64::consts::PI / (angles - 1) as f64;
    // Image is inside an axis-aligned square -1..1, so max radius is sqrt(2).
    let ray_offset = 2_f64.sqrt();
    let ray_step = 2.0 * ray_offset / (rays - 1) as f64;
    for angle in (0..angles).map(|x| x as f64 * angle_step) {
        let axis_start = (angle.cos(), angle.sin());
        let axis_end = (-axis_start.0, -axis_start.1);
        for ray in (0..rays).map(|x| x as f64 * ray_step - ray_offset) {
            // Rays are paralell, so move end points at a normal.
            let parallel_offset = (ray * -axis_start.1, ray * axis_start.0);
            let ray_start = (
                axis_start.0 + parallel_offset.0,
                axis_start.1 + parallel_offset.1,
            );
            let ray_end = (
                axis_end.0 + parallel_offset.0,
                axis_end.1 + parallel_offset.1,
            );

            let weights = calculate_scan_weights(ray_start, ray_end, image.width, image.height);
            let integral = weights
                .iter()
                .map(|(x, y, w)| image.data[y * image.width + x] as f64 * w)
                .sum();
            res.push(integral);
        }
    }

    res
}

// Converts a scan to an image and saves it, perhaps useful for understanding the transform.
fn scan_save(path: &Path, angles: usize, rays: usize, data: &[f64]) {
    let image = TomoImage {
        width: rays,
        height: angles,
        data: DVector::from_iterator(rays * angles, data.iter().map(|x| (x / 2_f64.sqrt()) as u8)),
    };

   image_save(path, image);
}

////////////////////////////////////////////////////////////////////////
// Main entry point
//

fn main() {
    let src_img = image_load(Path::new("images/test.png"));
    print!("Processing... ");
    let scan = scan(&src_img, 100, 200);
    let dst_img = src_img;
    println!("done!");
    image_save(Path::new("results/test.png"), dst_img);
    scan_save(Path::new("results/test_scan.png"), 100, 200, &scan);
}

////////////////////////////////////////////////////////////////////////
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    // We may sometimes *just* touch the corner of a pixel. Skip them.
    fn assert_eq_pixels(expected: &[(usize, usize, f64)], actual: &[(usize, usize, f64)]) {
        let filtered_actual: Vec<_> = actual.into_iter().filter(|(_, _, w)| *w > 1e-14).collect();

        assert_eq!(expected.len(), filtered_actual.len());
        for ((x1, y1, w1), (x2, y2, w2)) in expected.iter().zip(filtered_actual.iter()) {
            assert_eq!((x1, y1), (x2, y2));
            assert!((w1 - w2).abs() < 1.0e-14);
        }
    }

    #[test]
    fn test_calc_weights_inner_horizontal() {
        let expected = (0..10).map(|x| (x, 2, 0.1)).collect::<Vec<_>>();
        let actual = calculate_weights_inner(2.5, 0.0, 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_calc_weights_inner_up() {
        // 'base' = length travelled as we move 1 unit in X direction.
        let base = (0.1f64 * 0.1 + 0.2 * 0.2).sqrt();
        let expected = vec![
            (0, 2, base * 0.5),
            (0, 3, base * 0.5),
            (1, 3, base * 0.5),
            (1, 4, base * 0.5),
            (2, 4, base * 0.5),
        ];
        let actual = calculate_weights_inner(2.5, 1.0, 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_calc_weights_inner_down() {
        // 'base' = length travelled as we move 1 unit in X direction.
        let base = (0.1f64 * 0.1 + 0.2 * 0.2).sqrt();
        let expected = vec![
            (2, 4, base * 0.5),
            (3, 4, base * 0.5),
            (3, 3, base * 0.5),
            (4, 3, base * 0.5),
            (4, 2, base * 0.5),
            (5, 2, base * 0.5),
            (5, 1, base * 0.5),
            (6, 1, base * 0.5),
            (6, 0, base * 0.5),
            (7, 0, base * 0.5),
        ];
        let actual = calculate_weights_inner(7.5, -1.0, 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_calc_weights_inner_out_of_bounds() {
        assert!(calculate_weights_inner(-1.0, -1.0, 20, 20).is_empty());
    }

    #[test]
    fn test_calc_weights_inner_non_diagonal_up() {
        // 'base' = length travelled as we move 1 unit in X direction.
        let base = (0.2f64 * 0.2 + 0.15 * 0.15).sqrt();
        let expected = vec![
            (0, 2, base * 2.0 / 3.0),
            (0, 3, base / 3.0),
            (1, 3, base),
            (2, 4, base),
            (3, 4, base / 3.0),
        ];
        let actual = calculate_weights_inner(2.5, 0.75, 5, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_calc_weights_inner_non_diagonal_down() {
        // 'base' = length travelled as we move 1 unit in X direction.
        let base = (0.2f64 * 0.2 + 0.15 * 0.15).sqrt();
        let expected = vec![
            (0, 2, base * 2.0 / 3.0),
            (0, 1, base / 3.0),
            (1, 1, base),
            (2, 0, base),
            (3, 0, base / 3.0),
        ];
        let actual = calculate_weights_inner(2.5, -0.75, 5, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_scan_weights_h() {
        let base = (1.0f64 * 1.0 + 0.5 * 0.5).sqrt() / 10.0;
        let expected = vec![
            (0, 1, base),
            (1, 1, base),
            (2, 1, base),
            (3, 2, base),
            (4, 2, base),
            (5, 2, base),
            (6, 2, base),
            (7, 3, base),
            (8, 3, base),
            (9, 3, base),
        ];
        let actual = calculate_scan_weights((-2.0, -1.0), (2.0, 1.0), 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_scan_weights_v() {
        let base = (1.0f64 * 1.0 + 0.5 * 0.5).sqrt() / 10.0;
        let expected = vec![
            (2, 0, base),
            (3, 0, base),
            (3, 1, base),
            (4, 1, base),
            (4, 2, base),
            (5, 2, base),
            (5, 3, base),
            (6, 3, base),
            (6, 4, base),
            (7, 4, base),
        ];
        let actual = calculate_scan_weights((-1.0, -2.0), (1.0, 2.0), 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_scan_weights_commutes() {
        let p1 = (0.2, 0.9);
        let p2 = (0.5, 0.1);
        let res1 = calculate_scan_weights(p1, p2, 10, 5);
        let res2 = calculate_scan_weights(p2, p1, 10, 5);
        assert_eq!(res1, res2);
    }

    #[test]
    fn test_scan_blank() {
        let image = TomoImage {
            width: 1,
            height: 1,
            data: DVector::from_element(1, 0),
        };
        let scanned = scan(&image, 5, 7);
        assert_eq!(scanned.len(), 5 * 7);
        assert!(scanned.iter().all(|x| *x == 0.0))
    }

    #[test]
    fn test_scan_full() {
        let image = TomoImage {
            width: 1,
            height: 1,
            data: DVector::from_element(1, 1),
        };
        let expected = vec![
            // Horizontal
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            // Diagonal
            0.0,
            2_f64.sqrt() / 3.0,
            2_f64.sqrt() * 2.0 / 3.0,
            2_f64.sqrt(),
            2_f64.sqrt() * 2.0 / 3.0,
            2_f64.sqrt() / 3.0,
            0.0,
            // Vertical
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            // Diagonal
            0.0,
            2_f64.sqrt() / 3.0,
            2_f64.sqrt() * 2.0 / 3.0,
            2_f64.sqrt(),
            2_f64.sqrt() * 2.0 / 3.0,
            2_f64.sqrt() / 3.0,
            0.0,
            // Horizontal, other way.
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
        ];
        let actual = scan(&image, 5, 7);
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 1e-14);
        }
    }

    #[test]
    fn test_scan_scales() {
        let image1 = TomoImage {
            width: 1,
            height: 1,
            data: DVector::from_element(1, 1),
        };
        let scan1 = scan(&image1, 5, 7);

        let image2 = TomoImage {
            width: 5,
            height: 5,
            data: DVector::from_element(8 * 8, 1),
        };
        let scan2 = scan(&image2, 5, 7);

        assert_eq!(scan1.len(), scan2.len());
        for (s1, s2) in scan1.iter().zip(scan2.iter()) {
            assert!((s1 - s2).abs() < 1e-14);
        }
    }
}
