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
    let base_weight = (1.0 + dy_dx * dy_dx).sqrt();

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

// TODO: Calculate weights in pixel space, from start point and direction.
// knowing |dy/dx| < 1

// TODO: Calculate weights in pixel space from start point and direction.

// TODO: Calculate weights in scan space, from start point and direction.

////////////////////////////////////////////////////////////////////////
// Main entry point
//

fn main() {
    let src_img = image_load(Path::new("images/test.png"));
    print!("Processing... ");
    let dst_img = src_img;
    println!("done!");
    image_save(Path::new("results/test.png"), dst_img);
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

    // fn calculate_weights_inner(y_0: f64, dy_dx: f64, w: usize, h:usize) -> Vec<(usize, usize, f64)> {
    #[test]
    fn test_calc_weights_inner_horizontal() {
        let expected = (0..10).map(|x| (x, 2, 1.0)).collect::<Vec<_>>();
        let actual = calculate_weights_inner(2.5, 0.0, 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_calc_weights_inner_up() {
        // 'base' = length travelled as we move 1 unit in X direction.
        let base = 0.5_f64.sqrt();
        let expected = vec![
            (0, 2, base),
            (0, 3, base),
            (1, 3, base),
            (1, 4, base),
            (2, 4, base),
        ];
        let actual = calculate_weights_inner(2.5, 1.0, 10, 5);
        assert_eq_pixels(&expected, &actual);
    }

    #[test]
    fn test_calc_weights_inner_down() {
        // 'base' = length travelled as we move 1 unit in X direction.
        let base = 0.5_f64.sqrt();
        let expected = vec![
            (2, 4, base),
            (3, 4, base),
            (3, 3, base),
            (4, 3, base),
            (4, 2, base),
            (5, 2, base),
            (5, 1, base),
            (6, 1, base),
            (6, 0, base),
            (7, 0, base),
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
        let base = 1.25;
        let expected = vec![
            (0, 2, base * 2.0 / 3.0),
            (0, 3, base / 3.0),
            (1, 3, base),
            (2, 4, base),
            (3, 4, base / 3.0),
        ];
        let actual = calculate_weights_inner(2.5, 0.75, 10, 5);
        assert_eq_pixels(&expected, &actual);
    }
}
