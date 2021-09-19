use clap::{AppSettings, ArgEnum, Clap};
use image::{GrayImage, Pixel};
use nalgebra::base::{DMatrix, DVector};
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
            .map(|(y, x, wt)| (*x, *y, *wt))
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

struct TomoScan {
    angles: usize,
    rays: usize,
    data: Vec<f64>,
}

// Convert the image into a transformed version, where the returned vector
// is a flattened array of (outer layer) different angles with (inner layer)
// parallel rays.
//
// 'angles' returns how many angles are scanned, over a 180 degree range
// (only 180 degrees is needed due to symmetry)
// 'rays' counts how many parallel rays will be spread over the object.
fn scan(image: &TomoImage, angles: usize, rays: usize) -> TomoScan {
    // We could just construct a great big matrix and apply that to
    // image.data, but given the inverse transformation is going to be
    // an inverse of that matrix, that feels a bit cheaty. So we'll
    // do the forwards transformation by hand.
    assert!(angles > 0);
    assert!(rays > 1);
    let mut res = Vec::new();
    let angle_step = std::f64::consts::PI / angles as f64;
    // Image is inside an axis-aligned square -1..1, so max radius is sqrt(2).
    let ray_offset = 2_f64.sqrt();
    let ray_step = 2.0 * ray_offset / (rays - 1) as f64;
    for angle in (0..angles).map(|x| x as f64 * angle_step) {
        let axis_start = (angle.cos(), angle.sin());
        let axis_end = (-axis_start.0, -axis_start.1);
        for ray in (0..rays).map(|x| x as f64 * ray_step - ray_offset) {
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

            let weights = calculate_scan_weights(ray_start, ray_end, image.width, image.height);
            let integral = weights
                .iter()
                .map(|(x, y, wt)| image.data[y * image.width + x] as f64 * wt)
                .sum();
            res.push(integral);
        }
    }

    TomoScan { angles, rays, data: res }
}

// Converts a scan to an image and saves it, perhaps useful for
// understanding the transform.
fn scan_save(path: &Path, scan: &TomoScan) {
    let image = TomoImage {
        width: scan.rays,
        height: scan.angles,
        data: DVector::from_iterator(
            scan.rays * scan.angles,
            scan.data.iter().map(|x| (x / 2_f64.sqrt()) as u8)),
    };

   image_save(path, image);
}

// If we're supporting saving, let's support loading.
fn scan_load(path: &Path) -> TomoScan {
    let image = image_load(path);
    TomoScan {
        rays: image.width,
        angles: image.height,
        data: image.data
            .iter()
            .map(|x| *x as f64 * 2_f64.sqrt())
            .collect(),
    }
}

////////////////////////////////////////////////////////////////////////
// Image reconstruction via matrix inversion
//

fn generate_forwards_matrix(width: usize, height: usize, angles:usize, rays: usize) -> DMatrix<f64> {
    let src_dim = width * height;
    let dst_dim = angles * rays;

    let mut res = DMatrix::from_element(dst_dim, src_dim, 0.0f64);

    // Construction of rays of copy-and-paste from "scan"...
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

fn generate_inverse_matrix(width: usize, height: usize, angles:usize, rays: usize) -> DMatrix<f64> {
    let forwards = generate_forwards_matrix(width, height, angles, rays);
    forwards.pseudo_inverse(1e-6).unwrap()
}

fn reconstruct_matrix_invert(scan: &TomoScan, width: usize, height: usize) -> TomoImage {
    let matrix = generate_inverse_matrix(width, height, scan.angles, scan.rays);
    let input: DVector<f64> = DVector::from_iterator(scan.angles * scan.rays, scan.data.iter().copied());
    let reconstruction = matrix * input;
    let recon_as_u8: DVector<u8> = DVector::from_iterator(width * height, reconstruction
        .iter()
        .map(|x| x.max(0.0).min(255.0) as u8));

    TomoImage { width, height, data: recon_as_u8 }
}

// TODO: See what effect adding noise and changing rays/angles has on
// accuracy.

////////////////////////////////////////////////////////////////////////
// Reconstruction via deconvolution
//

// Integrate across all the rays that pass through a given point.
// Given point is in (-1..1, -1..1) coordinates.
fn circular_integral(scan: &TomoScan, x: f64, y: f64) -> f64 {
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
// TomoScan. The result is a convolved image.
fn generate_convolved_tomo(scan: &TomoScan, w: usize, h: usize) -> Vec<f64> {
    let mut res = Vec::new();

    let y_scale = 2.0 / h as f64;
    let y_offset = -1.0;
    // Sample at the centre of the pixels - add 0.5
    for y in (0..h).map(|idx| (idx as f64 + 0.5) * y_scale + y_offset) {
        let x_scale = 2.0 / w as f64;
        let x_offset = -1.0;
        for x in (0..w).map(|idx| (idx as f64 + 0.5) * x_scale + x_offset) {
            res.push(circular_integral(scan, x, y));
        }
    }

    res
}

// Not a real attempt at reconstruction, we just do the
// convolution-generation step and return it without attempting to
// deconvolve. It does give a nice blurry version of the original!

fn reconstruct_convolution(scan: &TomoScan, width: usize, height: usize) -> TomoImage {
    let recon = generate_convolved_tomo(&scan, width, height);

    let recon_as_u8: DVector<u8> = DVector::from_iterator(width * height, recon
        .iter()
        .map(|x| x.max(0.0).min(255.0) as u8));

    TomoImage { width, height, data: recon_as_u8 }
}

// TODO: Implement some tests
// * Generate the convolution filter.
// * Do the convolution the long way around, from the original image, and check that they match.
// * Implement convolution via FFT, check it works the same.
// * Generate the deconvolution filter, test end-to-end.

////////////////////////////////////////////////////////////////////////
// Main entry point
//

#[derive(ArgEnum)]
pub enum Algorithm {
    MatrixInversion,
    Convolution,
}

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "0.1", author = "Simon Frankau <sgf@arbitrary.name>")]
#[clap(about = "Simple test of tomography algorithms")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    /// Input image file, which will be scanned.
    #[clap(long)]
    input_image: Option<String>,
    /// Number of angles to scan from.
    #[clap(long)]
    angles: Option<usize>,
    /// Number of parallel rays fired from each angle.
    #[clap(long)]
    rays: Option<usize>,
    /// Alternatively, read a scan file directly. Incompatible with --input-image.
    #[clap(long)]
    input_scan: Option<String>,
    /// File to write the intermediate scan to.
    #[clap(long)]
    output_scan: Option<String>,
    /// File to write the reconstructed image to
    #[clap(long)]
    output_image: Option<String>,
    /// Width of reconstructed image
    #[clap(long)]
    width: Option<usize>,
    /// Height of reconstructed image
    #[clap(long)]
    height: Option<usize>,
    #[clap(arg_enum, long, default_value = "matrix-inversion")]
    algorithm: Algorithm,
}

// Generate a scan, and return the image it was generated from, if available.
fn generate_scan(opts: &Opts) -> (Option<TomoImage>, TomoScan) {
    // TODO: More graceful error handling than assert!/panic!.

    if let Some(name) = &opts.input_image {
        assert!(opts.input_scan.is_none(), "Please specify only one of --input-image and --input-scan");

        let image = image_load(Path::new(&name));
        let resolution = image.width.max(image.height);
        let angles = opts.angles.unwrap_or_else(|| {
            eprintln!("--angles not specified, using {}.", resolution);
            resolution
        });
        let rays = opts.rays.unwrap_or_else(|| {
            eprintln!("--rays not specified, using {}.", resolution);
            resolution
        });
        let scanned = scan(&image, angles, rays);
        (Some(image), scanned)
    } else if let Some(name) = &opts.input_scan {
        assert!(opts.angles.is_none(), "--angles cannot be used with --input-scan");
        assert!(opts.rays.is_none(), "--rays cannot be used with --input-scan");

        (None, scan_load(Path::new(&name)))
    } else {
        panic!("One of --input-image and --input-scan must be specified");
    }
}

fn generate_reconstruction(opts: &Opts, original: &Option<TomoImage>, scan: &TomoScan) -> TomoImage {
    // When choosing image size, prefer the command-line flag,
    // otherwise infer from original size, otherwise guess based on
    // scan size.
    let resolution = scan.angles.max(scan.rays);
    let width = opts.width.or(original.as_ref().map(|x| x.width)).unwrap_or_else(|| {
        eprintln!("No --width or --input-image, using width of {}", resolution);
        resolution
    });
    let height = opts.height.or(original.as_ref().map(|x| x.height)).unwrap_or_else(|| {
        eprintln!("No --height or --input-image, using height of {}", resolution);
        resolution
    });

    match opts.algorithm {
        Algorithm::MatrixInversion => reconstruct_matrix_invert(&scan, width, height),
        Algorithm::Convolution => reconstruct_convolution(&scan, width, height),
    }
}

fn calculate_error(base_image: &TomoImage, new_image: &TomoImage) {
    if base_image.width != new_image.width {
        eprintln!("Base image width does not match reconstructed image width ({} vs. {}). Not calculating error.",
            base_image.width, new_image.width);
        return;
    }

    if base_image.height != new_image.height {
        eprintln!("Base image height does not match reconstructed image height ({} vs. {}). Not calculating error.",
            base_image.height, new_image.height);
        return;
    }

    let total_error: f64 = base_image.data
        .iter()
        .zip(new_image.data.iter())
        .map(|(&p1, &p2)| (p1 as f64 - p2 as f64).abs())
        .sum();

    let average_error = total_error / (base_image.width * base_image.height) as f64;

    println!("Average per-pixel error: {}", average_error);
}

fn main() {
    let opts: Opts = Opts::parse();

    let (input_image, scan) = generate_scan(&opts);

    eprint!("Processing... ");
    let reconstruction = generate_reconstruction(&opts, &input_image, &scan);
    eprintln!("done!");

    if let Some(image) = input_image {
        calculate_error(&image, &reconstruction);
    } else {
        eprintln!("No --input-image supplied, not calculating transformation error vs. base image");
    }

    if let Some(name) = opts.output_scan {
        scan_save(Path::new(&name), &scan);
    }

    if let Some(name) = opts.output_image {
        image_save(Path::new(&name), reconstruction);
    }
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
        let scanned = scan(&image, 4, 7).data;
        assert_eq!(scanned.len(), 4 * 7);
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
        ];
        let actual = scan(&image, 4, 7).data;
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
        let scan1 = scan(&image1, 4, 7).data;

        let image2 = TomoImage {
            width: 8,
            height: 8,
            data: DVector::from_element(8 * 8, 1),
        };
        let scan2 = scan(&image2, 4, 7).data;

        assert_eq!(scan1.len(), scan2.len());
        for (s1, s2) in scan1.iter().zip(scan2.iter()) {
            assert!((s1 - s2).abs() < 1e-14);
        }
    }

    #[test]
    fn test_forwards_matrix_construction() {
        let rays = 75;
        let angles = 50;

        let src_img = image_load(Path::new("images/test.png"));
        let src_data: DMatrix<f64> = nalgebra::convert(src_img.data.clone());

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

        let src_img = image_load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);
        let dst_img = reconstruct_matrix_invert(&scan, src_img.width, src_img.height);

        let total_error: f64 = src_img.data
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
        let data = DVector::from_fn(8 * 8, |p, _| {
            let (x, y) = (p % 8, p / 8);
            (x * 8 + if y >= 4 { 64 } else { 0 }) as u8
            });

        let src_img = TomoImage { width, height, data };
        let scan = scan(&src_img, angles, rays);
        let dst_img = reconstruct_matrix_invert(&scan, src_img.width, src_img.height);

        let total_error: f64 = src_img.data
            .iter()
            .zip(dst_img.data.iter())
            .map(|(&p1, &p2)| (p1 as f64 - p2 as f64).abs())
            .sum();

        let average_error = total_error / (dst_img.width * dst_img.height) as f64;

        assert!(average_error < 0.5);
    }
 }
