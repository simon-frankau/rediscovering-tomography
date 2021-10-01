use clap::{AppSettings, ArgEnum, Clap};
use rustfft::{num_complex::Complex64, FftDirection, FftPlanner};
use std::path::Path;

mod tomo_image;
mod tomo_scan;
mod matrix_inversion_solver;

use tomo_image::Image;
use tomo_scan::{Scan, scan};

////////////////////////////////////////////////////////////////////////
// Reconstruction via deconvolution
//

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
fn generate_convolved_tomo(scan: &Scan, w: usize, h: usize) -> Vec<f64> {
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
fn build_convolution_filter(
    width: usize,
    height: usize,
    overscan: f64,
) -> (usize, usize, Vec<f64>) {
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

    // TODO: Maybe make a proper struct?
    (w_full, h_full, res)
}

// Not a real attempt at reconstruction, we just do the
// convolution-generation step and return it without attempting to
// deconvolve. It does give a nice blurry version of the original!
fn reconstruct_convolution(scan: &Scan, width: usize, height: usize) -> Image {
    let data = generate_convolved_tomo(scan, width, height);

    Image {
        width,
        height,
        data: data,
    }
}

// We're not going to do any fancy real-valued optimisations for the
// FFT, we simply convert to and from complex numbers.
fn to_complex(v: &[f64]) -> Vec<Complex64> {
    v.iter().map(|x| Complex64::new(*x, 0.0)).collect()
}

// Doing a clever real-to-complex conversion for the FFTs would allow
// us to define away the error case of the complex component being
// non-trivial, but we're keeping things simple.
fn from_complex(v: &[Complex64]) -> Vec<f64> {
    const EPSILON: f64 = 1e-7;
    assert!(
        v.iter()
            .map(|z| z.im.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            < EPSILON
    );
    v.iter().map(|x| x.re).collect()
}

fn transpose<T: Copy>(width: usize, height: usize, data: &[T]) -> Vec<T> {
    assert_eq!(width * height, data.len());

    let mut res = Vec::with_capacity(width * height);
    // Bounds in reverse to what you might expect, since we're transposing.
    for y in 0..width {
        for x in 0..height {
            res.push(data[x * width + y]);
        }
    }
    res
}

// Perform a 2D FFT
fn fourier_transform(
    width: usize,
    height: usize,
    data: &[Complex64],
    dir: FftDirection,
) -> Vec<Complex64> {
    assert_eq!(width * height, data.len());

    let zero = Complex64::new(0.0, 0.0);

    let mut matrix = data.iter().copied().collect::<Vec<_>>();

    // FFT pass in width direction
    {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(width, dir);
        let mut scratch = vec![zero; width];
        fft.process_with_scratch(&mut matrix, &mut scratch);
    }

    matrix = transpose(width, height, &matrix);

    // FFT pass in height direction
    {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(height, dir);
        let mut scratch = vec![zero; height];
        fft.process_with_scratch(&mut matrix, &mut scratch);
    }

    transpose(height, width, &matrix)
}

// The way convolution works means the deconvolved image is shifted by
// an amount equivalent to the coordinates of the centre of the filter.
// "shift" can undo this.

fn shift<T: Copy>(
    width: usize,
    height: usize,
    data: &[T],
    x_shift: usize,
    y_shift: usize,
) -> Vec<T> {
    let mut res = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let src_y = (y + y_shift) % height;
            let src_x = (x + x_shift) % width;
            res.push(data[src_y * width + src_x]);
        }
    }
    res
}

// TODO: Implement some tests
// * Implement convolution via FFT, check it works the same.
// * Improve accuracy of deconvolution.

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
fn generate_scan(opts: &Opts) -> (Option<Image>, Scan) {
    // TODO: More graceful error handling than assert!/panic!.

    if let Some(name) = &opts.input_image {
        assert!(
            opts.input_scan.is_none(),
            "Please specify only one of --input-image and --input-scan"
        );

        let image = Image::load(Path::new(&name));
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
        assert!(
            opts.angles.is_none(),
            "--angles cannot be used with --input-scan"
        );
        assert!(
            opts.rays.is_none(),
            "--rays cannot be used with --input-scan"
        );

        (None, Scan::load(Path::new(&name)))
    } else {
        panic!("One of --input-image and --input-scan must be specified");
    }
}

fn generate_reconstruction(
    opts: &Opts,
    original: &Option<Image>,
    scan: &Scan,
) -> Image {
    // When choosing image size, prefer the command-line flag,
    // otherwise infer from original size, otherwise guess based on
    // scan size.
    let resolution = scan.angles.max(scan.rays);
    let width = opts
        .width
        .or_else(|| original.as_ref().map(|x| x.width))
        .unwrap_or_else(|| {
            eprintln!("No --width or --input-image, using width of {}", resolution);
            resolution
        });
    let height = opts
        .height
        .or_else(|| original.as_ref().map(|x| x.height))
        .unwrap_or_else(|| {
            eprintln!(
                "No --height or --input-image, using height of {}",
                resolution
            );
            resolution
        });

    match opts.algorithm {
        Algorithm::MatrixInversion => matrix_inversion_solver::reconstruct(scan, width, height),
        Algorithm::Convolution => reconstruct_convolution(scan, width, height),
    }
}

fn calculate_error(base_image: &Image, new_image: &Image) {
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

    let total_error: f64 = base_image
        .data
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
        scan.save(Path::new(&name));
    }

    if let Some(name) = opts.output_image {
        reconstruction.save(Path::new(&name));
    }
}

////////////////////////////////////////////////////////////////////////
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    // Check that the integral over the convolution filter is what we expect
    // (integral = r).
    #[test]
    fn test_convolution_integral_even() {
        let (base_width, base_height) = (128, 128);
        let overscan = 0.3;
        let (width, height, filter) = build_convolution_filter(base_width, base_height, overscan);
        assert!(width % 2 == 0);
        assert!(height % 2 == 0);

        // Yet more C&P of the pixel iterator code...
        let x_step = 2.0 / base_width as f64;
        let y_step = 2.0 / base_height as f64;

        // Sample the centre of pixels - offset by 0.5.
        let x_offset = (width as f64 - 1.0) / 2.0;
        let y_offset = (height as f64 - 1.0) / 2.0;

        let mut integral_120 = 0.0;
        let mut integral_100 = 0.0;
        let mut integral_070 = 0.0;
        let mut integral_050 = 0.0;

        for y_idx in 0..height {
            let y = (y_idx as f64 - y_offset) * y_step;
            for x_idx in 0..width {
                let x = (x_idx as f64 - x_offset) * x_step;

                let r = (x * x + y * y).sqrt();
                let weight = filter[y_idx * width + x_idx];
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
        let (width, height, filter) = build_convolution_filter(base_width, base_height, overscan);
        assert!(width % 2 == 1);
        assert!(height % 2 == 1);

        // Yet more C&P of the pixel iterator code...
        let x_step = 2.0 / base_width as f64;
        let y_step = 2.0 / base_height as f64;

        // Sample the centre of pixels - offset by 0.5.
        let x_offset = (width as f64 - 1.0) / 2.0;
        let y_offset = (height as f64 - 1.0) / 2.0;

        let mut integral_120 = 0.0;
        let mut integral_100 = 0.0;
        let mut integral_070 = 0.0;
        let mut integral_050 = 0.0;

        for y_idx in 0..height {
            let y = (y_idx as f64 - y_offset) * y_step;
            for x_idx in 0..width {
                let x = (x_idx as f64 - x_offset) * x_step;

                let r = (x * x + y * y).sqrt();
                let weight = filter[y_idx * width + x_idx];
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
        let (width, height, filter) = build_convolution_filter(base_width, base_height, overscan);
        assert!(width % 2 == 0);
        assert!(height % 2 == 1);

        for y1 in 0..height {
            let y2 = height - 1 - y1;

            for x1 in 0..width {
                let x2 = width - 1 - x1;

                let p11 = filter[y1 * width + x1];
                let p12 = filter[y1 * width + x2];
                let p21 = filter[y2 * width + x1];
                let p22 = filter[y2 * width + x2];

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
        let (width, height, filter) = build_convolution_filter(base_width, base_height, overscan);
        let (width_no_over, height_no_over, filter_no_over) =
            build_convolution_filter(base_width, base_height, 0.0);
        assert_eq!(width_no_over, base_width);
        assert_eq!(height_no_over, base_height);

        let w_adj = (width - base_width) / 2;
        let h_adj = (height - base_height) / 2;

        for y in 0..base_height {
            let y2 = y + h_adj;
            for x in 0..width_no_over {
                let x2 = x + w_adj;

                let p1 = filter_no_over[y * base_width + x];
                let p2 = filter[y2 * width + x2];

                assert_eq!(p1, p2);
            }
        }
    }

    // Scales down the image by the given factor, which must divide
    // width and height.
    //
    // TODO: Currently only used for testing, but may come in handy...
    fn downscale(image: &[f64], width: usize, height: usize, factor: usize) -> Vec<f64> {
        assert_eq!(image.len(), width * height);
        assert!(width % factor == 0);
        assert!(height % factor == 0);

        let new_w = width / factor;
        let new_h = height / factor;

        let mut res = Vec::new();
        for y in (0..new_h).map(|y| y * factor) {
            for x in (0..new_w).map(|x| x * factor) {
                let mut pixel: f64 = 0.0;
                for sub_y in 0..factor {
                    for sub_x in 0..factor {
                        pixel += image[(y + sub_y) * width + (x + sub_x)];
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
        let (_, _, generated) = build_convolution_filter(width, height, 0.0);

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

        // Make sure we don't lose precision in the scan.
        let rays = o_width * 2;
        let angles = o_height * 2;

        let src_img = Image {
            width: o_width,
            height: o_height,
            data: image,
        };
        let scan = scan(&src_img, angles, rays);
        let reconstructed = generate_convolved_tomo(&scan, o_width, o_height);

        // The resulting image's integral up to radius 1.0 is circle_area.

        // Downscaling the image reduces the integral by oversample_factor^2.
        let downscaled = downscale(&reconstructed, o_width, o_height, oversample_factor);

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

    #[test]
    fn test_transpose() {
        let width = 5;
        let height = 7;
        let v = (0..(height * width)).collect::<Vec<_>>();

        let vt = transpose(width, height, &v);

        for y in 0..height {
            for x in 0..width {
                assert_eq!(v[y * width + x], vt[x * height + y]);
            }
        }
    }

    #[test]
    fn test_double_fft() {
        // Give it some awkward sizes. :)
        let width = 13;
        let height = 17;

        let input = (0..(height * width)).map(|x| x as f64).collect::<Vec<_>>();

        let input_complex = to_complex(&input);

        let fft = fourier_transform(width, height, &input_complex, FftDirection::Forward);

        let output_complex = fourier_transform(width, height, &fft, FftDirection::Inverse);

        let output = from_complex(&output_complex);

        assert_eq!(input.len(), output.len());
        for (in_val, out_val) in input.iter().zip(output.iter()) {
            assert!((in_val - out_val / (width * height) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deconvolve() {
        let (width, height) = (65, 65);

        let (_, _, generated) = build_convolution_filter(width, height, 0.0);

        let deconvolution_fft = fourier_transform(
            width,
            height,
            &to_complex(&generated),
            FftDirection::Forward,
        )
        .iter()
        .map(Complex64::inv)
        .collect::<Vec<_>>();

        let fftd = fourier_transform(
            width,
            height,
            &to_complex(&generated),
            FftDirection::Forward,
        );

        let decon = fftd
            .iter()
            .zip(deconvolution_fft.iter())
            .map(|(z1, z2)| z1 * z2)
            .collect::<Vec<_>>();

        let res = fourier_transform(width, height, &decon, FftDirection::Inverse);

        let res2 = from_complex(&res)
            .iter()
            .map(|x| x / (width * height) as f64)
            .collect::<Vec<_>>();

        for (i, actual) in res2.iter().enumerate() {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!((actual - expected).abs() <= 1e-10);
        }
    }

    // TODO: This function is messy, but we'll submit first and tidy later.
    #[test]
    fn test_end_to_end_deconvolve() {
        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let rays = 35;
        let angles = 40;

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let width = src_img.width;
        let height = src_img.height;

        let dst_img = reconstruct_convolution(&scan, width, height);

        let (_, _, generated) = build_convolution_filter(width, height, 0.0);

        let deconvolution_fft = fourier_transform(
            width,
            height,
            &to_complex(&generated),
            FftDirection::Forward,
        )
        .iter()
        .map(|z| {
            if z.norm_sqr() < 1e-4 {
                Complex64::new(0.0, 0.0)
            } else {
                z.inv()
            }
        })
        .collect::<Vec<_>>();

        // TODO: Maybe use something that doesn't convert to matrix...
        let img = dst_img.data.iter().map(|x| *x as f64).collect::<Vec<f64>>();

        let fftd = fourier_transform(width, height, &to_complex(&img), FftDirection::Forward);

        let decon = fftd
            .iter()
            .zip(deconvolution_fft.iter())
            .map(|(z1, z2)| z1 * z2)
            .collect::<Vec<_>>();

        let res = fourier_transform(width, height, &decon, FftDirection::Inverse);

        let res2 = from_complex(&res)
            .iter()
            .map(|x| x / (width * height) as f64)
            .collect::<Vec<_>>();

        let res3 = shift(width, height, &res2, width / 2, height / 2);

        let img = Image {
            width,
            height,
            data: res3,
        };

        /* TODO: For debugging...
                img.save(Path::new("full_cycle.png"));

                let diff: res3
                    .iter().zip(src_img.data.iter())
                    .map(|(x, y)| {
                         let diff = *x as i32 - *y as i32;
                         (diff + 128) as f64 })
                    .collect::<Vec<_>>();

                let diff_img = Image { width, height, data: diff };
                diff_img.save(Path::new("full_cycle_diff.png"));
        */

        let total_error: f64 = src_img
            .data
            .iter()
            .zip(img.data.iter())
            .map(|(&p1, &p2)| (p1 as f64 - p2 as f64).abs())
            .sum();

        let average_error = total_error / (dst_img.width * dst_img.height) as f64;

        // TODO: This error is pretty huge, but small enough to mean
        // the image is roughly right.
        assert!(average_error < 30.0);
    }
}
