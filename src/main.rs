use clap::{AppSettings, ArgEnum, Clap};
use rustfft::{num_complex::Complex64, FftDirection, FftPlanner};
use std::path::Path;

mod tomo_image;
mod tomo_scan;

mod convolution_solver;
mod matrix_inversion_solver;

use tomo_image::Image;
use tomo_scan::{Scan, scan};

use convolution_solver::build_convolution_filter;

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
        Algorithm::Convolution => convolution_solver::reconstruct(scan, width, height),
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

        let generated = build_convolution_filter(width, height, 0.0).data;

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

        let dst_img = convolution_solver::reconstruct(&scan, width, height);

        let generated = build_convolution_filter(width, height, 0.0).data;

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
