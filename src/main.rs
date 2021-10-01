use clap::{AppSettings, ArgEnum, Clap};
use std::path::Path;

mod tomo_image;
mod tomo_scan;

mod convolution_solver;
mod deconvolution_solver;
mod matrix_inversion_solver;

use tomo_image::Image;
use tomo_scan::{Scan, scan};

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
