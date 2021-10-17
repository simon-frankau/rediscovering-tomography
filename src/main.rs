use anyhow::{bail, ensure, Result};
use clap::{AppSettings, ArgEnum, Clap};
use std::path::Path;
use rand::prelude::*;
use rand_pcg::Pcg64;

mod tomo_image;
mod tomo_scan;

mod convolution_solver;
mod deconvolution_solver;
mod matrix_inversion_solver;

use tomo_image::Image;
use tomo_scan::{scan, Scan};

////////////////////////////////////////////////////////////////////////
// Main entry point
//

#[derive(ArgEnum)]
pub enum Algorithm {
    MatrixInversion,
    Convolution,
    Deconvolution,
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
    /// File to write the reconstructed image to.
    #[clap(long)]
    output_image: Option<String>,
    /// Width of reconstructed image
    #[clap(long)]
    width: Option<usize>,
    /// Height of reconstructed image
    #[clap(long)]
    height: Option<usize>,
    /// Where to write the diff between input and reconstructed image to.
    #[clap(long)]
    diff_image: Option<String>,
    #[clap(arg_enum, long, default_value = "matrix-inversion")]
    algorithm: Algorithm,
    #[clap(long)]
    /// How many times bigger the reconstruction image should be than
    /// the original. Should be at least 1.0, and only used in
    /// deconvolution mode.
    recon_multiplier: Option<f64>,
    #[clap(long)]
    /// How much uniform noise to add to the scan, as fraction of maximum scan value.
    noise: Option<f64>,
    #[clap(long)]
    /// Seed for the random noise (for reproducibility)
    seed: Option<u64>,
}

// Generate a scan, and return the image it was generated from, if available.
fn generate_scan(opts: &Opts) -> Result<(Option<Image>, Scan)> {
    if let Some(name) = &opts.input_image {
        ensure!(
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
        Ok((Some(image), scanned))
    } else if let Some(name) = &opts.input_scan {
        ensure!(
            opts.angles.is_none(),
            "--angles cannot be used with --input-scan"
        );
        ensure!(
            opts.rays.is_none(),
            "--rays cannot be used with --input-scan"
        );
        ensure!(
            opts.diff_image.is_none(),
            "--diff-image cannot be used with --input-scan"
        );

        Ok((None, Scan::load(Path::new(&name))))
    } else {
        bail!("One of --input-image and --input-scan must be specified");
    }
}

fn add_noise(scan: &Scan, opts: &Opts) -> Scan {
    const DEFAULT_SEED: u64 = 42;
    const DEFAULT_NOISE: f64 = 0.0;

    let seed = opts.seed.unwrap_or(DEFAULT_SEED);
    let noise = opts.noise.unwrap_or(DEFAULT_NOISE);

    let mut rng = Pcg64::seed_from_u64(seed);
    scan.add_noise(&mut rng, noise)
}

fn generate_reconstruction(opts: &Opts, original: &Option<Image>, scan: &Scan) -> Result<Image> {
    const DEFAULT_RECON_MULTIPLIER: f64 = 2.0;

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
    if let Algorithm::Deconvolution = opts.algorithm {
        ensure!(
            opts.recon_multiplier.unwrap_or(1.0) >= 1.0,
            "--recon-multiplier must be greater than or equal to 1.0"
        )
    } else {
        ensure!(
            opts.recon_multiplier.is_none(),
            "--recon-multiplier can only be used with --algorithm=deconvolution"
        );
    }

    Ok(match opts.algorithm {
        Algorithm::MatrixInversion => matrix_inversion_solver::reconstruct(scan, width, height),
        Algorithm::Convolution => convolution_solver::reconstruct(scan, width, height),
        Algorithm::Deconvolution => deconvolution_solver::reconstruct(
            scan,
            width,
            height,
            opts.recon_multiplier.unwrap_or(DEFAULT_RECON_MULTIPLIER),
        ),
    })
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

    let rms_error = base_image.rms_diff(new_image);
    println!("RMS of per-pixel error: {}", rms_error);
}

fn main() -> Result<()> {
    let opts: Opts = Opts::parse();

    let (input_image, mut scan) = generate_scan(&opts)?;

    if opts.noise.is_some() {
        scan = add_noise(&scan, &opts);
    } else {
        ensure!(
            opts.seed.is_none(),
            "--seed can only be used with --noise"
        );
    }

    eprint!("Processing... ");
    let reconstruction = generate_reconstruction(&opts, &input_image, &scan)?;
    eprintln!("done!");

    if let Some(ref image) = input_image {
        calculate_error(image, &reconstruction);
    } else {
        eprintln!("No --input-image supplied, not calculating transformation error vs. base image");
    }

    if let Some(name) = opts.output_scan {
        scan.save(Path::new(&name));
    }

    if let Some(name) = opts.output_image {
        reconstruction.save(Path::new(&name));
    }

    if let Some(name) = opts.diff_image {
        // Technically, the potential range of diff-then-offset_values
        // is -127..383, but almost all diffs will be in 0..255, and
        // "save" will apply a u8 cap/floor anyway, so we don't bother
        // to scale.
        input_image
            .unwrap()
            .diff(&reconstruction)
            .offset_values(128.0)
            .save(Path::new(&name));
    }

    Ok(())
}
