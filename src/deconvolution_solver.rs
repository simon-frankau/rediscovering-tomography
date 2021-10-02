//
// Generate a more accurate reconstruction by deconvolution
//
// This file takes the convolved reconstruction, and then
// deconvolves it to obtain a better approximation of the original.
//

use rustfft::{num_complex::Complex64, FftDirection, FftPlanner};
use std::path::Path;

use crate::tomo_image::Image;
use crate::tomo_scan::scan;

use crate::convolution_solver::{build_convolution_filter};

////////////////////////////////////////////////////////////////////////
// Utilities
//

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

////////////////////////////////////////////////////////////////////////
// FFT of an Image
//

#[derive(Clone, Debug)]
struct FFTImage {
    width: usize,
    height: usize,
    data: Vec<Complex64>,
}

impl FFTImage {
    // Transpose, mutating self. Used by fourier_transform.
    fn transpose(&mut self) {
        let mut new_data = Vec::with_capacity(self.width * self.height);
        for y in 0..self.width {
            for x in 0..self.height {
                new_data.push(self.data[x * self.width + y]);
            }
        }
        self.data = new_data;
        std::mem::swap(&mut self.width, &mut self.height);

    }

    // Perform a 1d FFT, in place. Used by fourier_transform.
    fn fft_pass_1d(&mut self, dir: FftDirection) {
        const ZERO: Complex64 = Complex64::new(0.0, 0.0);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(self.width, dir);
        let mut scratch = vec![ZERO; self.width];
        fft.process_with_scratch(&mut self.data, &mut scratch);
    }

    // Core of 2D FFT, used by from_image and to_image.
    fn fourier_transform(self, dir: FftDirection) -> FFTImage {
        let mut tmp = FFTImage {
            width:self.width,
            height: self.height,
            data: self.data,
        };

        tmp.fft_pass_1d(dir);
        tmp.transpose();
        tmp.fft_pass_1d(dir);
        tmp.transpose();

        tmp
    }

    // Construct an FFT from an image.
    fn from_image(img: &Image) -> FFTImage {
        let input = FFTImage {
            width: img.width,
            height: img.height,
            data: to_complex(&img.data)
        };
        input.fourier_transform(FftDirection::Forward)
    }

    // Create an image from the FFT.
    fn to_image(&self) -> Image {
        let output = self.clone().fourier_transform(FftDirection::Inverse);
        Image {
            width: self.width,
            height: self.height,
            data: from_complex(&output.data),
        }
    }

    // Create the inverse of an FFT. Undoes a convolution.
    fn invert(&mut self, threshold: f64) {
        // Square to match the norm.
        let t = threshold * threshold;
        self.data = self
            .data
            .iter()
            .map(|z| {
                // If a coefficient is tiny, this means a frequency
                // component is basically killed. Rather than take
                // the reciprocal and produce a huge number that will
                // produce a noisy inverse, we just live with that
                // frequency being killed off.
                if z.norm_sqr() < t {
                    Complex64::new(0.0, 0.0)
                } else {
                    z.inv()
                }
            })
            .collect::<Vec<_>>();
    }

    // Apply a convolution, which is a pointwise multiplication in the
    // frequency domain.
    fn convolve(&mut self, filter: &FFTImage) {
        assert_eq!(self.width, filter.width);
        assert_eq!(self.height, filter.height);

        self.data = self.data
            .iter()
            .zip(filter.data.iter())
            .map(|(z1, z2)| z1 * z2)
            .collect::<Vec<_>>();
    }
}

// TODO: Implement some tests
// * Implement convolution via FFT, check it works the same.
// * Improve accuracy of deconvolution.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let width = 5;
        let height = 7;
        let data = (0..(height * width))
            .map(|x| Complex64::new(x as f64, 0.0))
            .collect::<Vec<_>>();

        let v = FFTImage {
            width,
            height,
            data,
        };

        let mut vt = v.clone();
        vt.transpose();

        assert_eq!(v.width, vt.height);
        assert_eq!(v.height, vt.width);

        for y in 0..height {
            for x in 0..width {
                assert_eq!(v.data[y * v.width + x], vt.data[x * vt.width + y]);
            }
        }
    }

    #[test]
    fn test_double_fft() {
        // Give it some awkward sizes. :)
        let width = 13;
        let height = 17;
        let data = (0..(height * width))
            .map(|x| x as f64)
            .collect::<Vec<_>>();

        let input = Image {
            width,
            height,
            data,
        };

        let fft = FFTImage::from_image(&input);
        let inv = fft.to_image();
        let output = inv.normalise((inv.width * inv.height) as f64);

        assert_eq!(input.data.len(), output.data.len());
        for (in_val, out_val) in input.data.iter().zip(output.data.iter()) {
            assert!((in_val - out_val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deconvolve() {
        let (width, height) = (65, 65);

        // Generate the forward filter, and the inversion of the FFT
        // of the filter, to undo it.
        let filter = build_convolution_filter(width, height, 0.0);
        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-7);

        // Apply the deconvolution to the filter and convert back to
        // image domain - should produce a 1.0 at (0, 0), and 0.0
        // otherwise.
        let mut fftimage = FFTImage::from_image(&filter);
        fftimage.convolve(&filter_fft);
        let res = fftimage.to_image();
        let norm_res = res.normalise((res.width * res.height) as f64);

        // Check this!
        for (i, actual) in norm_res.data.iter().enumerate() {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!((actual - expected).abs() <= 1e-10);
        }
    }

    // TODO: This function is messy, but we'll submit first and tidy later.
    #[test]
    fn test_end_to_end_deconvolve() {
        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let (rays, angles) = (35, 40);

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let (width, height) = (src_img.width, src_img.height);

        let convolved =
            crate::convolution_solver::reconstruct(&scan, width, height);

        let filter = build_convolution_filter(width, height, 0.0);

        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-2);

        let mut fft = FFTImage::from_image(&convolved);
        fft.convolve(&filter_fft);

        let res = fft.to_image();
        let norm_res = res.normalise((res.width * res.height) as f64);
        let dst_img = norm_res.shift(width / 2, height / 2);

        /* TODO: For debugging...
                dst_img.save(Path::new("full_cycle.png"));

                let diff = dst_img.data
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
            .zip(dst_img.data.iter())
            .map(|(&p1, &p2)| (p1 as f64 - p2 as f64).abs())
            .sum();

        let average_error = total_error / (width * height) as f64;

        // TODO: This error is pretty huge, but small enough to mean
        // the image is roughly right.
        assert!(average_error < 30.0);
    }

    // TODO: Another hack, generating the deconvolution filter and
    // looking to see how wide the main contributing part of this kernel
    // is.
    #[test]
    fn test_generate_decon_filter() {
        let (width, height) = (65, 65);

        let filter = build_convolution_filter(width, height, 5.0);

        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-5);
        let inv_filter = filter_fft.to_image();

        let mut ext_data = inv_filter.data.iter().enumerate()
            .map(|(idx, x)| (idx / inv_filter.width, idx % inv_filter.width, *x))
            .collect::<Vec<_>>();

        ext_data.sort_by(|(_, _, a), (_, _, b)| a.abs().partial_cmp(&b.abs()).unwrap());

        let norm = 1.0 / (inv_filter.width * inv_filter.height) as f64;
        for (y, x, v) in ext_data.iter().rev().take(100).collect::<Vec<_>>() {
            println!("{} {} {} {}", y, x, v * norm, (*y as isize - (inv_filter.height / 2 + 1)).abs() + (*x as isize - (inv_filter.width / 2 + 1).abs());
        }

        let fixed = inv_filter.data.iter()
            .map(|x| (x / 250000.0) + 128.0)
            .collect::<Vec<_>>();

        let fixed_image = Image {
            data: fixed,
            ..inv_filter
        };

        fixed_image.save(Path::new("inv_filter.png"));
    }
}
