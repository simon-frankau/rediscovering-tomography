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

    // Core of 2D FFT, used by fft_forward and fft_inverse.
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

    fn fft_forward(img: &Image) -> FFTImage {
        let input = FFTImage {
            width: img.width,
            height: img.height,
            data: to_complex(&img.data)
        };
        input.fourier_transform(FftDirection::Forward)
    }

    fn fft_inverse(&self) -> Image {
        let output = self.clone().fourier_transform(FftDirection::Inverse);
        Image {
            width: self.width,
            height: self.height,
            data: from_complex(&output.data),
        }
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
        let v = FFTImage {
            width,
            height,
            data: (0..(height * width)).map(|x| Complex64::new(x as f64, 0.0)).collect::<Vec<_>>(),
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

        let input = Image {
            width,
            height,
            data: (0..(height * width)).map(|x| x as f64).collect::<Vec<_>>(),
        };

        let fft = FFTImage::fft_forward(&input);
        let output = fft.fft_inverse();

        assert_eq!(input.data.len(), output.data.len());
        for (in_val, out_val) in input.data.iter().zip(output.data.iter()) {
            assert!((in_val - out_val / (width * height) as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deconvolve() {
        let (width, height) = (65, 65);

        let generated = build_convolution_filter(width, height, 0.0);

        let deconvolution_fft = FFTImage::fft_forward(&generated)
            .data
            .iter()
            .map(Complex64::inv)
            .collect::<Vec<_>>();

        let fftd = FFTImage::fft_forward(&generated).data;

        let decon = fftd
            .iter()
            .zip(deconvolution_fft.iter())
            .map(|(z1, z2)| z1 * z2)
            .collect::<Vec<_>>();

        let res = FFTImage {
            width,
            height,
            data: decon
        }.fft_inverse().data;

        let res2 = res
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

        let dst_img = crate::convolution_solver::reconstruct(&scan, width, height);

        let generated = build_convolution_filter(width, height, 0.0);

        let deconvolution_fft = FFTImage::fft_forward(&generated)
            .data
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

        let fftd = FFTImage::fft_forward(&Image { width, height, data: img });

        let decon = fftd.data
            .iter()
            .zip(deconvolution_fft.iter())
            .map(|(z1, z2)| z1 * z2)
            .collect::<Vec<_>>();

        let res = FFTImage { width, height, data: decon }.fft_inverse().data;

        let res2 = res
            .iter()
            .map(|x| x / (width * height) as f64)
            .collect::<Vec<_>>();

        let res3 = Image {
            width,
            height,
            data: res2,
        };

        let img = res3.shift(width / 2, height / 2);

        /* TODO: For debugging...
                img.save(Path::new("full_cycle.png"));

                let diff: img.data
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
