//
// Generate a more accurate reconstruction by deconvolution
//
// This file takes the convolved reconstruction, and then
// deconvolves it to obtain a better approximation of the original.
//

use rustfft::{num_complex::Complex64, FftDirection, FftPlanner};
use std::ops::{Index, IndexMut};

use crate::tomo_image::Image;
use crate::tomo_scan::Scan;

use crate::convolution_solver::build_convolution_filter;

#[cfg(test)]
use crate::tomo_scan::scan;
#[cfg(test)]
use itertools::iproduct;
#[cfg(test)]
use std::path::Path;

////////////////////////////////////////////////////////////////////////
// Utilities
//

// We're not going to do any fancy real-valued optimisations for the
// FFT, we simply convert to and from complex numbers.
fn to_complex(v: &[f64]) -> Vec<Complex64> {
    v.iter().map(|re| Complex64::new(*re, 0.0)).collect()
}

// Doing a clever real-to-complex conversion for the FFTs would allow
// us to define away the error case of the complex component being
// non-trivial, but we're keeping things simple.
fn from_complex(v: &[Complex64]) -> Vec<f64> {
    const EPSILON: f64 = 1e-5;
    assert!(
        v.iter()
            .map(|z| z.im.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            < EPSILON
    );
    v.iter().map(|z| z.re).collect()
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

impl Index<(usize, usize)> for FFTImage {
    type Output = Complex64;
    fn index(&self, (x, y): (usize, usize)) -> &Complex64 {
        &self.data[y * self.width + x]
    }
}

impl IndexMut<(usize, usize)> for FFTImage {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Complex64 {
        &mut self.data[y * self.width + x]
    }
}

impl FFTImage {
    // Transpose, mutating self. Used by fourier_transform.
    fn transpose(&mut self) {
        let mut new_data = Vec::with_capacity(self.width * self.height);
        for y in 0..self.width {
            for x in 0..self.height {
                new_data.push(self[(y, x)]);
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
            width: self.width,
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
            data: to_complex(&img.data),
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

        self.data = self
            .data
            .iter()
            .zip(filter.data.iter())
            .map(|(z1, z2)| z1 * z2)
            .collect::<Vec<_>>();
    }
}

////////////////////////////////////////////////////////////////////////
// Deconvolution-based reconstruction entry point
//

pub fn reconstruct(scan: &Scan, width: usize, height: usize, recon_multiplier: f64) -> Image {
    // FFT assumes we're convolving repeating patterns, when actually we're
    // convolving a finite approximation of an infinite filter with a
    // bounded scan image. As we increase the space around the image,
    // in the limit it should converge on what we want.
    //
    // "recon_multiplier" is how many times the image used during
    // reconstruction is wider than the base image. Minimum 1.0.
    assert!(recon_multiplier >= 1.0);

    // Add half the extra amount on each side.
    let per_side_factor = (recon_multiplier - 1.0) / 2.0;
    let w_overscan = (width as f64 * per_side_factor) as usize;
    let h_overscan = (height as f64 * per_side_factor) as usize;

    // 1. Create the convolved image.
    let convolved_img = crate::convolution_solver::reconstruct_overscan(
        scan, width, height, w_overscan, h_overscan,
    );

    // 2. Build the deconvolution filter in frequency space.
    //
    // Make the filter centred at the centre of a pixel, by being
    // odd-sized.
    let filter = build_convolution_filter(width | 1, height | 1, w_overscan, h_overscan).trim(
        0,
        0,
        2 * w_overscan + width,
        2 * h_overscan + height,
    );

    let mut filter_fft = FFTImage::from_image(&filter);
    filter_fft.invert(1e-2);

    // 3. Perform the deconvolution.
    let mut fft = FFTImage::from_image(&convolved_img);
    fft.convolve(&filter_fft);
    let res = fft.to_image();

    let norm_res = res.scale_values(1.0 / (res.width * res.height) as f64);

    // 4. Shift image to centre, trim around it, and return the result.
    norm_res
        .shift((norm_res.width + 1) / 2, (norm_res.height + 1) / 2)
        .trim(w_overscan, h_overscan, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let width = 5;
        let height = 7;
        let data = (0..(height * width))
            .map(|re| Complex64::new(re as f64, 0.0))
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
                assert_eq!(v[(x, y)], vt[(y, x)]);
            }
        }
    }

    // Test that doing an FFT and then inverting it gets us back to
    // the starting image.
    #[test]
    fn test_double_fft() {
        // Give it some awkward sizes. :)
        let width = 13;
        let height = 17;
        let data = (0..(height * width)).map(|p| p as f64).collect::<Vec<_>>();

        let input = Image {
            width,
            height,
            data,
        };

        let fft = FFTImage::from_image(&input);
        let inv = fft.to_image();
        let output = inv.scale_values(1.0 / (inv.width * inv.height) as f64);

        assert_eq!(input.data.len(), output.data.len());
        for (in_val, out_val) in input.data.iter().zip(output.data.iter()) {
            assert!((in_val - out_val).abs() < 1e-10);
        }
    }

    // Test that deconvolving the filter returns an impulse function.
    #[test]
    fn test_filter_deconvolve() {
        let (width, height) = (65, 65);

        // Generate the forward filter, and the inversion of the FFT
        // of the filter, to undo it.
        let filter = build_convolution_filter(width, height, 0, 0);
        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-7);

        // Apply the deconvolution to the filter and convert back to
        // image domain - should produce a 1.0 at (0, 0), and 0.0
        // otherwise.
        let mut fftimage = FFTImage::from_image(&filter);
        fftimage.convolve(&filter_fft);
        let res = fftimage.to_image();
        let norm_res = res.scale_values(1.0 / (res.width * res.height) as f64);

        // Check this!
        for (i, actual) in norm_res.data.iter().enumerate() {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!((actual - expected).abs() <= 1e-10);
        }
    }

    // Test that performing a basic deconvolution, without paying
    // attention to any details, gets a rough reconstruction.
    #[test]
    fn test_basic_image_deconvolve() {
        // 1. Scan and generate the convolved image.

        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let (rays, angles) = (35, 40);

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let (width, height) = (src_img.width, src_img.height);

        let convolved_img = crate::convolution_solver::reconstruct(&scan, width, height);

        // 2. Build the deconvolution filter in frequency space.
        let filter = build_convolution_filter(width, height, 0, 0);

        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-2);

        // 3. Perform the deconvolution.
        let mut fft = FFTImage::from_image(&convolved_img);
        fft.convolve(&filter_fft);
        let res = fft.to_image();

        let norm_res = res.scale_values(1.0 / (res.width * res.height) as f64);
        let dst_img = norm_res.shift(width / 2, height / 2);

        // This error is pretty huge, but small enough to mean the
        // image is roughly right.
        //
        // Note the error is smaller than in
        // convolution_solver::test::test_reconstruct - we're doing
        // something right!
        let rms_error = src_img.rms_diff(&dst_img);
        assert!(40.0 < rms_error && rms_error < 45.0);
    }

    // This test documents how the majority of the weight of the
    // deconvolution filter is right in the centre.
    #[test]
    fn test_generate_decon_filter() {
        // 1. First we build a reasonably large deconvolution filter...
        let (width, height) = (65, 65);
        let overscan = 5;
        let (w_over, h_over) = (overscan * width, overscan * height);
        let filter = build_convolution_filter(width, height, w_over, h_over);
        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-5);
        let inv_filter = filter_fft.to_image();

        // 2. Then we show that the biggest weights are at the centre.

        // Get a sorted list of all the weights...
        let mut sorted_weights = inv_filter.data.iter().map(|p| p.abs()).collect::<Vec<_>>();
        sorted_weights.sort_by(|a, b| b.partial_cmp(&a).unwrap());

        // And then a sorted list of the weights in the centre.
        let centre_radius = 6;
        // Centre is off-by-one from where you might expect -
        // see comment on test_decon_kernel
        let centre_x = inv_filter.width as isize / 2 + 1;
        let centre_y = inv_filter.height as isize / 2 + 1;
        let mut centre_sorted_weights = iproduct!(0..(inv_filter.width), 0..(inv_filter.height))
            .filter(|(x, y)| {
                let dx = *x as isize - centre_x;
                let dy = *y as isize - centre_y;
                dx * dx + dy * dy < centre_radius * centre_radius
            })
            .map(|p| inv_filter[p].abs())
            .collect::<Vec<_>>();
        centre_sorted_weights.sort_by(|a, b| b.partial_cmp(&a).unwrap());

        // And show that the biggest weights in the centre are the
        // biggest weights.
        let last_centre_idx = centre_sorted_weights.len() - 1;
        assert_eq!(
            sorted_weights[last_centre_idx],
            centre_sorted_weights[last_centre_idx]
        );

        // 3. Show that the weights fall off quickly - the 21st element
        // is a tiny fraction of the biggest weight.
        let weight_ratio = sorted_weights[21] / sorted_weights[0];
        assert!(0.001 < weight_ratio && weight_ratio < 0.002);

        // 4. Show that the vast majority of the overall weight is
        // in the centre.
        let centre_weight = centre_sorted_weights.iter().sum::<f64>();
        let total_weight = sorted_weights.iter().sum::<f64>();
        let centre_fraction = centre_weight / total_weight;
        assert!(0.97 < centre_fraction && centre_fraction < 0.98);
    }

    // I've observed that the largest weights are towards the centre of
    // the convolution filter (see test_generate_decon_filter), so this
    // test attempts to perform a deconvolution with just a small kernel,
    // and see how well it performs.
    #[test]
    fn test_decon_kernel() {
        // Size of the filter kernel we will construct.
        // Use odd values to have a central point.
        let (k_width, k_height) = (9, 9);

        // 1. Create the convoluted image to deconvolve.

        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let (rays, angles) = (35, 40);

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let (width, height) = (src_img.width, src_img.height);

        // Generate a convolved image with overscan. The size of
        // the image will be reduced when we perform the convolution.
        let convolved = crate::convolution_solver::reconstruct_overscan(
            &scan,
            width,
            height,
            k_width / 2,
            k_height / 2,
        );

        // 2. Generate the image-space deconvolution kernel.

        // Make it odd in size, so that the filter is centred in the
        // middle of a pixel.
        let filter = build_convolution_filter(width | 1, height | 1, 0, 0);
        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-5);
        let inv_filter = filter_fft.to_image();

        // Cut out the core of the filter, to create a small kernel
        // filter that contains the biggest coefficients.
        //
        // The extra "+ 1" is because the centre of the inverse filter
        // is moved over by 1 pixel. It's not entirely clear to me
        // exactly why this is, but I suspect it's because the sample
        // points are at the start of the discretisation intervals, and
        // so when things are flipped in the "time" domain as part of it
        // being a convolution, the points align with the end of the
        // discretisation intervals, pushing everything along by 1 pixel.
        // This is horrific hand-waving around maths I don't fully get, but
        // suffice to say the shift is necessary.
        let k_x_offset = (inv_filter.width - k_width + 1) / 2;
        let k_y_offset = (inv_filter.height - k_height + 1) / 2;
        let mut kernel_img = inv_filter.trim(k_x_offset + 1, k_y_offset + 1, k_width, k_height);

        // To get the appropriate weighting, we need to find what
        // applying this new deconvolution filter to the original
        // convolution filter gives - we need to normalise this
        // to 1.0 to make it work.
        let weight = filter
            .trim(k_x_offset, k_y_offset, k_width, k_height)
            .naive_convolve(&kernel_img)[(0, 0)];
        kernel_img = kernel_img.scale_values(1.0 / weight);

        // 3. Apply the kernel to get a reconstruction.

        // The convolution shrinks the overscanned image back to its
        // original size.
        let res = convolved.naive_convolve(&kernel_img);

        // 4. Quantify the error in this reconstruction.

        let rms_diff = res.rms_diff(&src_img);

        // The RMS per-pixel difference is quite high, a lot of
        // which comes from the black parts coming out dark grey
        // because the small kernel filter doesn't remove the
        // contribution from far-away bright areas.
        assert!(60.0 < rms_diff && rms_diff < 65.0);
    }

    // Exercise a deconvolution where we use an overscan to take account
    // of the way the filter (and its inverse) is not limited to the size
    // of the original image.
    #[test]
    fn test_image_deconvolve() {
        let overscan_factor = 2;

        // 1. Scan and generate the convolved image.

        // Choose numbers different from the image dimensions, to
        // avoid missing bugs.
        let (rays, angles) = (35, 40);

        let src_img = Image::load(Path::new("images/test.png"));
        let scan = scan(&src_img, angles, rays);

        let (width, height) = (src_img.width, src_img.height);

        let convolved_img = crate::convolution_solver::reconstruct_overscan(
            &scan,
            width,
            height,
            width * overscan_factor,
            height * overscan_factor,
        );

        // 2. Build the deconvolution filter in frequency space.

        // Make the filter centred at the centre of a pixel, by being
        // odd-sized.
        let filter = build_convolution_filter(
            width | 1,
            height | 1,
            width * overscan_factor,
            height * overscan_factor,
        )
        .trim(
            0,
            0,
            width * (2 * overscan_factor + 1),
            height * (2 * overscan_factor + 1),
        );

        let mut filter_fft = FFTImage::from_image(&filter);
        filter_fft.invert(1e-2);

        // 3. Perform the deconvolution.
        let mut fft = FFTImage::from_image(&convolved_img);
        fft.convolve(&filter_fft);
        let res = fft.to_image();

        let norm_res = res.scale_values(1.0 / (res.width * res.height) as f64);
        // Shift image to centre, and trim around it.
        let dst_img = norm_res
            .shift((norm_res.width + 1) / 2, (norm_res.height + 1) / 2)
            .trim(
                width * overscan_factor,
                height * overscan_factor,
                width,
                height,
            );

        // Pretty decent improvement on test_basic_image_deconvolve.
        //
        // Having a pixel-centred filter (rather than between-pixel-centred)
        // filter reduces the error from around 38-39.
        let rms_error = src_img.rms_diff(&dst_img);
        assert!(17.0 < rms_error && rms_error < 18.0);
    }
}
