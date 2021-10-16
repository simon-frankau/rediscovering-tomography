///
// Image management
//
// Load and save into a vector of floats, plus basic image manipulation.
//

use image::{GrayImage, Pixel};
use std::ops::{Index, IndexMut};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl Index<(usize, usize)> for Image {
    type Output = f64;
    fn index<'a>(&'a self, (x, y): (usize, usize)) -> &'a f64 {
        &self.data[y * self.width + x]
    }
}

impl IndexMut<(usize, usize)> for Image {
    fn index_mut<'a>(&'a mut self, (x, y): (usize, usize)) -> &'a mut f64 {
        &mut self.data[y * self.width + x]
    }
}

impl Image {
    pub fn load(path: &Path) -> Image {
        let orig_img = image::open(path).unwrap();
        let grey_img = orig_img.into_luma8();

        let width = grey_img.width() as usize;
        let height = grey_img.height() as usize;

        // Do anything that returns a vector.
        Image {
            width,
            height,
            data: grey_img.pixels().map(|p| p.channels()[0] as f64).collect(),
        }
    }

    pub fn save(&self, path: &Path) {
        let data_as_u8: Vec<u8> = self.data
            .iter()
            .map(|p| p.max(0.0).min(255.0) as u8)
            .collect::<Vec<_>>();
        let img = GrayImage::from_vec(
            self.width as u32,
            self.height as u32, data_as_u8).unwrap();
        img.save(path).unwrap();
    }

    // Scales down the image by the given factor, which must divide
    // width and height. Useful for oversampling.
    pub fn downscale(&self, factor: usize) -> Image {
        assert!(self.width % factor == 0);
        assert!(self.height % factor == 0);

        let new_w = self.width / factor;
        let new_h = self.height / factor;

        let mut data = Vec::new();
        for y in (0..new_h).map(|y| y * factor) {
            for x in (0..new_w).map(|x| x * factor) {
                let mut pixel: f64 = 0.0;
                for sub_y in 0..factor {
                    for sub_x in 0..factor {
                        pixel += self[(x + sub_x, y + sub_y)];
                    }
                }
                data.push(pixel / (factor as f64 * factor as f64));
            }
        }

        Image {
            width: new_w,
            height: new_h,
            data,
        }
    }

    // The way FFT convolution works means the deconvolved image is shifted by
    // an amount equivalent to the coordinates of the centre of the filter.
    // "shift" can undo this.
    pub fn shift(&self, x_shift: usize, y_shift: usize) -> Image {
        let (width, height) = (self.width, self.height);

        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let src_y = (y + y_shift) % height;
                let src_x = (x + x_shift) % width;
                data.push(self[(src_x, src_y)]);
            }
        }

        Image {
            width,
            height,
            data,
        }
    }

    // Trim an image down to a sub-image, size (w, h), starting at
    // (x_off, y_off) in the original image.
    pub fn trim(&self, x_off: usize, y_off: usize, w: usize, h: usize) -> Image {
        assert!(x_off + w <= self.width);
        assert!(y_off + h <= self.height);

        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.push(self[(x + x_off, y + y_off)]);
            }
        }

        Image {
            width: w,
            height: h,
            data,
        }
    }

    // Expand an image by adding a border of zeros, placing the
    // original image at (x_off, y_off), and resizing the image to
    // (w, h).
    //
    // Useful debugging/testing function, even if not used right now.
    #[allow(dead_code)]
    pub fn expand(&self, x_off: usize, y_off: usize, w: usize, h: usize) -> Image {
        assert!(x_off + self.width <= w);
        assert!(y_off + self.height <= h);

        let mut img = Image {
            width: w,
            height: h,
            data: vec![0.0; w * h],
        };

        for y in 0..self.height {
            for x in 0..self.width {
                img[(x + x_off, y + y_off)] = self[(x, y)];
            }
        }

        img
    }

    // Perform a naive convolution by applying a filter on a per-pixel
    // basis, no FFT. Ok for small filter kernels. Technically there's
    // supposed to be some mirroring going on in a convolution, but
    // our filters are symmetric so it makes no difference here.
    // Rather than doing anything clever at the edges, we just produce
    // a smaller output image than the input.
    pub fn naive_convolve(&self, filter: &Image) -> Image {
        let mut data = Vec::new();
        for y in 0..(self.height - filter.height + 1) {
            for x in 0..(self.width - filter.width + 1) {
                let mut pixel = 0.0;
                for y2 in 0..filter.height {
                    for x2 in 0..filter.width {
                        pixel += filter[(x2, y2)] * self[(x + x2, y + y2)];
                    }
                }
                data.push(pixel);
            }
        }

        Image {
            width: self.width - filter.width + 1,
            height: self.height - filter.height + 1,
            data,
        }
    }

    // Find the difference between two images.
    pub fn diff(&self, other: &Image) -> Image {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);

        let diff = self.data
            .iter().zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();

        Image {
            width: self.width,
            height: self.height,
            data: diff
        }
    }

    // Calculate the RMS of per-pixel difference between two images
    pub fn rms_diff(&self, other: &Image) -> f64 {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);

        let sum_squares_diff: f64 = self.data
            .iter().zip(other.data.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        (sum_squares_diff / (self.width * self.height) as f64).sqrt()
    }

    // Multiply all the points by the given value.
    pub fn scale_values(&self, a: f64) -> Image {
        let data = self.data.iter().map(|b| a * b).collect::<Vec<_>>();
        Image {
            width: self.width,
            height: self.height,
            data
        }
    }

    // Add an offset to all values.
    pub fn offset_values(&self, a: f64) -> Image {
        let data = self.data.iter().map(|b| a + b).collect::<Vec<_>>();
        Image {
            width: self.width,
            height: self.height,
            data
        }
    }

    // Normalise the data between 0 and the given value. Useful for
    // making random test data into an image that can be viewed.
    //
    // It takes two percentiles, representing the points we want to
    // map to 0 and the given value. Well, not percentiles, but values
    // 0-1, but I'm not quite sure what to call that. Values outside
    // the range are *not* clamped.
    //
    // Useful debugging/testing function, even if not used right now.
    #[allow(dead_code)]
    pub fn normalise(&self, scale: f64, lower: f64, upper: f64) -> Image {
        let mut sorted_data = self.data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(&b).unwrap());

        let max_idx = sorted_data.len() - 1;
        let lower_val = sorted_data[(lower * max_idx as f64).round() as usize];
        let upper_val = sorted_data[(upper * max_idx as f64).round() as usize];

        let data = self.data
            .iter()
            .map(|p| (p - lower_val) * scale / (upper_val - lower_val))
            .collect::<Vec<_>>();

        Image {
            data,
            ..*self
        }
    }
}
