///
// Image management
//
// Load and save into a vector of floats, plus basic image manipulation.
//

use image::{GrayImage, Pixel};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
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
        let data_as_u8: Vec<u8> = self.data.iter().map(|x| x.max(0.0).min(255.0) as u8).collect::<Vec<_>>();
        let img = GrayImage::from_vec(self.width as u32, self.height as u32, data_as_u8).unwrap();
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
                        pixel += self.data[(y + sub_y) * self.width + (x + sub_x)];
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

    // The way convolution works means the deconvolved image is shifted by
    // an amount equivalent to the coordinates of the centre of the filter.
    // "shift" can undo this.
    pub fn shift(&self, x_shift: usize, y_shift: usize) -> Image {
        let (width, height) = (self.width, self.height);

        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let src_y = (y + y_shift) % height;
                let src_x = (x + x_shift) % width;
                data.push(self.data[src_y * width + src_x]);
            }
        }

        Image {
            width,
            height,
            data,
        }
    }

    // Normalise so that x is transformed to 1.0.
    pub fn normalise(&self, x: f64) -> Image {
        let data = self.data.iter().map(|y| y / x).collect::<Vec<_>>();
        Image {
            width: self.width,
            height: self.height,
            data
        }
    }
}
