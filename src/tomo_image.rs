///
// Image management
//
// Load and save into a vector of floats, plus basic image manipulation.
//

use image::{GrayImage, Pixel};
use std::path::Path;

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
}

// TODO: Add image manipulation functions.
