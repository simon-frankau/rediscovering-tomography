///
// Image management
//
// Load and save into a vector of floats, plus basic image manipulation.
//

use image::{GrayImage, Pixel};
use nalgebra::base::{DVector};
use std::path::Path;

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
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
            data: grey_img.pixels().map(|p| p.channels()[0]).collect(),
        }
    }

    pub fn save(&self, path: &Path) {
        let img = GrayImage::from_vec(self.width as u32, self.height as u32, self.data.clone()).unwrap();
        img.save(path).unwrap();
    }

    pub fn to_dvector(&self) -> DVector<u8> {
        DVector::from_iterator(self.width * self.height, self.data.iter().copied())
    }
}
