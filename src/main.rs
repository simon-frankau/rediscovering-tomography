extern crate image;
extern crate nalgebra;

use image::{GrayImage, Pixel};
use nalgebra::base::DVector;
use std::path::Path;

////////////////////////////////////////////////////////////////////////
// Image load/save
//

struct TomoImage {
    width: usize,
    height: usize,
    data: DVector<u8>,
}

fn image_load(path: &Path) -> TomoImage {
    let orig_img = image::open(path).unwrap();
    let grey_img = orig_img.into_luma8();

    let width = grey_img.width() as usize;
    let height = grey_img.height() as usize;

    // Do anything that returns a vector.
    TomoImage {
        width,
        height,
        data: DVector::from_iterator(
            width * height,
            grey_img.pixels().map(|p| p.channels()[0])),
    }
}

fn image_save(path: &Path, image: TomoImage) {
    let v = image.data.iter().map(|&x| x).collect::<Vec<u8>>();
    let img = GrayImage::from_vec(image.width as u32, image.height as u32, v).unwrap();
    img.save(path).unwrap();
}


////////////////////////////////////////////////////////////////////////
// Main entry point
//

fn main() {
    let src_img = image_load(&Path::new("images/test.png"));
    print!("Processing... ");
    let dst_img = src_img;
    println!("done!");
    image_save(&Path::new("results/test.png"), dst_img);
}
