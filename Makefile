# This Makefile just acts as a way of wrapping up a few
# commands/scripts. Not a real build thing.

target/release/rediscovering-tomography: Cargo.toml Cargo.lock src/main.rs
	cargo +nightly build --release

results:
	mkdir -p results

# Do a simple cycle from image to transformed, and back.
demo: target/release/rediscovering-tomography results
	target/release/rediscovering-tomography --input-image=images/test.png --rays=35 --angles=40 --output-scan=results/test_scan.png --output-image=results/test.png
