# This Makefile just acts as a way of wrapping up a few
# commands/scripts. Not a real build thing.

target/release/rediscovering-tomography: Cargo.toml Cargo.lock src/main.rs
	cargo +nightly build --release

results:
	mkdir -p results

# Do a simple cycle from image to transformed, and back, using matrix inversion
demo: target/release/rediscovering-tomography results
	target/release/rediscovering-tomography --input-image=images/test.png \
	    --rays=35 --angles=40 --output-scan=results/test_scan.png \
	    --output-image=results/test.png --diff-image=results/test-diff.png

# And a demo based on deconvolution
demo2: target/release/rediscovering-tomography results
	target/release/rediscovering-tomography --input-image=images/test.png \
	    --rays=35 --angles=40 --output-image=results/test2.png \
	    --diff-image=results/test2-diff.png --algorithm=deconvolution

demo3: target/release/rediscovering-tomography results
	target/release/rediscovering-tomography --input-image=images/test.png \
	    --rays=200 --angles=200 --output-image=results/test3.png \
	    --diff-image=results/test3-diff.png --algorithm=deconvolution \
	    --recon-multiplier=2.0

# Try a range of angles/rays around the resolution of the image
# (32x32), and see what the associated per-pixel error generated is.
#
# Since high rays and high angles is very slow, we do two chunks,
# avoiding the slowest cases
#
# The results are collected in
# https://docs.google.com/spreadsheets/d/1mdHI4n2HNloAuYGf7aMaZISQnpawe0drygIFAd0lKsA/edit#gid=0
error_data: target/release/rediscovering-tomography results
	for RAYS in {10..80} ; do \
	    for ANGLES in {10..40} ; do \
	        /bin/echo -n "$${RAYS} $${ANGLES} " ; \
	        target/release/rediscovering-tomography --input-image=images/test.png \
	            --rays=$${RAYS} --angles=$${ANGLES} \
	            --output-scan=results/test_scan_r$${RAYS}_a$${ANGLES}.png \
	            --output-image=results/test_r$${RAYS}_a$${ANGLES}.png | \
	                grep 'RMS of per-pixel error' | grep -oE '[0-9.]+' ; \
	    done \
	done

error_data2: target/release/rediscovering-tomography results
	for RAYS in {10..40} ; do \
	    for ANGLES in {41..80} ; do \
	        /bin/echo -n "$${RAYS} $${ANGLES} " ; \
	        target/release/rediscovering-tomography --input-image=images/test.png \
	            --rays=$${RAYS} --angles=$${ANGLES} \
	            --output-scan=results/test_scan_r$${RAYS}_a$${ANGLES}.png \
	            --output-image=results/test_r$${RAYS}_a$${ANGLES}.png | \
	                grep 'RMS of per-pixel error' | grep -oE '[0-9.]+' ; \
	    done \
	done
