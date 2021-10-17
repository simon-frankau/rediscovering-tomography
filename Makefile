# This Makefile just acts as a way of wrapping up a few
# commands/scripts. Not a real build thing.

target/release/rediscovering-tomography: Cargo.toml Cargo.lock src/*.rs
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

# This is a version of error_data for finding the source of error associated
# with deconvolution-based reconstruction
recon_error_data: target/release/rediscovering-tomography results
	for RAYS_ANGLES in 20 40 80 160 320 640; do \
	    for MULT in 1.0 2.0 3.0 4.0 5.0 ; do \
	        /bin/echo -n "$${RAYS_ANGLES},$${MULT}," ; \
	        target/release/rediscovering-tomography --input-image=images/test.png \
	            --rays=$${RAYS_ANGLES} --angles=$${RAYS_ANGLES} \
	            --output-image=results/test_recon_$${RAYS_ANGLES}_$${MULT}.png \
	            --diff-image=results/test_recon_diff_$${RAYS_ANGLES}_$${MULT}.png \
	            --algorithm=deconvolution --recon-multiplier=$${MULT} | \
	                grep 'RMS of per-pixel error' | grep -oE '[0-9.]+' ; \
	    done \
	done

# Inject varying degrees of noise into deconvolution-based reconstruction,
# to see what that does to the error reported
noise_error_data: target/release/rediscovering-tomography results
	for NOISE in 1.0e-10 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 ; do \
	    /bin/echo -n "$${NOISE}," ; \
	    target/release/rediscovering-tomography --input-image=images/test.png \
	        --rays=80 --angles=80 \
	        --output-image=results/test_noise_$${NOISE}.png \
	        --diff-image=results/test_noise_diff_$${NOISE}.png \
	        --algorithm=deconvolution --noise=$${NOISE} | \
	            grep 'RMS of per-pixel error' | grep -oE '[0-9.]+' ; \
	done
