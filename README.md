# Reinventing tomography

I knew nothing about Radon transforms beyond the name, and the vague
idea of how tomography works, by piecing together the original structure
from scans through it. However, it's always fascinated me, and since
having a CAT scan for the first time a few months ago, the idea's
returned to me:

*Wouldn't it be interesting to implement this, and create my own tool
that can bcak out a structure from scans?*

My initial thought was that I'd read up on the maths, and implement
it. However, it seems more fun to try to work it out for myself from
scratch, and then check my workings afterwards, so this is what I
intend to do here.

This doc is more-or-less a log of how I've reinvented tomography,
using the code in this repo, with my ideas and lessons learnt along
the way. In many places I'm playing fast and loose, since I didn't
want to take up too much time with this (despite having taken a bunch
of wall time to work on it).

*Knowledge gained in retrospect that's useful to know as we go along
will be in italics.*

## My starting point: A problem statement

The idea is that we're shooting a beam through an object at a bunch of
angles from a bunch of starting locations (in our case, in a 2D
plane), noting how much of the beam makes it through, and using that
to infer the amount of absorption at each point within that object.

What we're actually going to do here is take an image, pretend it's an
object, work out what the beams would do, and then take that data and
feed it through our own transform to try to infer the image that we
just fired beams through.

A bonus would be to see how stable the algorithms are - add noise to
the beam data, and see how much noise it adds to the inferred picture.

Somewhere near the end, I'll probably look up the actual theory for
these transforms and see how it compares.

## How to scan an object

In order to reconstruct an image from the scan data, we must first
generate the scan data. So, the first step is to write some code that
will "scan" an image.

In my project, I'll assume we'll be firing a set of parallel rays
through the object from all different angles. We could also assume the
rays are being fired in all directions from a point that moves around
the circumference of a circle. In the end we collect the same data,
being a set of rays at all different angles and different nearest
approaches to the origin. Using parallel rays seemed to simplify the
algorithms.

I'll assume we're scanning the object from all directions in a 180
degree arc (beyond that point we're just scanning the same rays from
the other side).

I'm assuming that in the real world, when you fire a beam through a
uniform medium, for every distance it travels through the material, a
certain percentage of the beam is knocked out. I don't know if this is
precisely physically true, but I'm going to pretend it is. In other
words, the medium exponentially attenuates the beam.

So, let's take the log of that, converting the product of attenuation
factors into a sum of log attenuation factors. In other words, we can
integrate the (log'd) factors along the path to get the numbers we
need. As it simplifies things and I don't want to chuck in extra
exponential/log calculations, we'll do everything with straightforward
integration over the ray paths.

This scan generates an image showing the amount of material along a
path for a given angle and distance the ray passes from the origin.

*What I have reinvented here is the "sinogram", and generating a
sinogram is the Radon transform. I'd been assuming that "Radon
transform" was the complicated reconstruction bit, but apparently it's
just the scanning algorithm. Where you see a "Scan" in my code, that's
a sinogram.*

## Reconstruction with a cheap trick

In the discrete image domain, each beam, being an integral along a
path, is just a weighted sum of pixels. Transforming from input image
to the set of ray values is just a linear transform!

The Radon transform is simply a matrix multiplication, and assuming
that the matrix is of the right shape and non-singular, we can invert
it! This means that we don't need to develop any clever algebraic
theory for the transform - we can just throw a computer at it and see
what the result is. Wow, wasn't that easy?!

## An interlude: Looking ahead, how might we do things "right"?

Note that the above approach would also work for Fourier transforms -
to do an inverse Fourier transform, we could "just" work out a forward
transform matrix, and then invert it. Of course, in practice we don't
because algebra tells us it's self-inverting.

My plan is to not throw any algebra at the problem yet, since matrix
inversion is a simple numeric method I can get immediately started on,
but it's still fun to think how I might approach it algebraically.

The problem is fundamentally inversion of the Radon transform. To make
use of symmetry, I'd probably use polar coordinates for the space
domain, so that the inverse function can be calculated in terms of
radius with a fixed angle, without loss of generality (being
rotationally symmetric).

Actually, scratch that. Assuming we've got the full set of paths
available, we can solve just for the origin, and translate the data so
that any point of interest moves to the origin. Duh.

At this point, I'd try to find ways of understanding the differentials
in density caused by differentials in the integral with respect to the
path. For example, if you move the path orthogonal to its direction,
you'll get the integral of the differential in density orthogonal to
the path, equally weighted. On the other hand, if you get the
differential of the path as you rotate the path around a point, it'll
now be weighted by distance from the center of rotation. We just need
to find a clever way of finding a version where the weighting along
the path is a Dirac delta function! Easy, I'm sure.

*In the end, the "we can just solve for the origin" solution worked
 nicely - I found an algorithm that could be applied for any point by
 translation. I never did make use of finding the differential in the
 sinogram value as we alter radius/angle, though.*

## Implementating the matrix inversion cheap trick

This section talks about the implementation of the algorithm used by
`--algorithm=matrix-inversion`.

As mentioned above, we're working with rays being simple line
integrals - no exponential attenuation.

I should have perhaps mentioned that most of the coding I'm doing
nowadays is also an excuse to learn Rust. I'm using the `nalgebra`
crate to do the matrix inversion, and the `image` crate to load and
save the (png) images being transformed.

To calculate the weights along the paths as they traverse the image,
I'm doing something akin to old-school scan conversion (in
`tomo_scan::calculate_weights_inner`). I'm making my own vectors for
this, as using a full 2D vector class just feels like overkill. The
model is that the pixels are solid squares, and the line is of zero
width, and we want to know how long the line segment that crosses each
pixel is.

I then use that to build the data for sets of parallel rays at a bunch
of different angles (*a sinogram*), in the function `tomo_scan::scan`.
Plotting this 2D data as an image reveals a nice pattern of
overlapping sinusoids that represent a chunk of matter rotating round.
The phase and amplitude represent the angle and radius of that piece
of mass. (Perhaps this implies a smarter algorithm for inverting the
transform?)

*(I never made use of this for my reconstruction algorithms, but
sinograms do look neat!)*

Finding the pseudo-inverse of the matrix and applying it... works!
Hurrah.

Interestingly, the size of the error seems to vary quite a lot with
the number of rays sampled, I'll follow this up with an investigation
into size of the error with count of parallel rays and angles.

`make demo` should exercise the image-to-transform-and-back cycle.

It's pretty darn slow, since it performs a matrix inverse on a matrix
whose dimensions are the width-height product of the image, and
rays-angles product of the scan, respectively. A naive implementation
is *O((wh)^3)*, where *w* and *h* are the dimensions of the image in
pixels. Images much bigger than 30x30 get pretty slow on my laptop.

## Convergence of the matrix inversion approach

Back in a previous life, I worked in a quant team at a bank, where
there were a lot of numerical methods going on, and ocasionally I'd be
interested in convergence of those methods (as well as trying to
remember what I learnt on the subject as an undergrad).

My memory was that doing this stuff could quite happily turn into a
huge time sink, but it's useful enough to do a bit of and then move
on...

What I did was run the conversion for various numbers of rays and
angles to see how quickly the reconstruction converges on the
original. Empirical studies of algorithmic convergence are fun!

The `Makefile` targets `error_data` and `error_data2` generate the
data needed (two targets were used to avoid trying to calculate the
error for a bunch of slow-to-calculate, uninteresting cases. The
results are in [this
sheet](https://docs.google.com/spreadsheets/d/1mdHI4n2HNloAuYGf7aMaZISQnpawe0drygIFAd0lKsA/edit).

Interesting things to note are:

 * For my test image, of size 32x32 (1024 pixels), convergence
   (stabilising at roughly an error of 0.6 per pixel on an 8-bit image
   - approximately 0.25%) seems to happen when rays * angles ~= 1300.
   Slight oversampling seems needed, I've no idea what the theoretical
   basis might be!
 * In the region where angles ~= rays, error seems to decrease in a
   straight line with angles * rays.
 * It looks like when rays << angles, or angles << rays, the smaller
   number constrains convergence above the minimal error, but I can't
   tell if this is a genuine asymptote or just much slower
   convergence.
 * As this is a small image, you can see discretisation effects,
   particularly with odd/even ray count.

It's all very intriguing, but probably best not to get too distracted
by it.

*I later discovered that the lower bound on the error comes from the
discretisation applied when generating an 8-bit image, and convergence
is much better if not discretised! One of the lessons I'd forgotten is
if you numerical method seems to converge on some non-zero error as
you tweak parameters, there's probably a source of error that isn't
associated with the parameters you're tweaking, and you may want to
investigate! At this stage, I was working out average per-pixel
absolute error. Later I switched to RMS per-pixel error.*

## In search of a more efficient algorithm

An algorithm that's *O(n^6)* in the length of the image side is... not
very attractive in the real world. While I'd proven to myself that I
could reconstruct a scan (what I'd originally set out to achieve),
could I do it in a way that's practical in the real world?

The next period of this project involved a fair amount of starting
into space and thinking about the problem in my spare time. It did,
admittedly, also involve writing out and manipulating a bunch of
integrals, but that turned out to be much less effective!

My key insight was that if we integrate over all the rays passing
through a point, we get a weighted sum of pixel values from the
original image, symmetric around the point, with the weights
increasing towards the centre. In other words, we've managed to get
the value of the original image passed through a blurring filter.
Moreover, this same-shaped filtered-value can be found for all points
in the image by integrating over the rays passing through that point.

Put another way, taking the Radon transform and, for each point in
image space, integrating over all the rays going through that point,
is equivalent to convolving the original image with a *1/r* weighted
filter.

*I later found out that this reconstruction approach by integrating
over all the rays through the point is known as "back propagation".*

To get this blurred reconstruction, you can use the
`--algorithm=convolution` mode on the binary.

Once we have a convolved image, we can apply deconvolution techniques
to get back to the original image!

## On deconvolution

**Disclaimer:** Convolution of two signals officially involves
flipping one of the signals over before integrating them against each
other, but the filters I'm looking at are symmetrical, so I'm skipping
over that bit. Generally, I'm playing fast and loose on the maths,
doing just enough to make things work...

I'd been aware of deconvolution for a while, and been looking for an
excuse to play around with it. The idea is that a convolution in the
frequency domain is achieved by performing a point-wise multiplication
of the image with the filter. Therefore, inverting the convolution is
simply a matter of doing a point-wise division by the filter instead.
Easy, eh?

In practice, this turned out to be the fiddliest, most "learn how to
do numerical methods less badly" part of the whole exercise, and I've
got to admit I never achieved absolutely awesome convergence. Read on
to learn more. ;)

*I found this deconvolution filter by building the convolution filter
and then numerically converting it to a deconvolution filter in
frequency space. I could have algebraically attempted to find the
deconvolution filter instead. The standard algorithm is to use a
"ramp" high-pass filter. This makes intuitive sense, since the
blurring performed by back-propagation is effectively a low-pass
filter, so to "balance the frequency response" back we want to apply a
high-pass filter. But no, I took the numerical route...*

I picked up the `rustfft` crate and hacked away. It's easy to use, but
getting good deconvolution results needs more than that:

### Building the filter

In order to build a deconvolution filter by inverting the convolution
filter, you need to build the convolution filter! This turned out to
be trickier than I expected, and is what the code in
`convolution_solver::build_convolution_filter` is about.

The filter applies a weight proportional to *1/r* to each pixel.
Unfortunately, that means... infinite weight at the origin?! We need
*some* value there, which is likely to be a really large weight to
behave correctly, but it certainly can't be infinite! We need some way
of discretising this appropriately. What I ended up doing is having a
minimal *r* that is half the radius of a pixel, and it seems to do the
trick. I can't say it's mathematically well-founded, though!

Various tests in `convolution_solver` then exercise the filter.
Because of the behaviour at small radii, it behaves somewhat
differently for odd (filter centred in the middle of a pixel) and even
(centre between pixels) filter size. The tests include building a
sinogram of a point and back-propagating it, and checking that it
roughly matches the filter we generate
(`test_convolution_comparison`). It roughly does, but it leaves me
wondering if it's a non-trivial source of error.

### Frequency response

A convolution filter might highly attenuate some frequencies - the
filter requires you to multiply by a very small number. The
deconvolution filter would then have you multiply by a very large
number, introducing noise into the reconstruction. This is indeed what
happened when I naively tried deconvolution, resulting in images with
highly visible high-frequency artefacts.

What do we do? Well, those frequencies are basically lost, there's no
way to reconstruct them, so let's not try. If a transform coefficient
is small, I just set the inverse coefficient to zero, instead of a
huge number. Yes, some component of the original image is lost, but it
was already lost, and adding noise doesn't bring it back.

### How big should the filter be?

Performing back-propagation is like using a *1/r* convolution filter,
and *1/r* dies off pretty slowly with radius. For perfect
reconstruction, we'd want an infinitely large image. How wide do we
need the convolved image to be in order to get a decent
reconstruction?

I'm all for the empirical approach, rather than hard work with
algebra. We can find out how the deconvolution filter works by taking
the convolution filter, FFTing it, inverting it in frequency space,
and FFTing it back. This will give us an image space deconvolution
filter that we can inspect.

If you do this, what you find is that most of the weight is right at
the centre of the filter kernel. This is what I did in
`deconvolution_solver::test_generate_decon_filter`.

Hmmm. If most of the weight is in a small filter kernel, maybe we can
just apply that small kernel (perhaps in image space), and get the
results we want quite cheaply? Let's try it!

### A dead-end: trying a small deconvolution filter kernel

So, we've just found out most of the weights in the deconvolution
filter are in the centre. Using a FFT to represent a tiny little
kernel seems both inefficient and unlikely to be accurate (a tiny
kernel having a lot of high-frequency components), so the next step is
to try to apply the kernel in the image domain and see how well it
performs.

In `deconvolution_solver::test_decon_kernel`, I snip a 9x9
deconvolution filter out of the FFT-based deconvolution filter (yes,
not a very scientific approach, I know), and try applying it as a
naive convolution...

It turns out the per-pixel RMS error is pretty much the same as that
from the unfiltered back-propagation approach in
`convolution_solver::test_reconstruct`. Looking at the reconstructed
images, the filtered images are sharper, but the bits that should be
totally black are dark grey because the filter fails to take account
of the more distant parts of the image that should be subtracted. In
other words, it looks like the low-weighted parts of the filter do
still have value, and we'd be best off using a large kernel and an
FFT-based deconvolution.

An interesting dead end, but it's time to get back to the main track.

### An Aside: Checking quality of fit

Actually, first a quick aside on checking how well an algorithm
performs. I'd been using per-pixel error calculations - initially with
absolute error, later switching to RMS. That's not the important bit.
The important bit was that I forgot to set a baseline. So, when
back-propagation gave me an error number, it didn't mean much to me,
but it looked plausible. What I'd failed to realise until rather later
is that because my test image is mostly black, a completely black
image doesn't have a huge per-pixel error. Only by finding out what
the base error was (in
`convolution_solver::test_null_reconstruction`), did I realise quite
how badly the basic algorithms scored!

The other lesson I learnt was that it's useful to not just look at a
summary number, but look at an image diffing the source and
reconstruction. This makes it much clearer when you've made a mistake,
in a way that a single number doesn't. For example, if the
reconstruction is misaligned with the original image by a pixel, the
error number might not be huge, but it's very clear in the image.

### Back to FFT-based deconvolution

Having decided a small deconvolution kernel was a dead end, I decided
to go back to full FFT-based deconvolution, and see if I could improve
on the error in the simple `test_basic_image_deconvolve`
implementation.

TODO: Rewrite from here on down.

While refactoring, I've noticed that the reconstruction error of the
matrix approach has shrunk dramatically since I first wrote the code,
and don't know why. Something for investigation! All the numbers in
the error sheet are now wrong.

And... it turns out to be the discretisation to u8 upon
reconstruction. Wihout that, the error goes way down in the case where
it's basically converged.

TODO: Consider regenerating the error data spreadsheet with this
source of error removed, in order to better see the convergence.

# More work...

Important factors in accuracy:

 * Getting the filter aligned on the centre of a pixel
 * Getting the input and output correctly aligned (previous point is basically
   sub-pixel alignment)
 * And hence, making sure your code works for odd- and even-sized images.

Creating another spreadsheet tracking the error from the
matrix-inversion approach with a variety of numbers of rays and
angles, the results are at
https://docs.google.com/spreadsheets/d/1I7ISM7KZHVbOBcUQOYR43Fx0JnAHdvksXe5aa2Dl9FE/edit#gid=0
. Interesting to note that now the error doesn't get floored around
0.05 per-pixel, but goes all the way to zero for sufficient rays and
angles - we get a fully-accurate reconstruction, which makes sense
when the inverse matrix is fully determined, but is still an
impressive result. The raw data is stored in analysis/errors.csv.

After this, we're implementing deconvolution-based reconstrution in
the command-line tool. I believe most of the error from this algorithm
comes from the fact that FFTs assume repeating waveforms are being
convoluted, when what we're actually trying to (de)convolve is (the
inverse of) a finite part of a filter that dies off slowly with 1/r,
and a bounded-size scan. We can reduce this error by increasing the
size of the scan and the filter around the original image, so that
asymptotically we approach the non-repeating case. (Some other error
comes from the frequencies attenuated by the convolution, but we're
ignoring that.)

Looking at the error as we vary the rays/angles count and the size of
the reconstruction image (raw data in analysis/recon_error.csv,
tabular data in the sheet), we can see that the error flattens off
around a --recon-multiplier of 2.0, and the error bottoms out when the
number of rays and angles is roughly twice the image size (for a 32x32
image, 80 rays and 80 angles actually minimises error, on this
relatively sparse set of test cases).

As error does not asymptotically head to zero, or even close to that,
it seems like there's some other source of error. Looking at the diff
images on the reconstruction of the test image, it's not noise, it's
the whole of the central question mark being slightly the wrong level,
even at many rays/angles and a large reconstruction image. So, some
normalisation step isn't quite right, or there's just some other
source of error. However, I feel I've got something that's not bad
now, and am hitting diminishing returns on playing about with this
thing.

# Switching over: The official algorithm

Watched https://www.youtube.com/watch?v=f0sxjhGHRPo from
https://twitter.com/sigfpe/status/1441455463439110148 , which also
talks about back-propagation. I don't tend to get the details of
mathematical arguments from videos, but it does look suspiciously like
there's a filter in the frequency domain and a reconstruction across
all angles. In the video, the filter is applied before the summing
across all angles. I haven't thought about it too hard, but given
everything's linear I would expect you could do the integration and
filter in either order.

Nice to see cutting off the high frequencies to reduce noise is
included, given the fun I had with finding the deconvolution due to
attenuated frequencies, mostly in the high-frequency part. The nice
thing about doing this mathematically is that you get the
deconvolution filter directly, rather than by inverting the forward
filter, which I assume improves accuracy.

Moving on to Wikipedia, judging by
https://en.wikipedia.org/wiki/Tomography, it looks like I've
reinvented "filtered back projection". It's interesting to see it's a
relatively noisy approach compared to
https://en.wikipedia.org/wiki/Iterative_reconstruction. The
statistical methods look really cool, but I'm not going to spend more
time on this!

In filtered back projection, looks like I've accidentally reinvented
back-projection as the integral around each point. The filter to use
is a "ramp" filter, applied before back-propagation. I will not be
implementing it.

Adding the ability to add noise to the intermediate scan, and varying
the noise, results are in
https://docs.google.com/spreadsheets/d/1I7ISM7KZHVbOBcUQOYR43Fx0JnAHdvksXe5aa2Dl9FE/edit#gid=1342061078
, also analysis/noise_errors.csv, we can see that for large errors
being added, the RMS error out is proportional to the noise added. As
there is some base level error in my reconstruction (darn it),
asymptotically it then goes to that base error as the added noise goes
to zero.

Interestingly, looking at the kind of noise we see in the resulting
image, having added noise into the scan, it does just look like fairly
uniform noise - there aren't artefacts like streaky lines or blobby
errors or anything. This could be because the noise was added
uniformly to the scan, and if we applied a big impulse of noise in one
place to the scan we might expect to see weird artefacts in the
reconstructed image. However, I don't plan to spend the time following
that up!

TODO: nonogram!

TODO:

