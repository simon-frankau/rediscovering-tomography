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

**TODO: Finish tidying up the contents of this file**

# Next steps

It'd be nice to work out a more mathematically-structured and
efficient way of performing the transformation. I think a useful
approach is to start looking at the transform in polar coordinates. In
that case, the inverse transform can be characterised by the linear
transform required for one radial slice, and then it can be applied to
all the different radial slices. Moreover... this kind of "perform the
dot product, but for all rotations" sounds suspiciously like a
convolution. A Fourier transform can make the implementation more
efficient, but also hints at how we might approach this transform
algebraically...

# Next steps again

Having thought a little further, we can, for each point, generate an
integral of all the rays that pass through that point, integrating
over the different angles. This will give us an integral over the
space that is radially symmetric, and we can get this for each point
in the grid. The result is a convolution of the image with a weighting
function, and we can deconvolve it. At least, that's the plan. Let's
try...

Right now we have `--algorithm=matrix-inversion` as the default
existing algorithm, and `--algorithm=convolution` as generating the
(not deconvolved) integral version.

# Further next steps

Deconvolution requires performing the inverse of a convolution, which
can be done by taking the Fourier transform of the filter, and taking
the reciprocal of all its values (since convolution is just
multiplication in the frequency domain).

One tricky step with this is that if a frequency is highly attenuated
by the filter, the coefficient in the Fourier domain will be tiny, its
reciprocal huge, and the deconvolution can be incredibly noisy. So we
just drop frequencies which have really small coefficients, and live
with dropping some frequencies.

The next piece of trickiness is to work out to size the filters. The
1/r convolution filter that performing the scan and then integrating
gives us an infinitely-wide filter, that only tails off gradually with
distance. It's possible we'll only get good results if we create an
integral-of-scan that's much larger than the original image. How big
does the filter need to be? Well, we can take the deconvolution filter
FFT we just created, and bring it back to the image domain.

If we do this, we find there's a really small kernel with large
coefficients, and the rest is hardly used (indeed, I don't know how
wide the kernel is if we did this algebraically, rather than with
throw-it-at-the-wall-and-see-what-sticks numerical methods). Using a
FFT to represent a tiny little kernel seems both inefficient and
unlikely to be accurate (a tiny kernel having a lot of high-frequency
components), so the next step is to try to apply the kernel in the
image domain and see how well it performs.

# Again...

If I try using a small kernel (`test_decon_kernel`), it locally
sharpens the image, but there is still some wider blurriness. I guess
this is because, while the weight of far-away pixels is low, there are
a lot of them, so they do meaningfully contribute. I will now try
going back to an FFT/large-size convolution filter approach, and see
if this works better...

# And again...

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