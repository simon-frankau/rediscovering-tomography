# Learning to redisocver tomography

I know nothing about Radon transforms beyond the name, and the vague
idea of how tomography works, piecing together the original structure
from scans through it. However, it's always fascinated me, and since
having a CAT scan for the first time a few months ago, the idea's
returned to me:

*Wouldn't it be interesting to implement this, and create my own tool
that can bcak out a structure from scans?*

My initial thought was that I'd read up on the maths, and implement
it. However, it seems more fun to try to work it out for myself, and
then check my workings afterwards, so this is what I intend to do
here.

## Problem statement

The idea is that we're shooting a beam through an object at a bunch of
angles from a bunch of starting locations (in our case, in a 2D
plane), collecting how much of the beam makes it through, and using
that to infer the amount of absorption at each point within that
object.

What we're actually going to do here is take an image, pretend it's an
object, work out what the beams would do, and then take that data and
feed it through our own transform to try to infer the image that we
just fired beams through.

I think a bonus project is to see how stable it is - add noise to the
beam data, and see how much noise it adds to the inferred picture.

Somewhere near the end, I'll probably look up the actual theory for
these transforms and see how it compares.

## My cheap theory

When you fire a beam through a uniform medium, for every distance it
travels through the material, a certain percentage of the beam is
knocked out. I don't know if this is precisely physically true, but
I'm going to pretend it is. In other words, the medium exponentially
attenuates the beam.

So, let's take the log of that. The amount of exponential attenuation
then becomes the exponential of the sum of the logs of the attenuation
factors for all the points along the path. In other words, we can
integrate the (log'd) factors along the path to get the numbers we
need.

In the discrete image domain, each beam is a weighted sum of pixels.
This is a linear transform. Going from log'd input image to log'd beam
attenuations is simply a matrix multiplication. And, assuming that
matrix is of the right shape and non-singular, we can invert it!

This means that we don't need to develop any clever algebraic theory
for the transform - we can just throw a computer at it and see what
the result is. It might be horribly unstable, but we can also play
about with it to see if this is the case, empirically.

Dumb fun!

## My theory of doing things right

Note that the above approach would also work for Fourier transforms -
to do an inverse Fourier transform, we could "just" work out a forward
transform matrix, and then invert it. Of course, in practice we don't
because algebra tells us it's self-inverting.

I'm skipping the algebra as I'm doing the cheap approach mentioned
above, but it's still fun to think how I might approach it. The
problem is fundamentally, given a function *A* that calculates log
attenuation in terms of an integral of a log density function *D* along a
line, where *A* is parameterised in terms of start point and
direction, and *D* in terms of location, invert this to find the
function *D* in terms of *A*.

To make use of symmetry, I'd probably put the location parameters for
*D* as polar coordinates, so that the inverse function can be
calculated purely in terms of radius (being rotationally symmetric).

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

Anyway, I'm not doing this right now, I'm doing the matrix inversion
version. :)

## Implementation

I'm not bothering to do the whole log/exp thing on the input and
output data. It doesn't really add anything as far as I care. I'm
using `nalgebra` to do the matrix inversion, and `image` to load and
save the images being transformed.

To calculate the weights along the paths as they traverse the pixels,
I'm doing something akin to old-school scan conversion. I'm making my
own vectors for this, as using a full 2D vector class just feels like
overkill. The model is that the pixels are solid squares, and the line
is of zero width, and we want to know how long the line segment that
crosses each pixel is.

I then use that to build the data for sets of parallel rays at a bunch
of different angles. If I plot this data as an image, you can see
sinusoids that represent a chunk of matter rotating round. The phase
and amplitude represent the angle and radius of that piece of mass.
(Perhaps this implies a smarter algorithm for inverting the
transform?)

Finding the pseudo-inverse of the matrix and applying it... works!
Hurrah. Interestingly, the size of the error seems to vary quite a lot
with the number of path samples, so this looks like it'll be worth
some investigation.

`make demo` should exercise the image-to-transform-and-back cycle.

It's pretty darn slow, since it performs a matrix inverse on a matrix
whose dimensions are the width-height product of the image, and
rays-angles product of the scan, respectively. Images much bigger than
30x30 get pretty slow on my laptop.

## Convergence

I tried modifying the number of rays and angles to see how quickly the
reconstruction converges on the original. I know this a bit of a
rabbit-hole for me, based on previous numerical code projects, so I've
done a bit of investigation and drawn a line under it! I should only
invest so much time in empirical studies of convergence.

I ran a bunch of conversions, and put the results in [this
sheet](https://docs.google.com/spreadsheets/d/1mdHI4n2HNloAuYGf7aMaZISQnpawe0drygIFAd0lKsA/edit?usp=sharing).
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

# TODOs

 * Investigate other algorithms for reconstruction?
 * Also, it'd be nice if the tools had decent error handling. It's
   kind of optional for this toy-level tool, but it's good practice,
   right?
 * Read up the official approach! :p
 * Restructure the README once I'm done to make sense?
