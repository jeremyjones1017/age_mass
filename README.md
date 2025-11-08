Age probability distribution is determined by:

* Take MIST mass track
* Interpolate them so that there is one point along the mass track for the given age_resolution in the star profile
* For each of these points, a gaussian weight is assigned based on the difference between each points L/T and the central L/T of the data point using the given uncertainties as sigma.
* A weighted histogram is generated based on all of these points - this forms the age probability distribution
* The final age estimate is the median of this distribution with 68% and 95% confidence intervals determined

Mass probability distribution is determined in an analogous way, but using MIST isochrones and interpolating them over a mass resolution

Provided here is the Star Profile for an example star. The MIST models will need to be downloaded and included in its own directory (age_mass/MIST_FeH+0.00/ for solar model)
