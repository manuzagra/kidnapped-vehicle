/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include <helper_functions.hpp>
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <type_traits>


using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
	// normal distributions for the noise in the initialization
	std::normal_distribution<double> noise_x{0,std[0]};
	std::normal_distribution<double> noise_y{0,std[1]};
	std::normal_distribution<double> noise_theta{0,std[2]};


	particles = std::vector<Particle>(num_particles);
	for (unsigned int i=0; i<num_particles; ++i){
		particles[i].id = i;
		particles[i].x = x + noise_x(rand_eng);
		particles[i].y = y + noise_y(rand_eng);
		particles[i].theta = theta + noise_theta(rand_eng);
		particles[i].weight = 1;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	// normal distributions for the noise in the movement
	// this is going to be constant, it would be more efficient to do it in the constructor
	std::normal_distribution<double> noise_x{0,std_pos[0]};
	std::normal_distribution<double> noise_y{0,std_pos[1]};
	std::normal_distribution<double> noise_theta{0,std_pos[2]};

	if(fabs(yaw_rate)>0.0001){
		// yaw_rate != 0
		auto move = [this, delta_t, std_pos, velocity, yaw_rate, &noise_x, &noise_y, &noise_theta](Particle &p){
			double x = p.x + velocity/yaw_rate*(sin(p.theta+yaw_rate*delta_t)-sin(p.theta)) + noise_x(this->rand_eng);
			double y = p.y + velocity/yaw_rate*(cos(p.theta)-cos(p.theta+yaw_rate*delta_t)) + noise_y(this->rand_eng);
			double theta = p.theta + yaw_rate*delta_t + noise_theta(this->rand_eng);
			p.x = x;
			p.y = y;
			p.theta = theta;
		};

		std::for_each(particles.begin(), particles.end(), move);
	}else{
		// yaw_rate ~= 0
		auto move = [this, delta_t, std_pos, velocity, yaw_rate, &noise_x, &noise_y, &noise_theta](Particle &p){
			double x = p.x + velocity*delta_t*cos(p.theta) + noise_x(this->rand_eng);
			double y = p.y + velocity*delta_t*sin(p.theta) + noise_y(this->rand_eng);
			double theta = p.theta + noise_theta(this->rand_eng);
			p.x = x;
			p.y = y;
			p.theta = theta;
		};

		std::for_each(particles.begin(), particles.end(), move);
	}

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

	auto find_similar = [&predicted](LandmarkObs &observation){
		// calculate the distance of every predicted to our observation and save it in distances
		vector<double> distances(predicted.size());
		std::transform(predicted.begin(), predicted.end(), distances.begin(),
				[&observation](LandmarkObs pred){
					return dist(observation.x, observation.y, pred.x, pred.y);
				}
		);
		// check for the minimum
		unsigned int min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
		// set the id of the observation to the id of the prediction
		observation.id = predicted[min_index].id;
	};

	// do the process in every observation
	std::for_each(observations.begin(), observations.end(), find_similar);

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */



	// this function is going to run for every particle in this->particles
	auto update_single_weight = [sensor_range, &std_landmark, &observations, &map_landmarks](Particle &p) mutable{

		// make a copy of observations
		vector<LandmarkObs> measurements(observations);

		auto SingleLandmark_to_LandmarkObs = [](const Map::single_landmark_s &map_landmark) -> LandmarkObs{
			return LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f};
		};
		vector<LandmarkObs> landmarks(map_landmarks.landmark_list.size());
		std::transform(map_landmarks.landmark_list.begin(), map_landmarks.landmark_list.end(), landmarks.begin(), SingleLandmark_to_LandmarkObs);


		// Only landmarks within the sensor_range will be used
		auto out_range = [sensor_range, &p](LandmarkObs &lm) -> bool{
			return (dist(p.x, p.y, lm.x, lm.y) > sensor_range);
		};
		std::remove_if(landmarks.begin(), landmarks.end(), out_range);


		// homogeneous transformation of the coordinates of the measurements
		auto to_map_coordinate_and_find_imilar = [&p, &landmarks](LandmarkObs &obs){
			// transform to map coordinate
			double x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
			double y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
			obs.x = x;
			obs.y = y;

			// associate every observation with a landmark
			// calculate the distance of every predicted to our observation and save it in distances
			vector<double> distances(landmarks.size());
			std::transform(landmarks.begin(), landmarks.end(), distances.begin(),
					[&obs](LandmarkObs pred){
						return dist(obs.x, obs.y, pred.x, pred.y);
					}
			);
			// check for the minimum
			unsigned int min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
			// set the id of the observation to the id of the prediction
			obs.id = landmarks[min_index].id;
		};
		std::for_each(measurements.begin(), measurements.end(), to_map_coordinate_and_find_imilar);


		// calculate the individual weight of every observation
		auto observation_weight = [&landmarks, &std_landmark](LandmarkObs &obs){
			// multivariate normal distribution to calculate the weight
			auto multivariate_normal_distribution = [](double mu_x, double mu_y, double std_x, double std_y, double x, double y){
			  // calculate normalization term
			  double gauss_norm = 1 / (2 * M_PI * std_x * std_y);
			  // calculate exponent
			  double exponent = (pow(x - mu_x, 2) / (2 * pow(std_x, 2)))
						   + (pow(y - mu_y, 2) / (2 * pow(std_y, 2)));
			  // calculate weight using obs.normalization terms and exponent
			  double weight = gauss_norm * exp(-exponent);
			  return weight;
			};

			// find the landmark with given id
			LandmarkObs related_lm = *std::find_if(landmarks.begin(), landmarks.end(), [&obs](LandmarkObs lm){return (obs.id == lm.id);});
			// return the probability
			return multivariate_normal_distribution(related_lm.x, related_lm.y, std_landmark[0], std_landmark[1], obs.x, obs.y);
		};
		vector<double> individual_weights(measurements.size());
		std::transform(measurements.begin(), measurements.end(), individual_weights.begin(), observation_weight);

		// update the weight of the particle
		p.weight = std::accumulate(individual_weights.begin(), individual_weights.end(), 1., std::multiplies<double>());
	};
	std::for_each(particles.begin(), particles.end(), update_single_weight);

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	// get the weights
	vector<double> weights(num_particles);
	std::transform(particles.begin(), particles.end(), weights.begin(),
			[](Particle &p){return p.weight;}
	);
	// using std::discrete_distribution to get the new particles
    std::discrete_distribution<> random_index_generator(weights.begin(), weights.end());

    std::vector<Particle> new_particles(num_particles);
    std::for_each(new_particles.begin(), new_particles.end(),
    		[this, &random_index_generator](Particle &p){
    			p = this->particles[random_index_generator(this->rand_eng)];
    		}
	);
    particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
