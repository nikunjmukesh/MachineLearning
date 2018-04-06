# Description : Identify the language of a given piece of text by testing them on n-gram models
# 	by generation of n-gram language models over characters

# Authors : Nikunj Kumar Patel (1543376) <nikunjku@ualberta.ca>,
#			Chirag Balakrishna (1559633) <cbalakri@ualberta.ca>, 
		 




"""

	For this assignment, n-gram language models for each of 55 different languages were generated from
	the Universal Declaration of Human Rights (UDHR). The likelihood of an every input n-gram being found in the
	training set was computed and the perplexity, the Nth root of the inverse of the probability(MLE), was computed over the total
	log probabilities. Three varitants of the language models can be generated with this implementation - Unsmoothed Model,
	Laplace Smoothed Model and Kneser Ney Smoothed Model. The models are generated over characters.

"""



from __future__ import division
import os
import sys
import nltk
from nltk.util import ngrams
import numpy as np
from collections import Counter
from collections import OrderedDict

input_ngrams_dict = OrderedDict()
train_ngrams_dict = OrderedDict()
training_text = OrderedDict()
train_prefix_dict = OrderedDict()
kneser_input = OrderedDict()
kneser_train = OrderedDict()

V = 0



def unsmoothed_ngram_probability(n, input_ngrams_dict, train_ngrams_dict, training_text, train_prefix_dict):

	"""

	The unsmoothed technique computes log probabilities of all n-grams generated from a piece of text.
	This function makes use of the Markov assumption that the probability of finding the nth character depends only
	only on the previous n-1 characters. 

	:param n: Specifies the size of the n-gram.

	:param input_ngrams_dict: A dictionary which stores n-grams genereated from a piece of text. In this case the key
		is the name of the file and the value is the list of its n-grams

	:param train_ngrams_dict: A dictionary which stores training n-grams with the same structure as input_ngrams_dict

	:param training_text: Stores training data as text. Useful for computing the length of the training set.

	:param train_prefix_dict: This dictionary stores n-1 grams for the training dataset in order to accomodate 
		the Markov assumption udring the calculation of Maximum Likelihood Estimation. Here key is the file name 
		and value is list of n-grams generated from that file.

	"""

	for input_key in input_ngrams_dict.keys():
		#for every input n-gram
		lowest_perplexity = None
		result_lang = None

		for train_key in train_ngrams_dict.keys():
			#and for every training n-gram

			perplexity = 0
			log_probability = 0
			log_probability_list = []
			chain_prob = 0
			probability_dic = {}
			inf_count = 0

			freq_dist_train_ngrams = nltk.FreqDist(train_ngrams_dict[train_key])
			freq_dist_train_prefix = nltk.FreqDist(train_prefix_dict[train_key])

			if (n != 1):
				for input_ngrams in input_ngrams_dict[input_key]:

					count_input = freq_dist_train_ngrams.freq(input_ngrams)
					count_prefix = freq_dist_train_prefix.freq(input_ngrams[0:n-1])  #computes number of n-1 grams 


					if((count_prefix != 0)):
						log_probability += (np.log(count_input/count_prefix))
					else:
						continue

						
				perplexity = (1/np.exp(log_probability))**(1/len(train_ngrams_dict[train_key]))
				print input_key.split("=")[0]+"\t"+train_key+"\t"+str(perplexity)+"\t"+str(n)


			else:
				for input_ngrams in input_ngrams_dict[input_key]:
					count_input = freq_dist_train_ngrams.freq(input_ngrams)
					
					log_probability += np.log(count_input) 

				
				perplexity = (1/np.exp(log_probability))**(1/len(input_ngrams_dict[input_key]))


			if lowest_perplexity == None:
				lowest_perplexity = perplexity
				result_lang = train_key
			elif lowest_perplexity > perplexity:
				lowest_perplexity = perplexity
				result_lang = train_key
			else:
				continue

			print str(input_key)+"\t"+str(result_lang)+"\t"+str(perplexity)+"\t"+str(n)
				# print input_key+"\t"+train_key+"\t"+str(perplexity)+"\t"+str(n)




def laplace_smoothing(n, input_ngrams_dict, train_ngrams_dict, training_text, train_prefix_dict):

	"""

	The laplace smoothing simply adds one to the counts of n-grams. This reduces the number of 
	n-grams with zero counts and then the MLE is computed.
	
	:param n: Specifies the size of the n-gram.

	:param input_ngrams_dict: A dictionary which stores n-grams genereated from a piece of text. Here the key
		is the name of the file and the value is the list of its n-grams.

	:param train_ngrams_dict: A dictionary which stores training n-grams with the same structure as input_ngrams_dict. Here 
		the key is the name of the file and the value is the list of its n-grams.

	:param training_text: Stores training data as text. Useful for computing the length of the training set.

	:param train_prefix_dict: This dictionary stores n-1 grams for the training dataset in order to accomodate 
		the Markov assumption udring the calculation of Maximum Likelihood Estimation. Here key is file name 
		and value is list of n-grams generated from that file.
	
	"""

	for input_key in input_ngrams_dict.keys():
		#for every input input file
		lowest_perplexity = None
		result_lang = None
		for train_key in train_ngrams_dict.keys():
			#and for every training file
			perplexity = 0
			log_probability = 0

			freq_dist_train_ngrams = nltk.FreqDist(train_ngrams_dict[train_key])
			freq_dist_train_prefix = nltk.FreqDist(train_prefix_dict[train_key])

			V = len(set(training_text[train_key]))

			for input_ngrams in input_ngrams_dict[input_key]:

				count_input = train_ngrams_dict[train_key].count(input_ngrams) + 1 
				count_prefix = train_ngrams_dict[train_key].count(input_ngrams[0:n-1]) + V #computes number of n-1 grams.

				log_probability += np.log(count_input/count_prefix)

			perplexity = (np.exp(log_probability))**(-1/len(input_ngrams_dict[input_key]))


			if lowest_perplexity == None:
				lowest_perplexity = perplexity
				result_lang = train_key
			elif lowest_perplexity > perplexity:
				lowest_perplexity = perplexity
				result_lang = train_key
			else:
				continue

			print str(input_key)+"\t"+str(result_lang)+"\t"+str(perplexity)+"\t"+str(n)

			# print input_key+"\t"+train_key+"\t"+str(perplexity)+"\t"+str(n)




def kneser_ney_smoothing(kneser_train, kneser_input):

	"""
	The Kneser-Ney smoothing technique computes the probability of a trigram given its prefix. 
	The Kneser-Ney technique works only for trigrams and makes use of the KneserNeyProbDist() 
	function to train on the training data.

	:param kneser_train: A dictionary of training data consisting of trigrams.

	:param kneser_input: A dictionary of input data consisting of trigrams. 
	"""

	for input_key in kneser_input.keys():

		lowest_perplexity = None
		result_lang = None

		for train_key in kneser_train.keys():

			probability = 1
			perplexity = None
			l = []

			freq_dist_train = nltk.FreqDist(kneser_train[train_key])
			kneser_ney_train = nltk.KneserNeyProbDist(freq_dist_train, bins=None, discount=0.75)


			for input_ngrams in kneser_input[input_key]:
				
				prob_kn = kneser_ney_train.prob(input_ngrams)
				if(prob_kn == 0):
					prob_kn = 0.1
				
				probability *= prob_kn


			perplexity = probability**(-1/len(kneser_input[input_key]))

			if lowest_perplexity ==None:
				lowest_perplexity = perplexity
				result_lang = train_key
			elif lowest_perplexity > perplexity:
				lowest_perplexity = perplexity
				result_lang = train_key
			else:
				continue

			print str(input_key)+"\t"+str(result_lang)+"\t"+str(perplexity)+"\t"+str(n)
			# print input_key+"\t"+train_key+"\t"+str(perplexity)+"\t"+str(n)



def character_training_data(n, train_folder):

	"""
	This function reads all files in a folder containing training data and 
	stores them in a dictionary where the key is the name of the file and
	the value is the list of n-grams generated from particular file.


	:param n: The N-Gram parameter.

	:param train_folder: Path to the folder containing training data. 
	"""


	for train_filename in sorted(os.listdir(train_folder)):

		language_string = ' '


		train_file = open(train_folder+str(train_filename), 'r')

		for train_line in train_file:
			language_string += train_line.strip()

		language_string = language_string+' '

		train_ngrams = nltk.ngrams(language_string, n, pad_left=True, pad_right=True, right_pad_symbol=' ')
		train_ngrams_dict[train_filename] = list(train_ngrams)

		kneser_ngrams = nltk.ngrams(language_string, 3, pad_left=True, pad_right=True, right_pad_symbol=' ')
		kneser_train[train_filename] = list(kneser_ngrams)

		train_prefix_ngrams = nltk.ngrams(language_string, n-1, pad_left=True, pad_right=True, right_pad_symbol=' ')
		training_text[train_filename] = language_string
		train_prefix_dict[train_filename] = list(train_prefix_ngrams)



def character_input_data(n, input_folder):

	"""
	This function reads all files in a folder containing input files for which the language
	is to be identified and stores them into a dictionary where key is file name and the value 
	is the list of n-grams generated from that file.

	:param n: The N-Gram parameter.

	:param input_folder: Path to the folder containing input data. 
	"""

	for input_filename in sorted(os.listdir(input_folder)):

		input_string = ' '

		input_file = open(input_folder+str(input_filename), 'r')

		for input_line in input_file:
			input_string += input_line.strip()

		input_string = input_string+' '

		input_ngrams = nltk.ngrams(input_string, n, pad_left=True, pad_right=True, right_pad_symbol=' ')
		input_ngrams_dict[input_filename] = list(input_ngrams)

		_kneser_input_ngrams = nltk.ngrams(input_string, 3, pad_left=True, pad_right=True, right_pad_symbol=' ')
		kneser_input[input_filename] = list(_kneser_input_ngrams)




if __name__ == "__main__":

	if len(sys.argv) < 2:
		print '\nUsage : '
		print 'python langid.py --[smoothing_methods] [n-gram parameter]\n'
		print 'Smoothing Methods available: \n'
		print '\t1. Laplacian -> Usage --laplace\n\t2. Kneser Ney -> Usage --kneser-ney\n\t3. Unsmoothed -> Usage --unsmoothed\n'
		print 'Example : python langid.py --laplace 3\n'

	else:
		smoothing_method = sys.argv[1]

		n = int(sys.argv[2])

		train_folder = "811_a1_train/"
		input_folder = "811_a1_dev/"

		character_input_data(n, input_folder)
		character_training_data(n, train_folder)

		if(smoothing_method == '--unsmoothed'):
			print "\n\n-------Results for Unsmoothed-------\n\n\n"
			unsmoothed_ngram_probability(n, input_ngrams_dict, train_ngrams_dict, training_text, train_prefix_dict)
			
		elif(smoothing_method == '--laplace'):
			print "\n\n-------Results for Laplace Smoothing-------\n\n\n"
			laplace_smoothing(n, input_ngrams_dict, train_ngrams_dict, training_text, train_prefix_dict)
			
		elif(smoothing_method == '--kneser-ney'):
			print "\n\n-------Results for Kneser-Ney Smoothing-------\n\n\n"
			kneser_ney_smoothing(kneser_train, kneser_input)
