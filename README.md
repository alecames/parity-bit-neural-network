# neural-network
Learns to predict the parity bit for a set of four bits.

	# Getting user parameters: 
		
		The program will check and handle most user errors that I could imagine running into.
		Additionally, if you want to test the program a few times with the same values
		[ENTER] or [RETURN] will automatically fill in some example values. They are printed at the bottom 
		of each training session, except for the training size, but that can be inferred from the 'success' tally.
		The values are:  hidden layer = 8, iterations = 10k, learning rate = 0.15, training size = 10.

		The split of values between training testing is set to 10/6, but it can be changed in the .py version 
		with the variable 'training_size' if desired

	1. Enter the hidden layer count from 1 to 256 (capped at 256, large numbers slow it down quite a bit)
	2. Enter learning rate between 0.0 - 1.0 (recommended values are 0.1 to 0.8 but I allowed for 0 and 1 for more options)
	3. Enter maximum number of iterations from 0 to 2^20 (allowed 0 incase wanted to see how the random weights would perform, and capped at ~1m iterations)

	# Training:	

	Next, the program will run through each iteration while printing the epoch#, the mean squared error 
	(MSE) and the current correct guesses the network has made so far. The data is also in a random order 
	for each run. What follows next are the training results, which includes a printout of the user entered node count and 
	learning rate, as well as the final MSE. Underneath is the data used to train, the expected output for each
	training example and the networks prediction, as well as the rounded prediction (0 or 1).
	
	The program will then prompt the user to either continue to test the network on the rest of the untouched 
	training sample. If answered 'no', it will then ask if the user would like to restart the training (ie. 
	restart the program). If not the program will close. Note: the .exe variant won't prompt to restart the program,
	it must be run again manually if needed

	# Testing: 

	If answered 'yes', it will proceed to the next step in which it wil test the remaining 6 4-bit permutations
	with the trained neural network and print the mean squared error of the networks predictions as well. 

	Finally, it prompts the user to decide to exit or restart training with the choice to change the training parameters.
