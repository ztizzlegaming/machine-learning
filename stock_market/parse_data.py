# Parses all of the given files in ./data/, using the closing price as the
# price of interest, and determining if the stock price went up or down in the
# last DAYS days, set in the constant.
# 
# Jordan Turley

import os

DAYS = 1

def main():
	# Move into the data directory and fetch all the filenames
	os.chdir('data_original')
	files = os.listdir('./')
	os.chdir('../')

	# Count how many increase and how many decrease
	increase = 0

	for filename in files:
		if filename == '.DS_Store': # Ignore .DS_Store
			continue

		stock = filename[:-4]
		f = open('data_original/' + filename)
		#f = open(filename)

		# Read the lines from the file
		lines = f.readlines()
		f.close()

		del lines[0] # Remove first line, this is the header

		lastPrice = None

		prices = []
		percentChanges = []
		for line in lines:
			# Get each part of each line: open, high, low, ...
			parts = line.split(',')
			date = parts[0]
			openPrice = parts[1]
			highPrice = parts[2]
			lowPrice = parts[3]
			closePrice = float(parts[4]) # This is the one we care about, parse as float
			adjClosePrice = parts[5]
			volume = parts[6]

			if lastPrice == None:
				lastPrice = closePrice
			else:
				# Calculate percent change from last day
				change = closePrice - lastPrice
				percentChange = (change / lastPrice) * 100
				percentChanges.append(percentChange)
				lastPrice = closePrice

			prices.append(closePrice)

		# Get the last price
		lastPrice = prices[-1]
		
		# Get the last price we 'see'
		lastTrainingPrice = prices[-(DAYS + 1)]

		# See if the stock has gone up or down
		upDown = 'decrease'
		if lastPrice > lastTrainingPrice:
			upDown = 'increase'
			increase += 1

		# Chop off everything up to the last training price
		prices = prices[:-DAYS]
		#print(prices[-5:])
		#print(prices[:-5])
		#percentChanges = percentChanges[:-DAYS]

		outFile = open('data_1_day/' + stock + '.txt', 'w')
		#outFile = open('temp.txt', 'w')
		
		for price in prices:
			outFile.write(str(price) + '\n')

		#for change in percentChanges:
		#	outFile.write(str(change) + '\n')

		outFile.write(upDown + '\n')

		outFile.close()

	print('Increase:', increase)
	print('Decrease:', 500 - increase)

if __name__ == '__main__':
	main()