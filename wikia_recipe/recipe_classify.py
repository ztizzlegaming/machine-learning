# Helper program for recipe data
# Pulls out COUNT of the recipes, strips all of the nontext characters,
# and allows the user to classify them as a recipe or not. Then writes
# them back to the file data.txt
# 
# Jordan Turley

import io
import random
import re

FILENAME = 'cookbook_import_pages_current.xml'
COUNT = 100

# Page title parts that imply it is not a recipe
EXCLUDE = [
	'User blog',
	'User blog comment',
	'Recipes Wiki talk',
	'Board Thread',
	'Turkey Talk',
	'MediaWiki talk',
	'Help talk',
	'Blog',
	'Category',
	'Thread',
	'User talk',
	'Forum',
	'Help',
	'Portal',
	'Talk',
	'File talk',
	'Category talk',
	'Board',
	'User',
	'Template',
	'Message Wall',
	'MediaWiki',
	'File',
	'Recipes Wiki'
]

# The number of content pages. There are something like 150k pages in all but
# we only want to select from the content pages.
PAGES = 50481

def main():
	f = io.open(FILENAME, mode = 'r', encoding = 'utf-8')

	nums = random.sample(range(1, PAGES), COUNT)

	recipes = getrecipesRaw(f, nums)

	f.close()

	# Go through each recipe and ask the user to classify it
	classifications = []
	for i1 in range(len(recipes)):
		recipe = recipes[i1]
		print(recipe)
		print((i1 + 1), 'Is this a recipe? y or n')
		resp = input()
		while resp != 'y' and resp != 'n':
			print('Is this a recipe? y or n')
			resp = input()
		classifications.append(resp)
		print('-' * 160, '\n\n\n')

	# Go through the recipes and format each recipe
	for i1 in range(len(recipes)):
		# Start by replacing \n, [[, ]], -, and : with just a space
		recipes[i1] = recipes[i1].replace('\n', ' ')
		recipes[i1] = recipes[i1].replace('[[', ' ')
		recipes[i1] = recipes[i1].replace(']]', ' ')
		recipes[i1] = recipes[i1].replace('-', ' ')
		recipes[i1] = recipes[i1].replace(':', ' ')

		# Only get the content of the <text> tag
		recipes[i1] = re.findall(r'<text.*?>(.*?)<\/text>', recipes[i1])[0]

		# Remove all non-alphanumeric characters
		recipes[i1] = re.sub(r'[^a-zA-Z0-9_ \']+', '', recipes[i1])

		# Remove excess whitespace
		recipes[i1] = ' '.join(recipes[i1].split())

	# Write everything back to a file to be used later
	f = open('data.txt', 'w')
	f.write(str(COUNT) + '\n')
	for i1 in range(len(recipes)):
		f.write(classifications[i1] + '\n')
		f.write(recipes[i1] + '\n')
	f.close()

def getrecipesRaw(f, nums):
	'''Gets the raw xml pages at the indexes in nums'''
	recipes = []

	recipe = ''
	add = False

	curCount = 0

	for line in f:
		# Strip off leading/trailing whitespace
		line_ = line.strip()

		# Add the line to the current recipe
		if add:
			recipe += line

			# Try to find the title tag
			match = re.findall('<title>(.*?)</title>', line)
			if match:
				match = match[0]
				parts = match.split(':')

				# Check if the part before the ':' is in EXCLUDE
				if parts[0] in EXCLUDE:
					# This is not a content page
					recipe = ''
					add = False

		# Beginning of a new page
		if line_ == '<page>':
			if add:
				print('ERROR')
				input()
			recipe += line
			add = True
		elif line_ == '</page>' and add: # Ending of a page
			# This is a recipe
			curCount += 1
			if curCount in nums: # Check if we want this page, if it's in nums
				recipe = str(recipe.encode('utf-8')).replace('\\n', '\n')
				recipe = recipe[2:len(recipe) - 1]

				recipes.append(recipe)

			recipe = ''
			add = False

	return recipes

if __name__ == '__main__':
	main()