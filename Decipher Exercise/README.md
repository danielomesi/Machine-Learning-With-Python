# Deciphering a Secret Code

## Task Goal
The goal of this task is to decipher a secret code using the partially-known rules of the cipher. As an example, the below phrase is encoded using the cipher:

**Encoded Phrase:**  
hexe wgmirgi mw jyr

We know that the ciphering process works as follows: given an English phrase and a secret number K, replace each character (except spaces) with another one that is located K characters further away in the English ABC. 

**Example:**  
In the above example, we found that K is 4 and the deciphered phrase is:  
data science is fun

## Files Provided
You are given two files:  
- config-decipher.json: Contains a secret phrase, top-10K most frequent words in the English lexicon, and the English ABC. You can change these locations in config-decipher.json, but do not make any changes to the main() code.
- decipher.py: Implement the decipher() function in this file. The function should print the value of K and the deciphered (original) phrase:
