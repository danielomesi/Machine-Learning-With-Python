import json

"""
takes a list of words and an integer k,
and subtracts k from the numeric value of each alphabetical letter into a new list of words.
If all the new words exist in the lexicon, returns true and the new list of words.
else, returns false.
"""
def try_decipher(list_of_words_to_decipher : str, k : int, lexicon : list):
    modified_list=[]
    is_deciphered=True
    for word in list_of_words_to_decipher:
        modified_word = ""
        for char in word:
            #substracing k from the numeric value of char, and normalizing it to the range of [0,25] so it would be a char
            modified_char = chr((ord(char) - ord('a') - k)%26+ord('a'))
            modified_word += modified_char
        if modified_word not in lexicon: #if we get one word that is not in the lexicon, there is no point to continue
            is_deciphered=False
            break
        modified_list.append(modified_word)

    #joining the list of words into a sentence
    deciphered_phrase=' '.join(modified_list)

    return (is_deciphered,deciphered_phrase)

""""
This function is trying to decipher the phrase,
returning a result as a map containing: status of success, original phrase, and decipher integer if succeeded
"""
def decipher_phrase(phrase, lexicon_filename, abc_filename):
    #consts for the statuses of deciphering the phrase and a const for a case when K is not found
    DECIPHERED=0
    NOT_DECIPHERED=1
    EMPTY=2
    NOT_FOUND=-1

    print(f'starting deciphering using {lexicon_filename} and {abc_filename}')

    # until the phrase is not deciphered, we define it as not deciphered
    status=NOT_DECIPHERED
    k=NOT_FOUND
    original_phrase=""

    #opening the file to use as our vocabulary
    with open(lexicon_filename,'r', encoding='utf-8') as fin:
        lexicon_string=fin.read()

    lexicon=lexicon_string.split()
    undeciphered_phrase_as_list=phrase.split()

    #checking whether there is nothing to decipher
    if len(undeciphered_phrase_as_list)==0:
        status=EMPTY
    else:
        for i in range(26):
            is_deciphered, original_phrase = try_decipher(undeciphered_phrase_as_list, i, lexicon)
            if is_deciphered:
                k=i
                status=DECIPHERED
                break



    result = {"status": status, "orig_phrase": original_phrase, "K": k}
    return result


# todo: fill in your student ids
students = {'id1': '207689092', 'id2': '208538488'}

if __name__ == '__main__':
    with open('config-decipher.json', 'r') as json_file:
        config = json.load(json_file)

    result = decipher_phrase(config['secret_phrase'],
                             config['lexicon_filename'],
                             config['abc_filename'])

    assert result["status"] in {0, 1, 2}

    if result["status"] == 0:
        print(f'deciphered phrase: {result["orig_phrase"]}, K: {result["K"]}')
    elif result["status"] == 1:
        print("cannot decipher the phrase!")
    else:  # result["status"] == 2:
        print("empty phrase")
