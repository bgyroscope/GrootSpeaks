'''module to test various example corpus '''


def split_and_cycle(text):
    tarr = text.split()
    tarr = tarr[-1:] + tarr[:-1]

    return ' '.join(tarr)

def cycle_through(text,nrepeat=1):
    ncycle = len(text.split()) * nrepeat
    arr = ["" for i in range(ncycle) ] 
    for i in range(ncycle):
        arr[i] = text
        text = split_and_cycle(text)

    return arr

def text_full(text_arr, nrepeat=1):
    return [x for txt in text_arr for x in cycle_through(txt, nrepeat)]



# I am groot example 
iamgroot_text = "I am Groot. I am Groot. I am Groot."
iamgroot = cycle_through(iamgroot_text, 100)

# See spot run example
spot_basic = ["See spot run.", 
              "I see spot run.", 
              "Did you see spot run? I see spot run."]
spot = text_full(spot_basic, 20)



# Text 1 through 3
text_1 = 'The quick brown fox jumps over the lazy dog.'
text_2 = 'My dog is quick and can jump over fences.'
text_3 = 'Your dog is so lazy that it sleeps all the day.'
ex_1 = [text_1, text_2, text_3]

# example text array 
text_arr = [ 
    "Dog barks at the cat.", 
    "The cat meows.", 
    "The cat meows.", 
    "The cat meows.", 
    "Cat chases the mouse." , 
    "The dog barks.", 
    "The mouse squeaks.", 
    "The dog and the cat are friends.", 
    "Mouse is afraid of the cat.", 
    "The dog and the cat play together.", 
    "The mouse does not play with the cat.", 
    "The dog does play with the cat.",  
    "The cat does not bark.",
    "The dog does not meow.", 
    "The dog barks. Look at the dog barking. The dog barks. The dog barks a lot of the time. I am watching the dog bark.",
    "The dog barks.", 
    "The dg barks.", 
    "The dg barks.", 
    "The dg and the cat are friends.", 
    "The dog barks.", 
]

