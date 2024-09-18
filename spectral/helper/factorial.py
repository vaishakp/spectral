fact_dict = {0: 1, 1: 1}

def factorial(number):
    if number not in fact_dict.keys():
        fact_dict.update({number: factorial(number - 1) * number})

    return fact_dict[number]